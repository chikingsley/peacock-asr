import { existsSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";

export const LANGS = ["dari", "farsi", "georgian", "russian", "tajik"] as const;
export type Lang = (typeof LANGS)[number];

// russian is a provided-text dataset (no youtube videos/clips funnel) -> verify only.
export const VERIFY_ONLY: ReadonlySet<Lang> = new Set(["russian"]);

const DATA_ROOT = "/home/simon/github/peacock-asr/projects";
const WORKER = fileURLToPath(new URL("./db-worker.ts", import.meta.url));
const YTDLP_IMAGE = "jauderho/yt-dlp:latest";

export const DISKS = [
  { label: "overflow", mount: "/mnt/overflow" },
  { label: "storage", mount: "/mnt/storage" },
  { label: "media", mount: "/mnt/media" },
  { label: "fast-ssd", mount: "/mnt/fast-ssd-2tb" },
] as const;

// Subprocess timeouts. A pathological / locked sqlite read gets killed here and
// we keep the last good value rather than freezing.
const QUEUE_TIMEOUT_MS = 8000;
const CURATOR_TIMEOUT_MS = 10000;

// Poll cadences. Cheap stuff fast; the multi-GB curator full-table count slow
// (it only changes by hundreds/sec, so a few seconds stale is fine, and the
// russian count(scribe_wer) has NO index -> it is a 9.66M-row full scan).
const CHEAP_POLL_MS = 1500; // queue GROUP BYs, df, docker lanes
const CURATOR_POLL_MS = 6000; // curator counts
const FLAC_POLL_MS = 15000; // create/ FLAC counts (can be a multi-second walk)

const dataDir = (lang: Lang) => `${DATA_ROOT}/${lang}-asr/data`;

export interface LangStats {
  lang: Lang;
  verifyOnly: boolean;
  // download stage
  downloadActive: boolean;
  flacCount: number;
  // youtube funnel
  videosTotal: number;
  videosSegmented: number;
  clipsDone: number;
  clipsPending: number;
  // verify stage
  samplesTotal: number;
  samplesScored: number;
  // freshness (ms timestamps of last successful read, 0 = never)
  queueUpdatedAt: number;
  curatorUpdatedAt: number;
}

export interface DiskStats {
  label: string;
  mount: string;
  totalGb: number;
  freeGb: number;
  error: string | null;
}

// What the UI reads. Snapshot is cheap to build and never blocks.
export interface Snapshot {
  ts: Date; // when the snapshot was assembled (UI tick time)
  langs: LangStats[];
  disks: DiskStats[];
  disksUpdatedAt: number;
  // in-flight indicators (a slow read currently running)
  queuePolling: boolean;
  curatorPolling: boolean;
}

function usableDb(path: string): boolean {
  try {
    return existsSync(path) && statSync(path).size > 0;
  } catch {
    return false;
  }
}

// Run a command, capture stdout, kill after timeoutMs. Resolves null on
// timeout / nonzero exit / spawn error so a slow read can never hang us.
async function runText(cmd: string[], timeoutMs: number): Promise<string | null> {
  try {
    const proc = Bun.spawn(cmd, { stdout: "pipe", stderr: "ignore" });
    const killer = setTimeout(() => {
      try {
        proc.kill();
      } catch {}
    }, timeoutMs);
    const out = await new Response(proc.stdout).text();
    const code = await proc.exited;
    clearTimeout(killer);
    if (code !== 0) return null;
    return out.trim() || null;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Cache (mutated by background pollers, read by the UI tick)
// ---------------------------------------------------------------------------
const cache: Map<Lang, LangStats> = new Map(
  LANGS.map((lang) => [
    lang,
    {
      lang,
      verifyOnly: VERIFY_ONLY.has(lang),
      downloadActive: false,
      flacCount: 0,
      videosTotal: 0,
      videosSegmented: 0,
      clipsDone: 0,
      clipsPending: 0,
      samplesTotal: 0,
      samplesScored: 0,
      queueUpdatedAt: 0,
      curatorUpdatedAt: 0,
    } satisfies LangStats,
  ]),
);

let diskCache: DiskStats[] = DISKS.map((d) => ({
  label: d.label,
  mount: d.mount,
  totalGb: 0,
  freeGb: 0,
  error: "pending",
}));
let disksUpdatedAt = 0;
let queuePolling = false;
let curatorPolling = false;

// ---------------------------------------------------------------------------
// Individual reads (async, isolated)
// ---------------------------------------------------------------------------
async function readQueue(lang: Lang): Promise<void> {
  if (VERIFY_ONLY.has(lang)) return;
  const path = `${dataDir(lang)}/queue.sqlite`;
  if (!usableDb(path)) return;
  const out = await runText(["bun", WORKER, "queue", path], QUEUE_TIMEOUT_MS);
  if (!out) return;
  try {
    const q = JSON.parse(out);
    const v = (q.videos ?? {}) as Record<string, number>;
    const c = (q.clips ?? {}) as Record<string, number>;
    const s = cache.get(lang)!;
    if (Object.keys(v).length) {
      s.videosTotal = Object.values(v).reduce((a, b) => a + b, 0);
      s.videosSegmented = v.segmented ?? 0;
    }
    if (Object.keys(c).length) {
      s.clipsDone = c.done ?? 0;
      s.clipsPending = c.pending ?? 0;
    }
    s.queueUpdatedAt = Date.now();
  } catch {}
}

async function readCurator(lang: Lang): Promise<void> {
  const path = `${dataDir(lang)}/curator.sqlite`;
  if (!usableDb(path)) return;
  const out = await runText(["bun", WORKER, "curator", path], CURATOR_TIMEOUT_MS);
  if (!out) return;
  try {
    const row = JSON.parse(out);
    const s = cache.get(lang)!;
    s.samplesTotal = row.total ?? 0;
    s.samplesScored = row.scored ?? 0;
    s.curatorUpdatedAt = Date.now();
  } catch {}
}

async function readDisks(): Promise<void> {
  const out = await runText(["df", "-BG", ...DISKS.map((d) => d.mount)], 4000);
  if (!out) return;
  const next: DiskStats[] = DISKS.map((d) => ({
    label: d.label,
    mount: d.mount,
    totalGb: 0,
    freeGb: 0,
    error: "unknown",
  }));
  for (const line of out.split("\n").slice(1)) {
    const parts = line.trim().split(/\s+/);
    if (parts.length < 6) continue;
    const mount = parts[parts.length - 1];
    const total = parseInt(parts[1], 10);
    const avail = parseInt(parts[3], 10);
    const t = next.find((o) => o.mount === mount);
    if (t && !Number.isNaN(total)) {
      t.totalGb = total;
      t.freeGb = Number.isNaN(avail) ? 0 : avail;
      t.error = null;
    }
  }
  diskCache = next;
  disksUpdatedAt = Date.now();
}

// Download lanes: which langs have a yt-dlp container actively running.
// Container NAMES are docker-random (don't encode the lang), so we identify the
// lang from each container's bind-mount path, which is .../<lang>-asr/data/create/...
// Only the lang(s) actually downloading light up — no fallback-to-all.
async function readDownloadLanes(): Promise<void> {
  const ids = await runText(
    ["docker", "ps", "--filter", `ancestor=${YTDLP_IMAGE}`, "--format", "{{.ID}}"],
    4000,
  );
  const idList = ids ? ids.split("\n").filter(Boolean) : [];
  const activeLangs = new Set<string>();
  if (idList.length > 0) {
    const mounts = await runText(
      ["docker", "inspect", ...idList, "--format", "{{range .Mounts}}{{.Source}} {{end}}"],
      4000,
    );
    for (const line of mounts ? mounts.toLowerCase().split("\n") : []) {
      for (const lang of LANGS) {
        // match the project dir (.../<lang>-asr/...) OR the SSD roots (.../peacock-{create,clips}/<lang>/...)
        if (
          line.includes(`${lang}-asr`) ||
          line.includes(`peacock-create/${lang}`) ||
          line.includes(`peacock-clips/${lang}`)
        ) {
          activeLangs.add(lang);
        }
      }
    }
  }
  for (const lang of LANGS) {
    cache.get(lang)!.downloadActive = !VERIFY_ONLY.has(lang) && activeLangs.has(lang);
  }
}

// FLACs sitting in create/ (downloaded). Cheap file count, not a du.
async function readFlacCounts(): Promise<void> {
  for (const lang of LANGS) {
    if (VERIFY_ONLY.has(lang)) continue;
    const dir = `${dataDir(lang)}/create`;
    if (!existsSync(dir)) continue;
    const out = await runText(
      ["bash", "-c", `find ${JSON.stringify(dir)} -maxdepth 2 -name '*.flac' 2>/dev/null | wc -l`],
      8000,
    );
    if (out == null) continue;
    const n = parseInt(out, 10);
    if (!Number.isNaN(n)) cache.get(lang)!.flacCount = n;
  }
}

// ---------------------------------------------------------------------------
// Snapshot for the UI (synchronous, never blocks)
// ---------------------------------------------------------------------------
export function getSnapshot(): Snapshot {
  return {
    ts: new Date(),
    langs: LANGS.map((l) => ({ ...cache.get(l)! })),
    disks: diskCache.map((d) => ({ ...d })),
    disksUpdatedAt,
    queuePolling,
    curatorPolling,
  };
}

// One-shot collect for --once: run all reads to completion once.
export async function collectOnce(): Promise<Snapshot> {
  await Promise.all([
    readDisks(),
    readDownloadLanes(),
    readFlacCounts(),
    ...LANGS.map(readQueue),
    ...LANGS.map(readCurator),
  ]);
  return getSnapshot();
}

// ---------------------------------------------------------------------------
// Background pollers (live mode). Each loop awaits its own work then waits the
// cadence, so a slow read self-throttles instead of piling up.
// ---------------------------------------------------------------------------
function loop(fn: () => Promise<void>, everyMs: number, onState?: (busy: boolean) => void): void {
  const run = async () => {
    for (;;) {
      onState?.(true);
      try {
        await fn();
      } catch {}
      onState?.(false);
      await Bun.sleep(everyMs);
    }
  };
  run();
}

export function startPolling(): void {
  // Cheap, fast: queue counts + df + download lanes.
  loop(
    async () => {
      await Promise.all([readDisks(), readDownloadLanes(), ...LANGS.map(readQueue)]);
    },
    CHEAP_POLL_MS,
    (b) => {
      queuePolling = b;
    },
  );
  // Expensive, slow: curator full-table counts (no index on scribe_wer).
  loop(
    async () => {
      await Promise.all(LANGS.map(readCurator));
    },
    CURATOR_POLL_MS,
    (b) => {
      curatorPolling = b;
    },
  );
  // Filesystem walk for downloaded FLACs.
  loop(readFlacCounts, FLAC_POLL_MS);
}
