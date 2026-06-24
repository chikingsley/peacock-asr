import React, { useEffect, useState } from "react";
import { useKeyboard, useRenderer, useTerminalDimensions } from "@opentui/react";
import { getSnapshot, type Snapshot, type LangStats, type DiskStats } from "./data";
import { bar, diskColor, num, pct, rampColor } from "./format";

const TICK_MS = 1000; // UI render cadence (independent of data polling)
const WIDE_BARW = 13;
const NARROW_AT = 90; // < this many cols => compact one-line layout

function ageStr(updatedAt: number, now: number): string {
  if (!updatedAt) return "—";
  const s = Math.max(0, Math.round((now - updatedAt) / 1000));
  if (s < 1) return "now";
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  return `${m}m${s % 60}s ago`;
}

// ---------------------------------------------------------------------------
// WIDE layout
// ---------------------------------------------------------------------------
function Stage({
  label,
  detail,
  sub,
  fraction,
}: {
  label: string;
  detail: string;
  sub?: string;
  fraction: number;
}) {
  const p = fraction * 100;
  return (
    <box style={{ flexDirection: "column", width: 22, marginRight: 1 }}>
      <text fg="#94a3b8">{label}</text>
      <text fg="#e2e8f0">{detail}</text>
      {sub ? <text fg="#7dd3fc">{sub}</text> : null}
      <box style={{ flexDirection: "row" }}>
        <text fg={rampColor(p)}>{bar(fraction, WIDE_BARW)}</text>
        <text fg="#64748b"> {p.toFixed(0).padStart(3)}%</text>
      </box>
    </box>
  );
}

function Arrow() {
  return (
    <box style={{ justifyContent: "center", marginRight: 1 }}>
      <text fg="#475569">{"→"}</text>
    </box>
  );
}

function DownloadStage({ active, flacs }: { active: boolean; flacs: number }) {
  return (
    <box style={{ flexDirection: "column", width: 20, marginRight: 1 }}>
      <text fg="#94a3b8">Downloads</text>
      <text fg={active ? "#4ade80" : "#64748b"}>{active ? "● active" : "○ idle"}</text>
      <text fg="#cbd5e1">{`${num(flacs)} flac`}</text>
    </box>
  );
}

function cardTitle(s: LangStats, now: number): string {
  const queueStale = !s.verifyOnly && s.queueUpdatedAt > 0 && now - s.queueUpdatedAt > 4000;
  const curatorStale = s.curatorUpdatedAt > 0 && now - s.curatorUpdatedAt > 9000;
  const updating = queueStale || curatorStale;
  const freshTs = Math.max(s.queueUpdatedAt, s.curatorUpdatedAt);
  return (
    s.lang.toUpperCase() +
    (s.verifyOnly ? "  (datasets)" : "") +
    `   updated ${ageStr(freshTs, now)}` +
    (updating ? "  · updating…" : "")
  );
}

function WideCard({ s, now }: { s: LangStats; now: number }) {
  const title = cardTitle(s, now);

  if (s.verifyOnly) {
    const f = s.samplesTotal ? s.samplesScored / s.samplesTotal : 0;
    return (
      <box
        style={cardBox}
        title={title}
      >
        <Stage
          label="Verified"
          detail={`${num(s.samplesScored)} / ${num(s.samplesTotal)}`}
          fraction={f}
        />
        <box style={{ justifyContent: "center" }}>
          <text fg="#64748b">scored samples (near export)</text>
        </box>
      </box>
    );
  }

  const clipTotal = s.clipsDone + s.clipsPending;
  return (
    <box style={cardBox} title={title}>
      <DownloadStage active={s.downloadActive} flacs={s.flacCount} />
      <Arrow />
      <Stage
        label="Enqueued"
        detail={`${num(s.videosTotal)} videos`}
        fraction={s.videosTotal > 0 ? 1 : 0}
      />
      <Arrow />
      <Stage
        label="Segmented"
        detail={`${num(s.videosSegmented)} / ${num(s.videosTotal)} vid`}
        sub={`${num(clipTotal)} clips out`}
        fraction={pct(s.videosSegmented, s.videosTotal) / 100}
      />
      <Arrow />
      <Stage
        label="Clips labeled"
        detail={`${num(s.clipsDone)} / ${num(clipTotal)}`}
        fraction={pct(s.clipsDone, clipTotal) / 100}
      />
      <Arrow />
      <Stage
        label="Verified"
        detail={`${num(s.samplesScored)} / ${num(s.samplesTotal)}`}
        fraction={pct(s.samplesScored, s.samplesTotal) / 100}
      />
    </box>
  );
}

const cardBox = {
  flexDirection: "row" as const,
  border: true,
  borderColor: "#334155",
  paddingLeft: 1,
  paddingRight: 1,
  marginBottom: 1,
};

// ---------------------------------------------------------------------------
// NARROW layout (compact one line per language, never clipped)
// ---------------------------------------------------------------------------
function pctStr(part: number, whole: number): string {
  return `${Math.round(pct(part, whole)).toString().padStart(3)}%`;
}

function NarrowRow({ s }: { s: LangStats }) {
  const name = s.lang.toUpperCase().padEnd(9);

  if (s.verifyOnly) {
    const p = pct(s.samplesScored, s.samplesTotal);
    return (
      <box style={{ flexDirection: "row" }}>
        <text fg="#22d3ee">{name}</text>
        <text fg={rampColor(p)}>{`✓ ${Math.round(p)}%`}</text>
        <text fg="#94a3b8">{` (${num(s.samplesScored)}/${num(s.samplesTotal)})`}</text>
      </box>
    );
  }

  const clipTotal = s.clipsDone + s.clipsPending;
  const segP = pct(s.videosSegmented, s.videosTotal);
  const lblP = pct(s.clipsDone, clipTotal);
  const verP = pct(s.samplesScored, s.samplesTotal);
  const dl = s.downloadActive ? "⬇●" : "⬇○";

  return (
    <box style={{ flexDirection: "row" }}>
      <text fg="#22d3ee">{name}</text>
      <text fg={s.downloadActive ? "#4ade80" : "#64748b"}>{dl}</text>
      <text fg={rampColor(segP)}>{`  seg ${pctStr(s.videosSegmented, s.videosTotal)}`}</text>
      <text fg={rampColor(lblP)}>{`  lbl ${pctStr(s.clipsDone, clipTotal)}`}</text>
      <text fg={rampColor(verP)}>{`  ✓${pctStr(s.samplesScored, s.samplesTotal)}`}</text>
    </box>
  );
}

// ---------------------------------------------------------------------------
// Disk
// ---------------------------------------------------------------------------
function DiskBar({ d, width }: { d: DiskStats; width: number }) {
  const usedFrac = d.totalGb ? (d.totalGb - d.freeGb) / d.totalGb : 0;
  const freeFrac = d.totalGb ? d.freeGb / d.totalGb : 0;
  return (
    <box style={{ flexDirection: "row", marginRight: 2 }}>
      <text fg="#94a3b8">{d.label.padEnd(9)}</text>
      <text fg={diskColor(freeFrac)}>{bar(usedFrac, width)}</text>
      <text fg="#e2e8f0">
        {" "}
        {d.error ? "(n/a)" : `${d.freeGb}G/${d.totalGb}G`}
      </text>
    </box>
  );
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------
export function App({ snapshot }: { snapshot?: Snapshot }) {
  const [snap, setSnap] = useState<Snapshot>(() => snapshot ?? getSnapshot());
  const [now, setNow] = useState<number>(Date.now());
  const renderer = useRenderer();
  const { width } = useTerminalDimensions();
  const narrow = width < NARROW_AT;

  useEffect(() => {
    if (snapshot) return;
    const id = setInterval(() => {
      setSnap(getSnapshot());
      setNow(Date.now());
    }, TICK_MS);
    return () => clearInterval(id);
  }, [snapshot]);

  useKeyboard((key) => {
    if (key.name === "q" || (key.ctrl && key.name === "c")) {
      // destroy() runs OpenTUI's full teardown (native terminal restore:
      // disables mouse, shows cursor, leaves alt-screen) and our onDestroy
      // safety net. Do this BEFORE exiting so we don't leave the terminal stuck.
      try {
        renderer?.destroy?.();
      } catch {}
      process.exit(0);
    }
  });

  const clock = (snapshot ? snap.ts : new Date(now)).toLocaleTimeString("en-US", {
    hour12: false,
  });
  const polling = snap.queuePolling || snap.curatorPolling;

  return (
    <box style={{ flexDirection: "column", padding: 1 }}>
      <box
        style={{
          flexDirection: narrow ? "column" : "row",
          justifyContent: "space-between",
          marginBottom: 1,
        }}
      >
        <text fg="#22d3ee" attributes={1}>
          {narrow ? "peacock-asr" : "  peacock-asr — pipeline"}
        </text>
        <text fg="#64748b">{`${clock}${polling ? "  ◍ polling" : ""}  •  q quit`}</text>
      </box>

      {narrow ? (
        <box
          style={{
            flexDirection: "column",
            border: true,
            borderColor: "#334155",
            paddingLeft: 1,
            paddingRight: 1,
            marginBottom: 1,
          }}
          title="pipeline"
        >
          {snap.langs.map((s) => (
            <NarrowRow key={s.lang} s={s} />
          ))}
          <text fg="#475569">{"seg=video segmented · lbl=clips done · ✓=verified"}</text>
        </box>
      ) : (
        snap.langs.map((s) => <WideCard key={s.lang} s={s} now={now} />)
      )}

      <box
        style={{
          flexDirection: "column",
          border: true,
          borderColor: "#334155",
          paddingLeft: 1,
          paddingRight: 1,
        }}
        title={`disk   updated ${ageStr(snap.disksUpdatedAt, now)}`}
      >
        <box style={{ flexDirection: narrow ? "column" : "row", flexWrap: "wrap" }}>
          {snap.disks.map((d) => (
            <DiskBar key={d.mount} d={d} width={narrow ? 12 : 16} />
          ))}
        </box>
      </box>
    </box>
  );
}
