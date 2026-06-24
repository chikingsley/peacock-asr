// Standalone DB reader run as a short-lived subprocess so a pathological /
// locked sqlite read can be killed by timeout without freezing the TUI.
//
// argv: <kind> <db-path>
//   kind = "queue"   -> { videos: {status:count}, clips: {status:count} }
//   kind = "curator" -> { total, scored }
// Prints one JSON line to stdout on success; exits non-zero on failure.
import { Database } from "bun:sqlite";

const kind = process.argv[2];
const path = process.argv[3];

function groupCounts(db: Database, table: string): Record<string, number> {
  const out: Record<string, number> = {};
  const rows = db
    .query(`SELECT status, count(*) c FROM ${table} GROUP BY status`)
    .all() as { status: string; c: number }[];
  for (const r of rows) out[r.status] = r.c;
  return out;
}

try {
  const db = new Database(path, { readonly: true });
  db.exec("PRAGMA busy_timeout = 2000");

  if (kind === "queue") {
    const result: { videos?: Record<string, number>; clips?: Record<string, number> } = {};
    try {
      result.videos = groupCounts(db, "videos");
    } catch {}
    try {
      result.clips = groupCounts(db, "clips");
    } catch {}
    process.stdout.write(JSON.stringify(result));
  } else if (kind === "curator") {
    const row = db
      .query("SELECT count(*) total, count(scribe_wer) scored FROM samples")
      .get() as { total: number; scored: number } | null;
    process.stdout.write(JSON.stringify(row ?? { total: 0, scored: 0 }));
  } else {
    process.exit(3);
  }
  db.close();
  process.exit(0);
} catch (e) {
  process.stderr.write(String((e as Error).message ?? e));
  process.exit(1);
}
