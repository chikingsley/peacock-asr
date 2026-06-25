// Compact number formatting: 4582438 -> "4.58M", 20150 -> "20.2K"
export function num(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 10_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toLocaleString("en-US");
}

export function pct(part: number, whole: number): number {
  if (whole <= 0) return 0;
  return Math.max(0, Math.min(100, (part / whole) * 100));
}

// Unicode block bar. width = total cells.
export function bar(fraction: number, width: number): string {
  const f = Math.max(0, Math.min(1, fraction));
  const filled = Math.round(f * width);
  return "█".repeat(filled) + "░".repeat(Math.max(0, width - filled));
}

// Color ramp by percentage for progress.
export function rampColor(p: number): string {
  if (p >= 80) return "#4ade80"; // green
  if (p >= 40) return "#fbbf24"; // amber
  if (p > 0) return "#fb923c"; // orange
  return "#6b7280"; // gray
}

// Disk free coloring: low free = red.
export function diskColor(freeFrac: number): string {
  if (freeFrac <= 0.1) return "#f87171"; // red
  if (freeFrac <= 0.25) return "#fbbf24"; // amber
  return "#4ade80"; // green
}
