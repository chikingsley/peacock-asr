"use client";

import { Award, CheckCircle2, Clock, Target } from "lucide-react";
import dynamic from "next/dynamic";
import { Navbar } from "@/components/navbar";
import { PROJECT_BREAKDOWN, STATS_CARDS } from "@/lib/data";
import { cn } from "@/lib/utils";

const StatisticsCharts = dynamic(
  () =>
    import("@/components/statistics-charts").then((m) => m.StatisticsCharts),
  {
    ssr: false,
    loading: () => (
      <div className="h-48 animate-pulse rounded-xl bg-muted/30" />
    ),
  }
);

const STAT_ICONS = {
  "Total Completed": CheckCircle2,
  "Avg. Quality Score": Award,
  "Hours Contributed": Clock,
  "Acceptance Rate": Target,
} as const;
const STAT_COLORS = {
  "Total Completed": "text-[var(--score-good)] bg-[var(--score-good)]/10",
  "Avg. Quality Score": "text-primary bg-primary/10",
  "Hours Contributed": "text-[var(--score-warn)] bg-[var(--score-warn)]/10",
  "Acceptance Rate": "text-[var(--score-good)] bg-[var(--score-good)]/10",
} as const;

function getScoreClass(score: number): string {
  if (score >= 8.5) {
    return "text-[var(--score-good)] bg-[var(--score-good)]/10";
  }
  if (score >= 7) {
    return "text-[var(--score-warn)] bg-[var(--score-warn)]/10";
  }
  return "text-[var(--score-bad)] bg-[var(--score-bad)]/10";
}

export default function StatisticsPage() {
  return (
    <div className="flex min-h-screen flex-col bg-background">
      <Navbar />
      <main className="mx-auto flex w-full max-w-screen-xl flex-1 flex-col gap-6 px-4 py-6">
        <div>
          <h1 className="font-bold text-foreground text-xl">Task Statistics</h1>
          <p className="mt-0.5 text-muted-foreground text-sm">
            Your annotation performance overview
          </p>
        </div>
        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          {STATS_CARDS.map(({ label, value, sub }) => {
            const Icon = STAT_ICONS[label as keyof typeof STAT_ICONS] ?? Target;
            const color =
              STAT_COLORS[label as keyof typeof STAT_COLORS] ??
              "text-primary bg-primary/10";
            return (
              <div
                className="flex items-start gap-3 rounded-xl border border-border bg-card p-4 shadow-sm"
                key={label}
              >
                <div
                  className={cn(
                    "flex h-9 w-9 shrink-0 items-center justify-center rounded-lg",
                    color
                  )}
                >
                  <Icon className="h-4 w-4" />
                </div>
                <div>
                  <div className="flex items-baseline gap-1">
                    <span className="font-bold font-mono text-2xl text-foreground">
                      {value}
                    </span>
                    <span className="text-muted-foreground text-xs">{sub}</span>
                  </div>
                  <p className="mt-0.5 text-muted-foreground text-xs">
                    {label}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
        <StatisticsCharts />
        <div className="overflow-hidden rounded-xl border border-border bg-card shadow-sm">
          <div className="flex items-center gap-2 border-border border-b bg-muted/60 px-4 py-3">
            <Target className="h-4 w-4 text-primary" />
            <h2 className="font-semibold text-foreground text-sm">
              Project Breakdown
            </h2>
          </div>
          <table className="w-full">
            <thead>
              <tr className="border-border border-b bg-muted/30">
                {["Project", "Completed / Total", "Progress", "Avg Score"].map(
                  (h) => (
                    <th
                      className="px-4 py-2.5 text-left font-medium text-muted-foreground text-xs"
                      key={h}
                    >
                      {h}
                    </th>
                  )
                )}
              </tr>
            </thead>
            <tbody>
              {PROJECT_BREAKDOWN.map((row) => {
                const pct = Math.round((row.completed / row.total) * 100);
                const sc = getScoreClass(row.avgScore);
                return (
                  <tr
                    className="border-border/50 border-b transition-colors hover:bg-muted/20"
                    key={row.project}
                  >
                    <td className="px-4 py-3 font-mono font-semibold text-foreground text-sm">
                      {row.project}
                    </td>
                    <td className="px-4 py-3 font-mono text-muted-foreground text-sm">
                      {row.completed} / {row.total}
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <div className="h-1.5 w-32 overflow-hidden rounded-full bg-border">
                          <div
                            className="h-full rounded-full bg-primary"
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                        <span className="font-mono font-semibold text-muted-foreground text-xs">
                          {pct}%
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <span
                        className={cn(
                          "rounded-lg px-2.5 py-1 font-bold font-mono text-sm",
                          sc
                        )}
                      >
                        {row.avgScore}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </main>
    </div>
  );
}
