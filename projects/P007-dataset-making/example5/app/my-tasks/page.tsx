import {
  AlertCircle,
  CheckCircle2,
  ChevronRight,
  Clock,
  FileAudio,
  Mic2,
} from "lucide-react";
import Link from "next/link";
import { Navbar } from "@/components/navbar";
import { TASKS } from "@/lib/data";
import { cn } from "@/lib/utils";

const STATUS_CONFIG = {
  completed: {
    label: "Completed",
    icon: CheckCircle2,
    color:
      "text-[var(--score-good)] bg-[var(--score-good)]/10 border-[var(--score-good)]/30",
  },
  "in-progress": {
    label: "In Progress",
    icon: Clock,
    color:
      "text-[var(--score-warn)] bg-[var(--score-warn)]/10 border-[var(--score-warn)]/30",
  },
  pending: {
    label: "Pending",
    icon: AlertCircle,
    color: "text-muted-foreground bg-muted border-border",
  },
};

export default function MyTasksPage() {
  const totalCompleted = TASKS.filter((t) => t.status === "completed").length;
  const totalInProgress = TASKS.filter(
    (t) => t.status === "in-progress"
  ).length;

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <Navbar />
      <main className="mx-auto flex w-full max-w-screen-xl flex-1 flex-col gap-6 px-4 py-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="font-bold text-foreground text-xl">My Tasks</h1>
            <p className="mt-0.5 text-muted-foreground text-sm">
              Your assigned transcription batches
            </p>
          </div>
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1.5 rounded-lg border border-[var(--score-good)]/30 bg-[var(--score-good)]/10 px-3 py-1.5 font-semibold text-[var(--score-good)] text-xs">
              <CheckCircle2 className="h-3.5 w-3.5" />
              {totalCompleted} completed
            </span>
            <span className="flex items-center gap-1.5 rounded-lg border border-[var(--score-warn)]/30 bg-[var(--score-warn)]/10 px-3 py-1.5 font-semibold text-[var(--score-warn)] text-xs">
              <Clock className="h-3.5 w-3.5" />
              {totalInProgress} in progress
            </span>
          </div>
        </div>
        <div className="overflow-hidden rounded-xl border border-border bg-card shadow-sm">
          <table className="w-full">
            <thead>
              <tr className="border-border border-b bg-muted/40">
                {[
                  "Task ID",
                  "Project / Batch",
                  "Language",
                  "Progress",
                  "Audio Size",
                  "Due Date",
                  "Status",
                  "",
                ].map((h) => (
                  <th
                    className="px-4 py-3 text-left font-medium text-muted-foreground text-xs"
                    key={h}
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {TASKS.map((task) => {
                const pct = Math.round((task.completed / task.total) * 100);
                const cfg = STATUS_CONFIG[task.status];
                const Icon = cfg.icon;
                return (
                  <tr
                    className="group border-border/50 border-b transition-colors hover:bg-muted/20"
                    key={task.id}
                  >
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <div className="flex h-7 w-7 items-center justify-center rounded-md bg-primary/10">
                          <FileAudio className="h-3.5 w-3.5 text-primary" />
                        </div>
                        <span className="font-mono font-semibold text-foreground text-sm">
                          #{task.id}
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-1.5">
                        <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-muted-foreground text-xs">
                          P{task.project}
                        </span>
                        <span className="text-muted-foreground/40 text-xs">
                          /
                        </span>
                        <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-muted-foreground text-xs">
                          B{task.batch}
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <span className="rounded border border-border bg-secondary px-2 py-0.5 font-mono font-semibold text-foreground text-xs">
                        {task.lang}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex min-w-[120px] items-center gap-2">
                        <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-border">
                          <div
                            className={cn(
                              "h-full rounded-full",
                              pct === 100
                                ? "bg-[var(--score-good)]"
                                : "bg-primary"
                            )}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                        <span className="font-mono font-semibold text-foreground text-xs tabular-nums">
                          {task.completed}/{task.total}
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-muted-foreground text-xs">
                      <Mic2 className="mr-1 inline h-3 w-3" />
                      {task.audioSize}
                    </td>
                    <td className="px-4 py-3 font-mono text-muted-foreground text-xs">
                      {task.due}
                    </td>
                    <td className="px-4 py-3">
                      <span
                        className={cn(
                          "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 font-medium text-xs",
                          cfg.color
                        )}
                      >
                        <Icon className="h-3 w-3" />
                        {cfg.label}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      {task.status !== "completed" && (
                        <Link
                          className="flex items-center gap-1 font-medium text-primary text-xs opacity-0 transition-opacity hover:underline group-hover:opacity-100"
                          href="/"
                        >
                          Continue <ChevronRight className="h-3 w-3" />
                        </Link>
                      )}
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
