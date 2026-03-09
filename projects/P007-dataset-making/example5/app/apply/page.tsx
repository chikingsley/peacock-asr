"use client";

import { CheckCircle2, Clock, Globe, Mic2, Send } from "lucide-react";
import { useState } from "react";
import { Navbar } from "@/components/navbar";
import { DIFFICULTY_COLORS, PROJECTS as INITIAL_PROJECTS } from "@/lib/data";
import { cn } from "@/lib/utils";

export default function ApplyForWorkPage() {
  const [projects, setProjects] = useState(INITIAL_PROJECTS);
  const [filter, setFilter] = useState<"all" | "applied">("all");

  const visible =
    filter === "applied" ? projects.filter((p) => p.applied) : projects;

  const handleApply = (id: number) => {
    setProjects((prev) =>
      prev.map((p) => (p.id === id ? { ...p, applied: true } : p))
    );
  };

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <Navbar />
      <main className="mx-auto flex w-full max-w-screen-xl flex-1 flex-col gap-6 px-4 py-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="font-bold text-foreground text-xl">
              Apply for Work
            </h1>
            <p className="mt-0.5 text-muted-foreground text-sm">
              Browse open annotation projects and apply
            </p>
          </div>
          <div className="flex items-center gap-2 rounded-lg bg-muted p-1">
            {(["all", "applied"] as const).map((f) => (
              <button
                className={cn(
                  "rounded-md px-3 py-1.5 font-semibold text-xs capitalize transition-colors",
                  filter === f
                    ? "border border-border bg-card text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                )}
                key={f}
                onClick={() => setFilter(f)}
                type="button"
              >
                {f === "all" ? "All Projects" : "Applied"}
              </button>
            ))}
          </div>
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          {visible.map((proj) => (
            <div
              className="flex flex-col overflow-hidden rounded-xl border border-border bg-card shadow-sm transition-shadow hover:shadow-md"
              key={proj.id}
            >
              <div className="flex items-start justify-between gap-3 border-border border-b px-4 pt-4 pb-3">
                <div className="flex items-start gap-3">
                  <div className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary/10">
                    <Mic2 className="h-4 w-4 text-primary" />
                  </div>
                  <div>
                    <h2 className="font-bold text-foreground text-sm leading-snug">
                      {proj.title}
                    </h2>
                    <div className="mt-1 flex flex-wrap items-center gap-2">
                      <span className="rounded border border-border bg-secondary px-1.5 py-0.5 font-mono font-semibold text-foreground text-xs">
                        {proj.lang}
                      </span>
                      <span className="flex items-center gap-1 text-muted-foreground text-xs">
                        <Globe className="h-3 w-3" />
                        {proj.type}
                      </span>
                    </div>
                  </div>
                </div>
                <span
                  className={cn(
                    "shrink-0 rounded-full border px-2.5 py-1 font-semibold text-xs",
                    DIFFICULTY_COLORS[proj.difficulty]
                  )}
                >
                  {proj.difficulty}
                </span>
              </div>
              <div className="flex flex-1 flex-col gap-3 px-4 py-3">
                <p className="text-muted-foreground text-xs leading-relaxed">
                  {proj.description}
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {proj.tags.map((tag) => (
                    <span
                      className="rounded border border-border bg-muted px-2 py-0.5 font-medium text-[10px] text-muted-foreground"
                      key={tag}
                    >
                      {tag}
                    </span>
                  ))}
                </div>
                <div className="grid grid-cols-3 gap-2 text-xs">
                  {(
                    [
                      ["Rate", proj.rate],
                      ["Tasks", String(proj.tasks)],
                      ["Slots left", String(proj.slots)],
                    ] as const
                  ).map(([label, val]) => (
                    <div
                      className="flex flex-col gap-0.5 rounded-lg bg-muted/60 p-2"
                      key={label}
                    >
                      <span className="font-medium text-[10px] text-muted-foreground uppercase tracking-wide">
                        {label}
                      </span>
                      <span
                        className={cn(
                          "font-bold font-mono",
                          label === "Slots left" && proj.slots <= 3
                            ? "text-[var(--score-bad)]"
                            : "text-foreground"
                        )}
                      >
                        {val}
                      </span>
                    </div>
                  ))}
                </div>
                <div className="mt-auto flex items-center justify-between border-border border-t pt-2">
                  <div className="flex items-center gap-1 text-muted-foreground text-xs">
                    <Clock className="h-3 w-3" />
                    Due {proj.deadline}
                  </div>
                  {proj.applied ? (
                    <span className="inline-flex items-center gap-1.5 rounded-lg border border-[var(--score-good)]/30 bg-[var(--score-good)]/10 px-3 py-1.5 font-semibold text-[var(--score-good)] text-xs">
                      <CheckCircle2 className="h-3.5 w-3.5" /> Applied
                    </span>
                  ) : (
                    <button
                      className="inline-flex items-center gap-1.5 rounded-lg bg-primary px-3 py-1.5 font-semibold text-primary-foreground text-xs transition-opacity hover:opacity-90"
                      onClick={() => handleApply(proj.id)}
                      type="button"
                    >
                      <Send className="h-3 w-3" /> Apply Now
                    </button>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </main>
    </div>
  );
}
