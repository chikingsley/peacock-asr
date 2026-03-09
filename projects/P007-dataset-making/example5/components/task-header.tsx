"use client";

import { ChevronLeft, ChevronRight, Users } from "lucide-react";

interface TaskHeaderProps {
  audioSize: string;
  batch: number;
  current: number;
  onNext: () => void;
  onPrev: () => void;
  onShowOthers: () => void;
  project: number;
  task: number;
  total: number;
}

export function TaskHeader({
  project,
  batch,
  task,
  audioSize,
  current,
  total,
  onPrev,
  onNext,
  onShowOthers,
}: TaskHeaderProps) {
  const progress = (current / total) * 100;

  return (
    <div className="border-border border-b bg-card">
      <div className="mx-auto flex max-w-screen-2xl flex-wrap items-center gap-3 px-4 py-2.5">
        {/* Meta info */}
        <div className="flex min-w-0 flex-1 items-center gap-1.5">
          <div className="flex flex-wrap items-center gap-1">
            <span className="inline-flex items-center gap-1 rounded-md bg-muted px-2 py-0.5 font-mono text-muted-foreground text-xs">
              Project
              <span className="font-semibold text-foreground">{project}</span>
            </span>
            <span className="text-border">/</span>
            <span className="inline-flex items-center gap-1 rounded-md bg-muted px-2 py-0.5 font-mono text-muted-foreground text-xs">
              Batch
              <span className="font-semibold text-foreground">{batch}</span>
            </span>
            <span className="text-border">/</span>
            <span className="inline-flex items-center gap-1 rounded-md bg-muted px-2 py-0.5 font-mono text-muted-foreground text-xs">
              Task
              <span className="font-semibold text-foreground">{task}</span>
            </span>
          </div>
          <span className="ml-1 hidden text-muted-foreground text-xs sm:inline">
            · {audioSize}
          </span>
        </div>

        {/* Navigation */}
        <div className="flex shrink-0 items-center gap-1 rounded-lg bg-muted p-0.5">
          <button
            aria-label="Previous task"
            className="rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-card hover:text-foreground disabled:opacity-40"
            onClick={onPrev}
            type="button"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
          <div className="flex items-center gap-2 px-2">
            <div className="relative h-1.5 w-16 overflow-hidden rounded-full bg-border">
              <div
                className="absolute inset-y-0 left-0 rounded-full bg-primary transition-all"
                style={{ width: `${progress}%` }}
              />
            </div>
            <span className="font-semibold text-foreground text-xs tabular-nums">
              {current}
            </span>
            <span className="text-muted-foreground text-xs">/</span>
            <span className="text-muted-foreground text-xs tabular-nums">
              {total}
            </span>
          </div>
          <button
            aria-label="Next task"
            className="rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-card hover:text-foreground"
            onClick={onNext}
            type="button"
          >
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>

        {/* Show others */}
        <button
          className="flex items-center gap-1.5 rounded-md border border-border bg-muted px-3 py-1.5 font-medium text-muted-foreground text-xs transition-colors hover:bg-secondary-foreground/10 hover:text-foreground"
          onClick={onShowOthers}
          type="button"
        >
          <Users className="h-3.5 w-3.5" />
          Show Others
        </button>
      </div>
    </div>
  );
}
