"use client";

import { ArrowRight, ChevronRight, FileText, Save } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";
import { Navbar } from "@/components/navbar";
import {
  SentenceLevelScore,
  type SentenceScores,
} from "@/components/sentence-level-score";
import { TaskHeader } from "@/components/task-header";
import { WaveformPlayer } from "@/components/waveform-player";
import { type WordEntry, WordLevelScore } from "@/components/word-level-score";
import { CORPUS, type CorpusEntry } from "@/lib/data";
import { cn } from "@/lib/utils";

const TOTAL = CORPUS.length;

// Keyed by currentTask — React resets state when key changes, eliminating the
// extra render cycle that useEffect + setState caused.
function AnnotateContent({
  source,
  currentTask,
  onNavigate,
}: {
  source: CorpusEntry;
  currentTask: number;
  onNavigate: (task: number) => void;
}) {
  const [words, setWords] = useState<WordEntry[]>(source.words);
  const [sentence, setSentence] = useState<SentenceScores>(source.sentence);
  const [switchTo, setSwitchTo] = useState(currentTask);
  const [saved, setSaved] = useState(false);

  return (
    <main className="mx-auto flex w-full max-w-screen-xl flex-1 flex-col gap-5 px-4 py-5">
      <WaveformPlayer
        audioSize={source.audioSize}
        filename={source.file}
        totalLength={source.totalLength}
      />
      <div className="overflow-hidden rounded-xl border border-border bg-card shadow-sm">
        <div className="flex items-center gap-2 border-border border-b bg-muted/60 px-4 py-2.5">
          <FileText className="h-4 w-4 text-primary" />
          <span className="font-semibold text-foreground text-xs uppercase tracking-wide">
            Current Result
          </span>
          <span className="ml-auto font-mono text-muted-foreground text-xs">
            {source.file}
          </span>
        </div>
        <div className="px-4 py-3">
          <p className="font-medium text-base text-foreground leading-relaxed">
            {source.transcript}
          </p>
        </div>
      </div>
      <WordLevelScore
        onChange={(i, f, v) =>
          setWords((prev) =>
            prev.map((w, idx) => (idx === i ? { ...w, [f]: v } : w))
          )
        }
        words={words}
      />
      <SentenceLevelScore
        onChange={(f, v) => setSentence((prev) => ({ ...prev, [f]: v }))}
        scores={sentence}
        totalWords={words.length}
      />
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-border bg-card px-4 py-3 shadow-sm">
        <div className="flex items-center gap-3">
          <label
            className="whitespace-nowrap font-medium text-muted-foreground text-xs"
            htmlFor="switch-to-sentence"
          >
            Switch to sentence
          </label>
          <input
            className="w-20 rounded-md border border-border bg-background py-1.5 text-center font-mono font-semibold text-sm focus:outline-none focus:ring-1 focus:ring-primary"
            id="switch-to-sentence"
            max={100}
            min={1}
            onChange={(e) =>
              setSwitchTo(Number.parseInt(e.target.value, 10) || 1)
            }
            type="number"
            value={switchTo}
          />
          <button
            className="flex items-center gap-1 rounded-md border border-border px-3 py-1.5 text-muted-foreground text-xs transition-colors hover:border-primary/50 hover:text-foreground"
            onClick={() => onNavigate(switchTo)}
            type="button"
          >
            Go <ChevronRight className="h-3 w-3" />
          </button>
        </div>
        <button
          className={cn(
            "flex items-center gap-2 rounded-lg px-5 py-2 font-semibold text-sm shadow transition-all",
            saved
              ? "scale-95 bg-[var(--score-good)] text-white"
              : "bg-primary text-primary-foreground hover:opacity-90"
          )}
          disabled={saved}
          onClick={() => {
            setSaved(true);
            setTimeout(() => {
              setSaved(false);
              onNavigate(currentTask + 1);
            }, 800);
          }}
          type="button"
        >
          {saved ? (
            "Saved"
          ) : (
            <>
              <Save className="h-4 w-4" />
              Save &amp; Next
              <ArrowRight className="h-4 w-4" />
            </>
          )}
        </button>
      </div>
    </main>
  );
}

function AnnotateView() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const taskParam = Number.parseInt(searchParams.get("task") ?? "60", 10);
  const currentTask = Math.max(1, Math.min(100, taskParam));
  const corpusIndex = (currentTask - 1) % TOTAL;
  const source = CORPUS[corpusIndex];

  const navigate = (task: number) =>
    router.push(`/?task=${Math.max(1, Math.min(100, task))}`);

  return (
    <>
      <TaskHeader
        audioSize={source.audioSize}
        batch={722}
        current={currentTask}
        onNext={() => navigate(currentTask + 1)}
        onPrev={() => navigate(currentTask - 1)}
        onShowOthers={() => {
          /* no-op */
        }}
        project={56}
        task={113_373 + corpusIndex}
        total={100}
      />
      <AnnotateContent
        currentTask={currentTask}
        key={currentTask}
        onNavigate={navigate}
        source={source}
      />
    </>
  );
}

export default function TranscriptionPage() {
  return (
    <div className="flex min-h-screen flex-col bg-background">
      <Navbar />
      <Suspense fallback={<div className="min-h-screen bg-background" />}>
        <AnnotateView />
      </Suspense>
    </div>
  );
}
