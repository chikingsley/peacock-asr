"use client";

import { BookOpen } from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";

export interface WordEntry {
  phonemeAccuracy: number;
  phonemeScore: 0 | 1;
  phonetic: string;
  stressAccuracy: 5 | 10;
  word: string;
}

interface WordLevelScoreProps {
  onChange: (
    index: number,
    field: keyof WordEntry,
    value: number | string
  ) => void;
  words: WordEntry[];
}

function ScoreInput({
  value,
  onChange,
  id,
  readOnly,
}: {
  value: number;
  onChange?: (v: number) => void;
  id: string;
  readOnly?: boolean;
}) {
  let color: string;
  if (value >= 9) {
    color = "text-[var(--score-good)]";
  } else if (value >= 6) {
    color = "text-[var(--score-warn)]";
  } else {
    color = "text-[var(--score-bad)]";
  }

  return (
    <input
      className={cn(
        "w-16 rounded-md border border-border bg-background text-center font-mono font-semibold text-xs",
        "py-1 transition-colors focus:outline-none focus:ring-1 focus:ring-primary",
        readOnly && "cursor-default bg-muted opacity-70",
        color
      )}
      id={id}
      max={10}
      min={0}
      onChange={
        onChange
          ? (e) => onChange(Number.parseInt(e.target.value, 10) || 0)
          : undefined
      }
      readOnly={readOnly}
      step={1}
      type="number"
      value={value}
    />
  );
}

function RadioToggle({
  options,
  value,
  onChange,
  name,
}: {
  options: number[];
  value: number;
  onChange: (v: number) => void;
  name: string;
}) {
  return (
    <div
      aria-label={name}
      className="flex items-center justify-center gap-1"
      role="radiogroup"
    >
      {options.map((opt) => (
        <label
          className={cn(
            "flex h-6 w-7 cursor-pointer items-center justify-center rounded-md border font-semibold text-xs transition-all",
            value === opt
              ? "border-primary bg-primary text-primary-foreground shadow-sm"
              : "border-border text-muted-foreground hover:border-primary/50 hover:text-foreground"
          )}
          key={opt}
        >
          <input
            checked={value === opt}
            className="sr-only"
            name={name}
            onChange={() => onChange(opt)}
            type="radio"
            value={opt}
          />
          {opt}
        </label>
      ))}
    </div>
  );
}

export function WordLevelScore({ words, onChange }: WordLevelScoreProps) {
  const [hoveredWord, setHoveredWord] = useState<number | null>(null);

  return (
    <section className="overflow-hidden rounded-xl border border-border bg-card shadow-sm">
      <div className="flex items-center gap-2 border-border border-b bg-muted/60 px-4 py-3">
        <BookOpen className="h-4 w-4 text-primary" />
        <h2 className="font-semibold text-foreground text-sm">
          Word-level Score
        </h2>
        <span className="ml-auto text-muted-foreground text-xs">
          {words.length} words
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px]">
          <thead>
            <tr className="border-border border-b bg-muted/30">
              <th className="w-36 px-4 py-2.5 text-left font-medium text-muted-foreground text-xs">
                Metric
              </th>
              {words.map((w, i) => (
                <th
                  className={cn(
                    "px-3 py-2.5 text-center font-semibold text-xs transition-colors",
                    hoveredWord === i
                      ? "bg-primary/5 text-primary"
                      : "text-foreground"
                  )}
                  key={`${w.word}-${w.phonetic}`}
                  onMouseEnter={() => setHoveredWord(i)}
                  onMouseLeave={() => setHoveredWord(null)}
                >
                  {w.word}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {/* Phonetic */}
            <tr className="border-border/50 border-b transition-colors hover:bg-muted/20">
              <td className="px-4 py-2 font-medium text-muted-foreground text-xs">
                Phonetic
              </td>
              {words.map((w, i) => (
                // biome-ignore lint/a11y/noNoninteractiveElementInteractions: column hover highlight
                <td
                  className={cn(
                    "px-3 py-2 text-center transition-colors",
                    hoveredWord === i ? "bg-primary/5" : ""
                  )}
                  key={`${w.word}-${w.phonetic}`}
                  onMouseEnter={() => setHoveredWord(i)}
                  onMouseLeave={() => setHoveredWord(null)}
                >
                  <span className="rounded border border-border/60 bg-secondary px-1.5 py-0.5 font-mono text-secondary-foreground text-xs">
                    {w.phonetic}
                  </span>
                </td>
              ))}
            </tr>

            {/* Phoneme Score */}
            <tr className="border-border/50 border-b transition-colors hover:bg-muted/20">
              <td className="px-4 py-2 font-medium text-muted-foreground text-xs">
                Phoneme Score
              </td>
              {words.map((w, i) => (
                // biome-ignore lint/a11y/noNoninteractiveElementInteractions: column hover highlight
                <td
                  className={cn(
                    "px-3 py-2 text-center transition-colors",
                    hoveredWord === i ? "bg-primary/5" : ""
                  )}
                  key={`${w.word}-${w.phonetic}`}
                  onMouseEnter={() => setHoveredWord(i)}
                  onMouseLeave={() => setHoveredWord(null)}
                >
                  <RadioToggle
                    name={`phonemeScore-${i}`}
                    onChange={(v) => onChange(i, "phonemeScore", v)}
                    options={[0, 1]}
                    value={w.phonemeScore}
                  />
                </td>
              ))}
            </tr>

            {/* Phoneme Accuracy — single row */}
            <tr className="border-border/50 border-b transition-colors hover:bg-muted/20">
              <td className="px-4 py-2 font-medium text-muted-foreground text-xs">
                Phoneme Accuracy
              </td>
              {words.map((w, i) => (
                // biome-ignore lint/a11y/noNoninteractiveElementInteractions: column hover highlight
                <td
                  className={cn(
                    "px-3 py-2 text-center transition-colors",
                    hoveredWord === i ? "bg-primary/5" : ""
                  )}
                  key={`${w.word}-${w.phonetic}`}
                  onMouseEnter={() => setHoveredWord(i)}
                  onMouseLeave={() => setHoveredWord(null)}
                >
                  <div className="flex justify-center">
                    <ScoreInput
                      id={`pa-${i}`}
                      onChange={(v) => onChange(i, "phonemeAccuracy", v)}
                      value={w.phonemeAccuracy}
                    />
                  </div>
                </td>
              ))}
            </tr>

            {/* Stress Accuracy */}
            <tr className="border-border/50 border-b transition-colors hover:bg-muted/20">
              <td className="px-4 py-2 font-medium text-muted-foreground text-xs">
                Stress Accuracy
              </td>
              {words.map((w, i) => (
                // biome-ignore lint/a11y/noNoninteractiveElementInteractions: column hover highlight
                <td
                  className={cn(
                    "px-3 py-2 text-center transition-colors",
                    hoveredWord === i ? "bg-primary/5" : ""
                  )}
                  key={`${w.word}-${w.phonetic}`}
                  onMouseEnter={() => setHoveredWord(i)}
                  onMouseLeave={() => setHoveredWord(null)}
                >
                  <RadioToggle
                    name={`stress-${i}`}
                    onChange={(v) => onChange(i, "stressAccuracy", v)}
                    options={[5, 10]}
                    value={w.stressAccuracy}
                  />
                </td>
              ))}
            </tr>

            {/* Total Score — auto-calculated, read-only */}
            <tr className="bg-muted/20 transition-colors hover:bg-muted/30">
              <td className="px-4 py-2.5 font-semibold text-foreground text-xs">
                Total Score
                <span className="ml-1 font-normal text-[10px] text-muted-foreground">
                  (auto)
                </span>
              </td>
              {words.map((w, i) => {
                const auto = Number.parseFloat(
                  ((w.phonemeAccuracy + w.stressAccuracy) / 2).toFixed(2)
                );
                return (
                  // biome-ignore lint/a11y/noNoninteractiveElementInteractions: column hover highlight
                  <td
                    className={cn(
                      "px-3 py-2.5 text-center transition-colors",
                      hoveredWord === i ? "bg-primary/5" : ""
                    )}
                    key={`${w.word}-${w.phonetic}`}
                    onMouseEnter={() => setHoveredWord(i)}
                    onMouseLeave={() => setHoveredWord(null)}
                  >
                    <div className="flex justify-center">
                      <ScoreInput id={`total-${i}`} readOnly value={auto} />
                    </div>
                  </td>
                );
              })}
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  );
}
