"use client";

import { BarChart3 } from "lucide-react";
import { cn } from "@/lib/utils";

export interface SentenceScores {
  accuracy: number;
  fluency: number;
  integrityWords: number;
  prosody: number;
}

interface SentenceLevelScoreProps {
  onChange: (field: keyof SentenceScores, value: number) => void;
  scores: SentenceScores;
  totalWords: number;
}

function getScoreColor(score: number, variant: "badge" | "border"): string {
  if (variant === "badge") {
    if (score >= 8) {
      return "bg-[var(--score-good)]/15 text-[var(--score-good)]";
    }
    if (score >= 5) {
      return "bg-[var(--score-warn)]/15 text-[var(--score-warn)]";
    }
    return "bg-[var(--score-bad)]/15 text-[var(--score-bad)]";
  }
  if (score >= 8) {
    return "border-[var(--score-good)] bg-[var(--score-good)]/8 text-[var(--score-good)]";
  }
  if (score >= 5) {
    return "border-[var(--score-warn)] bg-[var(--score-warn)]/8 text-[var(--score-warn)]";
  }
  return "border-[var(--score-bad)] bg-[var(--score-bad)]/8 text-[var(--score-bad)]";
}

function getRadioColor(opt: number, isSelected: boolean): string {
  if (opt >= 8) {
    return isSelected
      ? "bg-[var(--score-good)] border-[var(--score-good)] text-white"
      : "border-[var(--score-good)]/30 text-[var(--score-good)]/60 hover:border-[var(--score-good)]/60";
  }
  if (opt >= 5) {
    return isSelected
      ? "bg-[var(--score-warn)] border-[var(--score-warn)] text-white"
      : "border-[var(--score-warn)]/30 text-[var(--score-warn)]/60 hover:border-[var(--score-warn)]/60";
  }
  return isSelected
    ? "bg-[var(--score-bad)] border-[var(--score-bad)] text-white"
    : "border-[var(--score-bad)]/30 text-[var(--score-bad)]/60 hover:border-[var(--score-bad)]/60";
}

function RadioScale({
  name,
  value,
  onChange,
}: {
  name: string;
  value: number;
  onChange: (v: number) => void;
}) {
  const options = Array.from({ length: 11 }, (_, i) => i);
  return (
    <div aria-label={name} className="flex flex-wrap gap-1" role="radiogroup">
      {options.map((opt) => {
        const isSelected = value === opt;
        const color = getRadioColor(opt, isSelected);
        return (
          <label
            className={cn(
              "flex h-8 w-8 cursor-pointer items-center justify-center rounded-lg border-2 font-semibold text-xs transition-all hover:scale-105",
              color
            )}
            key={opt}
          >
            <input
              checked={isSelected}
              className="sr-only"
              name={name}
              onChange={() => onChange(opt)}
              type="radio"
              value={opt}
            />
            {opt}
          </label>
        );
      })}
    </div>
  );
}

export function SentenceLevelScore({
  scores,
  onChange,
  totalWords,
}: SentenceLevelScoreProps) {
  // Integrity as a score out of 10, proportional to how many words were attempted
  const integrityScore =
    totalWords > 0
      ? Number.parseFloat(
          ((scores.integrityWords / totalWords) * 10).toFixed(2)
        )
      : 0;

  const totalScore = Number.parseFloat(
    (
      (scores.accuracy + integrityScore + scores.fluency + scores.prosody) /
      4
    ).toFixed(2)
  );

  interface Row {
    content: React.ReactNode;
    description: string;
    label: string;
  }

  const rows: Row[] = [
    {
      label: "Accuracy",
      description: "The accuracy of the whole sentence",
      content: (
        <RadioScale
          name="accuracy"
          onChange={(v) => onChange("accuracy", v)}
          value={scores.accuracy}
        />
      ),
    },
    {
      label: "Integrity",
      description: `How many words were pronounced / attempted out of ${totalWords}`,
      content: (
        <div className="flex items-center gap-3">
          <select
            className="w-20 cursor-pointer rounded-md border border-border bg-background py-1.5 text-center font-mono font-semibold text-sm focus:outline-none focus:ring-1 focus:ring-primary"
            onChange={(e) =>
              onChange("integrityWords", Number.parseInt(e.target.value, 10))
            }
            value={scores.integrityWords}
          >
            {Array.from({ length: totalWords + 1 }, (_, n) => n).map((n) => (
              <option key={`integrity-${n}`} value={n}>
                {n}
              </option>
            ))}
          </select>
          <span className="text-muted-foreground text-xs">/ {totalWords}</span>
          <div
            className={cn(
              "ml-auto flex items-center gap-1 rounded-lg px-2.5 py-1 font-mono font-semibold text-xs",
              getScoreColor(integrityScore, "badge")
            )}
          >
            {integrityScore}
            <span className="font-normal text-muted-foreground">/ 10</span>
          </div>
        </div>
      ),
    },
    {
      label: "Fluency",
      description: "It is used to reflect the oral proficiency of the speaker",
      content: (
        <RadioScale
          name="fluency"
          onChange={(v) => onChange("fluency", v)}
          value={scores.fluency}
        />
      ),
    },
    {
      label: "Prosody Soundness",
      description:
        "Refers to how words pause (the division of meaning groups), the stress and speed of words in the middle of two pauses, and whether the continuous reading, weak reading and ellipsis, and the tone and intonation are correct.",
      content: (
        <RadioScale
          name="prosody"
          onChange={(v) => onChange("prosody", v)}
          value={scores.prosody}
        />
      ),
    },
  ];

  return (
    <section className="overflow-hidden rounded-xl border border-border bg-card shadow-sm">
      {/* Header */}
      <div className="flex items-center gap-2 border-border border-b bg-muted/60 px-4 py-3">
        <BarChart3 className="h-4 w-4 text-primary" />
        <h2 className="font-semibold text-foreground text-sm">
          Sentence-level Score
        </h2>
      </div>

      <table className="w-full">
        <thead>
          <tr className="border-border border-b bg-muted/30">
            <th className="w-72 px-4 py-2.5 text-left font-medium text-muted-foreground text-xs">
              Metric
            </th>
            <th className="px-4 py-2.5 text-left font-medium text-muted-foreground text-xs">
              Score
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ label, description, content }) => (
            <tr
              className="border-border/50 border-b transition-colors hover:bg-muted/20"
              key={label}
            >
              <td className="px-4 py-3 align-top">
                <p className="font-semibold text-foreground text-sm">{label}</p>
                <p className="mt-0.5 text-muted-foreground text-xs leading-relaxed">
                  {description}
                </p>
              </td>
              <td className="px-4 py-3 align-middle">{content}</td>
            </tr>
          ))}

          {/* Total Score — auto */}
          <tr className="bg-muted/20">
            <td className="px-4 py-3 align-top">
              <p className="font-semibold text-foreground text-sm">
                Total Score
                <span className="ml-1.5 font-normal text-[10px] text-muted-foreground">
                  (auto)
                </span>
              </p>
              <p className="mt-0.5 text-muted-foreground text-xs">
                Average of Accuracy, Integrity, Fluency and Prosody
              </p>
            </td>
            <td className="px-4 py-3 align-middle">
              <div
                className={cn(
                  "inline-flex items-center gap-1 rounded-lg border-2 px-3 py-1.5 font-bold font-mono text-base",
                  getScoreColor(totalScore, "border")
                )}
              >
                {totalScore}
                <span className="font-normal text-muted-foreground text-xs">
                  / 10
                </span>
              </div>
            </td>
          </tr>
        </tbody>
      </table>
    </section>
  );
}
