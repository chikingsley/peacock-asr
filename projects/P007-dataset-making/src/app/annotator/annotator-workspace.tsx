import { ChevronRight, Pause, Play } from "lucide-react";
import { type ReactNode, useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const sentenceScale = Array.from({ length: 11 }, (_, index) => index);
const stressScale = [5, 10];
const phoneScale = [0, 1, 2];
const wordScale = Array.from({ length: 11 }, (_, index) => index);
const defaultReviewerId = "demo-reviewer";

type LoadState = "idle" | "loading" | "ready" | "error";

type Reviewer = {
  reviewerId?: string;
  reviewerName?: string;
  id?: string;
  name?: string;
};

type SessionBootstrap = {
  reviewer: Reviewer;
  queueSummary: {
    totalUtterances: number;
    completed: number;
    inProgress: number;
    remaining: number;
  };
  currentUtteranceId: string | null;
};

type QueueSummary = SessionBootstrap["queueSummary"] & {
  percent: number;
};

type WordDetail = {
  id: string;
  position: number;
  text: string;
  refPhones: string[];
  candidatePhones: string[];
  sourceAccuracy: number | null;
  sourceStress: number | null;
  sourceTotal: number | null;
};

type ExistingWordAnnotation = {
  accuracy: number | null;
  stress: number | null;
  phoneScores: number[];
  total: number | null;
  selectedVariant?: string | null;
};

type ExistingAnnotation = {
  utteranceId: string;
  reviewerId: string;
  sentenceScores: {
    accuracy: number | null;
    completeness: number | null;
    fluency: number | null;
    prosody: number | null;
    total?: number | null;
  };
  wordAnnotations: Record<string, ExistingWordAnnotation>;
};

type UtteranceDetail = {
  id: string;
  audioUrl: string;
  text: string;
  sourceScores: {
    accuracy: number | null;
    completeness: number | null;
    fluency: number | null;
    prosody: number | null;
    total: number | null;
  };
  words: WordDetail[];
  existingAnnotation: ExistingAnnotation | null;
};

type WordAnnotation = {
  accuracy: number | null;
  stress: number | null;
  phoneScores: number[];
  selectedVariant: string | null;
};

type SentenceScores = {
  accuracy: number | null;
  completeness: number | null;
  fluency: number | null;
  prosody: number | null;
};

type AnnotationPayload = {
  reviewerId: string;
  reviewerName?: string;
  utteranceId: string;
  status: "in_progress" | "completed";
  notes?: string | null;
  sentenceScores: SentenceScores & { total: number | null };
  wordScores: Array<{
    wordPosition: number;
    accuracy: number | null;
    stress: number | null;
    total: number | null;
    selectedVariant: string | null;
    notes?: string | null;
    phoneVotes: Array<{
      phoneIndex: number;
      canonicalPhone: string;
      label: "correct" | "accented" | "incorrect";
      observedPhone?: string | null;
      note?: string | null;
    }>;
  }>;
};

type BackendUtteranceDetail = {
  reviewer: {
    reviewerId: string;
    reviewerName: string;
  };
  id: string;
  transcript: string;
  audioUrl: string;
  sentenceScores: {
    accuracy: number[];
    completeness: number[];
    fluency: number[];
    prosodic: number[];
    total: number[];
  };
  words: Array<{
    position: number;
    text: string;
    refPhones: string;
    referencePhoneTokens: string[];
    candidateSequences: string[];
    referenceWordScores: {
      accuracy: number[];
      stress: number[];
      total: number[];
      phones: string[];
    };
  }>;
  existingAnnotation: null | {
    id: number;
    status: string;
    updatedAt: string;
    notes: string | null;
    sentenceScores: {
      accuracy: number | null;
      completeness: number | null;
      fluency: number | null;
      prosody: number | null;
      total?: number | null;
    };
    payload: unknown;
    wordScores: Array<{
      wordPosition: number;
      accuracy: number | null;
      stress: number | null;
      total: number | null;
      selectedVariant: string | null;
      notes: string | null;
      phoneVotes: Array<{
        phoneIndex: number;
        canonicalPhone: string;
        label: "correct" | "accented" | "incorrect" | "unclear";
        observedPhone?: string | null;
        note?: string | null;
      }>;
    }>;
  };
};

function averageScore(values: number[] | undefined) {
  if (!values || values.length === 0) {
    return null;
  }

  const total = values.reduce((sum, value) => sum + value, 0);
  return Math.round((total / values.length) * 10) / 10;
}

function normalizeCompleteness(value: number | null) {
  if (value === null) {
    return null;
  }

  if (value <= 1) {
    return Math.round(value * 100);
  }

  return Math.round(value);
}

function formatPercent(value: number) {
  return `${Math.round(value)}%`;
}

function splitPhoneSequence(sequence: string | null | undefined) {
  if (!sequence) {
    return [];
  }

  return sequence
    .trim()
    .split(/\s+/)
    .filter(Boolean);
}

function toQueueSummary(summary: SessionBootstrap["queueSummary"]): QueueSummary {
  return {
    ...summary,
    percent: summary.totalUtterances === 0 ? 0 : (summary.completed / summary.totalUtterances) * 100,
  };
}

function reviewerIdOf(reviewer: Reviewer | null) {
  return reviewer?.reviewerId ?? reviewer?.id ?? defaultReviewerId;
}

function reviewerNameOf(reviewer: Reviewer | null) {
  return reviewer?.reviewerName ?? reviewer?.name ?? "Demo Reviewer";
}

function scoreFromVoteLabel(label: "correct" | "accented" | "incorrect" | "unclear") {
  if (label === "correct") {
    return 2;
  }

  if (label === "accented") {
    return 1;
  }

  return 0;
}

function voteLabelFromScore(score: number) {
  if (score >= 2) {
    return "correct" as const;
  }

  if (score === 1) {
    return "accented" as const;
  }

  return "incorrect" as const;
}

function initSentenceScores(existing: ExistingAnnotation | null): SentenceScores {
  return {
    accuracy: existing?.sentenceScores.accuracy ?? null,
    completeness: normalizeCompleteness(existing?.sentenceScores.completeness ?? null),
    fluency: existing?.sentenceScores.fluency ?? null,
    prosody: existing?.sentenceScores.prosody ?? null,
  };
}

function computeWordTotal(accuracy: number | null, stress: number | null) {
  if (accuracy === null && stress === null) {
    return null;
  }

  if (accuracy === null) {
    return stress;
  }

  if (stress === null) {
    return accuracy;
  }

  return Math.round((accuracy * 0.8 + stress * 0.2) * 10) / 10;
}

function computeSentencePreviewTotal(scores: SentenceScores) {
  const values = [scores.accuracy, scores.fluency, scores.prosody].filter(
    (value): value is number => value !== null,
  );

  if (values.length === 0) {
    return null;
  }

  return Math.round((values.reduce((sum, value) => sum + value, 0) / values.length) * 10) / 10;
}

function clampPhoneIndex(index: number, length: number) {
  if (length <= 0) {
    return 0;
  }

  return Math.min(Math.max(index, 0), length - 1);
}

function activePhonesForWord(word: WordDetail, annotation: WordAnnotation) {
  const selected = splitPhoneSequence(annotation.selectedVariant);
  return selected.length > 0 ? selected : word.refPhones;
}

function initWordAnnotation(word: WordDetail, source: ExistingWordAnnotation | undefined): WordAnnotation {
  const selectedVariant = source?.selectedVariant ?? word.candidatePhones[0] ?? word.refPhones.join(" ");
  const phones = splitPhoneSequence(selectedVariant);

  return {
    accuracy: source?.accuracy ?? null,
    stress: source?.stress ?? null,
    selectedVariant,
    phoneScores:
      source?.phoneScores && source.phoneScores.length === phones.length
        ? source.phoneScores
        : phones.map((_, index) => source?.phoneScores?.[index] ?? 2),
  };
}

function initialSelectedPhoneIndex(word: WordDetail, annotation: WordAnnotation) {
  const phones = activePhonesForWord(word, annotation);
  const firstMarked = annotation.phoneScores.findIndex((value) => value !== 2);
  return clampPhoneIndex(firstMarked === -1 ? 0 : firstMarked, phones.length);
}

function toInternalUtterance(detail: BackendUtteranceDetail): UtteranceDetail {
  const existingAnnotation: ExistingAnnotation | null = detail.existingAnnotation
    ? {
        utteranceId: detail.id,
        reviewerId: detail.reviewer.reviewerId,
        sentenceScores: detail.existingAnnotation.sentenceScores,
        wordAnnotations: Object.fromEntries(
          detail.existingAnnotation.wordScores.map((wordScore) => [
            String(wordScore.wordPosition),
            {
              accuracy: wordScore.accuracy,
              stress: wordScore.stress,
              total: wordScore.total,
              selectedVariant: wordScore.selectedVariant,
              phoneScores: (wordScore.phoneVotes ?? []).map((vote) => scoreFromVoteLabel(vote.label)),
            },
          ]),
        ),
      }
    : null;

  return {
    id: detail.id,
    audioUrl: detail.audioUrl,
    text: detail.transcript,
    sourceScores: {
      accuracy: averageScore(detail.sentenceScores.accuracy),
      completeness: averageScore(detail.sentenceScores.completeness),
      fluency: averageScore(detail.sentenceScores.fluency),
      prosody: averageScore(detail.sentenceScores.prosodic),
      total: averageScore(detail.sentenceScores.total),
    },
    words: detail.words.map((word) => ({
      id: String(word.position),
      position: word.position,
      text: word.text,
      refPhones: word.referencePhoneTokens,
      candidatePhones: word.candidateSequences,
      sourceAccuracy: averageScore(word.referenceWordScores.accuracy),
      sourceStress: averageScore(word.referenceWordScores.stress),
      sourceTotal: averageScore(word.referenceWordScores.total),
    })),
    existingAnnotation,
  };
}

async function fetchJson<T>(input: string, init?: RequestInit): Promise<T> {
  const response = await fetch(input, init);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export function AnnotatorWorkspace() {
  const waveContainerRef = useRef<HTMLDivElement | null>(null);
  const waveRef = useRef<WaveSurfer | null>(null);

  const [loadState, setLoadState] = useState<LoadState>("idle");
  const [reviewer, setReviewer] = useState<Reviewer | null>(null);
  const [queueSummary, setQueueSummary] = useState<QueueSummary | null>(null);
  const [utteranceId, setUtteranceId] = useState<string | null>(null);
  const [utterance, setUtterance] = useState<UtteranceDetail | null>(null);
  const [sentenceScores, setSentenceScores] = useState<SentenceScores>(initSentenceScores(null));
  const [wordAnnotations, setWordAnnotations] = useState<Record<string, WordAnnotation>>({});
  const [selectedPhones, setSelectedPhones] = useState<Record<string, number>>({});
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [playerState, setPlayerState] = useState<"ready" | "playing" | "paused">("paused");
  const [audioDurationSeconds, setAudioDurationSeconds] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;

    const loadSession = async () => {
      setLoadState("loading");

      try {
        const bootstrap = await fetchJson<SessionBootstrap>(`/api/session/${defaultReviewerId}`);
        if (cancelled) {
          return;
        }
        setReviewer(bootstrap.reviewer);
        setQueueSummary(toQueueSummary(bootstrap.queueSummary));
        setUtteranceId(bootstrap.currentUtteranceId);
        setLoadState("ready");
      } catch (error) {
        console.error(error);
        if (!cancelled) {
          setLoadState("error");
        }
      }
    };

    void loadSession();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const activeReviewerId = reviewerIdOf(reviewer);

    if (!utteranceId || !activeReviewerId) {
      setUtterance(null);
      return;
    }

    let cancelled = false;

    const loadUtterance = async () => {
      setLoadState("loading");
      try {
        const detail = await fetchJson<BackendUtteranceDetail>(
          `/api/utterances/${utteranceId}?reviewerId=${encodeURIComponent(activeReviewerId)}`,
        );
        if (cancelled) {
          return;
        }

        const mapped = toInternalUtterance(detail);
        const nextWordAnnotations = Object.fromEntries(
          mapped.words.map((word) => [
            word.id,
            initWordAnnotation(word, mapped.existingAnnotation?.wordAnnotations[word.id]),
          ]),
        );

        setUtterance(mapped);
        setSentenceScores(initSentenceScores(mapped.existingAnnotation));
        setWordAnnotations(nextWordAnnotations);
        setSelectedPhones(
          Object.fromEntries(
            mapped.words.map((word) => {
              const annotation = nextWordAnnotations[word.id] ?? initWordAnnotation(word, undefined);
              return [word.id, initialSelectedPhoneIndex(word, annotation)];
            }),
          ),
        );
        setAudioDurationSeconds(null);
        setSaveState("idle");
        setLoadState("ready");
      } catch (error) {
        console.error(error);
        if (!cancelled) {
          setLoadState("error");
        }
      }
    };

    void loadUtterance();

    return () => {
      cancelled = true;
    };
  }, [reviewer, utteranceId]);

  useEffect(() => {
    if (!utterance?.audioUrl || !waveContainerRef.current) {
      return;
    }

    waveRef.current?.destroy();

    const wave = WaveSurfer.create({
      container: waveContainerRef.current,
      height: 124,
      waveColor: "rgba(238, 231, 217, 0.22)",
      progressColor: "#d38c39",
      cursorColor: "rgba(255,255,255,0.95)",
      barWidth: 2,
      barRadius: 999,
      barGap: 2,
      normalize: true,
      url: utterance.audioUrl,
    });

    wave.on("ready", () => {
      setAudioDurationSeconds(wave.getDuration());
      setPlayerState("ready");
    });

    wave.on("play", () => {
      setPlayerState("playing");
    });

    wave.on("pause", () => {
      setPlayerState("paused");
    });

    wave.on("finish", () => {
      setPlayerState("paused");
    });

    waveRef.current = wave;

    return () => {
      wave.destroy();
      waveRef.current = null;
    };
  }, [utterance?.audioUrl]);

  const canSave = Boolean(reviewer && utterance);

  const togglePlayback = () => {
    waveRef.current?.playPause();
  };

  const setWordField = (wordId: string, patch: Partial<WordAnnotation>) => {
    setWordAnnotations((current) => {
      const existing = current[wordId];
      if (!existing) {
        return current;
      }

      return {
        ...current,
        [wordId]: {
          ...existing,
          ...patch,
        },
      };
    });
  };

  const setPhoneScore = (wordId: string, phoneIndex: number, phoneScore: number) => {
    setWordAnnotations((current) => {
      const existing = current[wordId];
      if (!existing) {
        return current;
      }

      return {
        ...current,
        [wordId]: {
          ...existing,
          phoneScores: existing.phoneScores.map((value, index) =>
            index === phoneIndex ? phoneScore : value,
          ),
        },
      };
    });
  };

  const setActivePhone = (wordId: string, phoneIndex: number) => {
    setSelectedPhones((current) => ({
      ...current,
      [wordId]: phoneIndex,
    }));
  };

  const setWordVariant = (wordId: string, variant: string) => {
    const nextPhones = splitPhoneSequence(variant);

    setWordAnnotations((current) => {
      const existing = current[wordId];
      if (!existing) {
        return current;
      }

      return {
        ...current,
        [wordId]: {
          ...existing,
          selectedVariant: variant,
          phoneScores: nextPhones.map((_, index) => existing.phoneScores[index] ?? 2),
        },
      };
    });

    setSelectedPhones((current) => ({
      ...current,
      [wordId]: clampPhoneIndex(current[wordId] ?? 0, nextPhones.length),
    }));
  };

  const setCompletenessFromSpokenWordCount = (nextCount: number) => {
    if (!utterance) {
      return;
    }

    const bounded = Math.max(0, Math.min(utterance.words.length, nextCount));
    const percent =
      utterance.words.length === 0 ? 0 : Math.round((bounded / utterance.words.length) * 100);

    setSentenceScores((current) => ({
      ...current,
      completeness: percent,
    }));
  };

  const saveAnnotation = async () => {
    if (!reviewer || !utterance) {
      return;
    }

    const sentenceTotal = computeSentencePreviewTotal(sentenceScores);

    const payload: AnnotationPayload = {
      reviewerId: reviewerIdOf(reviewer),
      reviewerName: reviewerNameOf(reviewer),
      utteranceId: utterance.id,
      status: "completed",
      notes: null,
      sentenceScores: {
        ...sentenceScores,
        total: sentenceTotal,
      },
      wordScores: utterance.words.map((word) => {
        const annotation = wordAnnotations[word.id] ?? initWordAnnotation(word, undefined);
        const phones = activePhonesForWord(word, annotation);

        return {
          wordPosition: word.position,
          accuracy: annotation.accuracy,
          stress: annotation.stress,
          total: computeWordTotal(annotation.accuracy, annotation.stress),
          selectedVariant: annotation.selectedVariant,
          notes: null,
          phoneVotes: phones.map((phone, index) => ({
            phoneIndex: index,
            canonicalPhone: phone,
            label: voteLabelFromScore(annotation.phoneScores[index] ?? 2),
          })),
        };
      }),
    };

    setSaveState("saving");

    try {
      const saved = await fetchJson<{ nextUtteranceId: string | null }>("/api/annotations", {
        method: "POST",
        headers: {
          "content-type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      setSaveState("saved");
      if (saved.nextUtteranceId) {
        setUtteranceId(saved.nextUtteranceId);
      }

      const bootstrap = await fetchJson<SessionBootstrap>(
        `/api/session/${encodeURIComponent(reviewerIdOf(reviewer))}`,
      );
      setQueueSummary(toQueueSummary(bootstrap.queueSummary));
    } catch (error) {
      console.error(error);
      setSaveState("error");
    }
  };

  if (loadState === "error") {
    return <ShellState heading="Annotator backend failed" text="Run `bun run dev` from the project root." />;
  }

  if (!utterance || !reviewer || !queueSummary) {
    return (
      <ShellState
        heading="Preparing the review sheet"
        text="Loading reviewer session, utterance queue, waveform source, and prior annotations."
      />
    );
  }

  const totalWords = utterance.words.length;
  const sentencePreviewTotal = computeSentencePreviewTotal(sentenceScores);
  const completenessPercent = sentenceScores.completeness ?? 100;
  const spokenWordCount =
    totalWords === 0 ? 0 : Math.round((Math.max(0, completenessPercent) / 100) * totalWords);
  const wordColumns = utterance.words.map((word) => {
    const annotation = wordAnnotations[word.id] ?? initWordAnnotation(word, undefined);
    const phones = activePhonesForWord(word, annotation);
    const selectedPhoneIndex = clampPhoneIndex(selectedPhones[word.id] ?? 0, phones.length);

    return {
      annotation,
      phones,
      selectedPhone: phones[selectedPhoneIndex] ?? "—",
      selectedPhoneIndex,
      total: computeWordTotal(annotation.accuracy, annotation.stress),
      word,
    };
  });

  return (
    <div className="relative overflow-hidden py-5">
      <div className="pointer-events-none absolute inset-0 opacity-75">
        <div className="absolute left-0 top-0 h-72 w-72 rounded-full bg-amber-300/18 blur-3xl" />
        <div className="absolute bottom-0 right-0 h-96 w-96 rounded-full bg-stone-200/10 blur-3xl" />
      </div>

      <div className="relative mx-auto w-full max-w-[1540px] px-3 sm:px-5">
        <div className="overflow-hidden rounded-[30px] border border-[#cdbfae] bg-[#ece5d8] text-[#1f1a14] shadow-[0_28px_90px_rgba(18,12,8,0.34)]">
          <div className="border-b border-[#d8ccbc] bg-[linear-gradient(180deg,#f8f3ea_0%,#ece2d4_100%)] px-4 py-3 sm:px-6">
            <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
              <div>
                <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-[#706251]">
                  SpeechOcean uTrans / review worksheet
                </p>
                <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-[#5f5245]">
                  <span>
                    Utterance <span className="font-semibold text-[#1f1a14]">{utterance.id}</span>
                  </span>
                  <span>
                    Reviewer <span className="font-semibold text-[#1f1a14]">{reviewerNameOf(reviewer)}</span>
                  </span>
                  <span>
                    Progress{" "}
                    <span className="font-semibold text-[#1f1a14]">
                      {queueSummary.completed} / {queueSummary.totalUtterances}
                    </span>
                  </span>
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-2">
                <StatusChip label={`${queueSummary.remaining} left`} tone="neutral" />
                <StatusChip label={formatPercent(queueSummary.percent)} tone="accent" />
                <StatusChip
                  label={
                    saveState === "saved"
                      ? "saved"
                      : saveState === "saving"
                        ? "saving"
                        : saveState === "error"
                          ? "error"
                          : "editing"
                  }
                  tone={saveState === "error" ? "danger" : "neutral"}
                />
              </div>
            </div>
          </div>

          <div className="px-4 py-4 sm:px-6">
            <div className="grid gap-4 lg:grid-cols-[1.4fr_0.65fr]">
              <section className="rounded-[24px] border border-[#d7ccbc] bg-[#f7f2e8] p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.65)]">
                <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                  <div>
                    <div className="text-[11px] font-semibold uppercase tracking-[0.22em] text-[#7a6c5a]">
                      Current result
                    </div>
                    <h1 className="mt-2 font-[family:var(--font-display)] text-[clamp(1.55rem,2vw,2.3rem)] leading-tight text-[#241d15]">
                      {utterance.text}
                    </h1>
                  </div>

                  <Button
                    className="h-10 rounded-full bg-[#1d1813] px-4 text-[#f7f0e4] hover:bg-[#0f0d0a]"
                    onClick={togglePlayback}
                    size="sm"
                  >
                    {playerState === "playing" ? (
                      <Pause className="mr-2 size-4" />
                    ) : (
                      <Play className="mr-2 size-4" />
                    )}
                    {playerState === "playing" ? "Pause" : "Play"} sentence
                  </Button>
                </div>

                <div className="mt-4 overflow-hidden rounded-[22px] border border-[#231c16] bg-[#090806] p-3 shadow-[inset_0_0_0_1px_rgba(255,255,255,0.04)]">
                  <div ref={waveContainerRef} />
                </div>

                <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-[12px] text-[#625444]">
                  <span>
                    Duration{" "}
                    <span className="font-semibold text-[#1f1a14]">
                      {audioDurationSeconds === null ? "—" : `${audioDurationSeconds.toFixed(2)}s`}
                    </span>
                  </span>
                  <span>
                    Source total{" "}
                    <span className="font-semibold text-[#1f1a14]">
                      {utterance.sourceScores.total ?? "—"}
                    </span>
                  </span>
                  <span>
                    Accuracy avg{" "}
                    <span className="font-semibold text-[#1f1a14]">
                      {utterance.sourceScores.accuracy ?? "—"}
                    </span>
                  </span>
                  <span>
                    Fluency avg{" "}
                    <span className="font-semibold text-[#1f1a14]">
                      {utterance.sourceScores.fluency ?? "—"}
                    </span>
                  </span>
                  <span>
                    Prosody avg{" "}
                    <span className="font-semibold text-[#1f1a14]">
                      {utterance.sourceScores.prosody ?? "—"}
                    </span>
                  </span>
                </div>
              </section>

              <section className="rounded-[24px] border border-[#d7ccbc] bg-[#f3ede2] p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.6)]">
                <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-[#7a6c5a]">
                  Workflow note
                </p>
                <div className="mt-3 space-y-3 text-sm leading-6 text-[#4a4034]">
                  <p>
                    The paper’s workflow is: read the sentence, listen several times, choose the
                    canonical phone sequence for ambiguous words, then score phones, word accuracy,
                    stress, and sentence-level judgments.
                  </p>
                  <p>
                    In the original app, the reviewer clicks a phone symbol first. The 0 / 1 score
                    applies to that selected phone, and every unselected phone is treated as 2 by
                    default. This rebuild keeps that logic but makes the selected phone explicit.
                  </p>
                </div>
              </section>
            </div>
          </div>

          <section className="border-t border-[#d8ccbc] bg-[#f8f3ea]">
            <div className="border-b border-[#ddd1c0] px-4 py-3 sm:px-6">
              <div className="flex flex-col gap-2 lg:flex-row lg:items-end lg:justify-between">
                <div>
                  <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-[#776857]">
                    Word-level score
                  </p>
                  <h2 className="mt-1 font-[family:var(--font-display)] text-2xl text-[#241d15]">
                    All words in one scoring matrix
                  </h2>
                </div>
                <p className="max-w-2xl text-sm text-[#5b4f41]">
                  Click a phone chip to focus it, then score 0, 1, or 2. Word total is derived as
                  0.8 × word accuracy + 0.2 × stress, which matches the reference scores in this
                  corpus.
                </p>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="min-w-full border-collapse text-sm">
                <tbody>
                  <tr className="bg-[#f2ebdf]">
                    <MatrixLabelCell
                      description="Target token in the sentence"
                      title="Word"
                    />
                    {wordColumns.map(({ word }) => (
                      <MatrixDataCell key={`word-${word.id}`}>
                        <div className="font-semibold uppercase tracking-[0.08em] text-[#1f1a14]">
                          {word.text}
                        </div>
                      </MatrixDataCell>
                    ))}
                  </tr>

                  <tr className="bg-[#faf7f0]">
                    <MatrixLabelCell
                      description="Choose the canonical sequence before scoring if more than one exists"
                      title="Phonetic symbols"
                    />
                    {wordColumns.map(({ annotation, phones, selectedPhoneIndex, word }) => (
                      <MatrixDataCell key={`phones-${word.id}`}>
                        <div className="space-y-2">
                          {word.candidatePhones.length > 1 ? (
                            <select
                              className="h-8 w-full rounded-md border border-[#cdbfae] bg-white/80 px-2 text-[12px] font-medium text-[#32281e] outline-none transition focus:border-[#8c6c3d] focus:ring-2 focus:ring-[#d2a96e]/35"
                              onChange={(event) => setWordVariant(word.id, event.target.value)}
                              value={annotation.selectedVariant ?? word.candidatePhones[0]}
                            >
                              {word.candidatePhones.map((candidate) => (
                                <option key={`${word.id}-${candidate}`} value={candidate}>
                                  {candidate}
                                </option>
                              ))}
                            </select>
                          ) : null}

                          <div className="flex flex-wrap gap-1.5">
                            {phones.map((phone, index) => (
                              <button
                                className={phoneChipClass(
                                  annotation.phoneScores[index] ?? 2,
                                  selectedPhoneIndex === index,
                                )}
                                key={`${word.id}-${phone}-${index}`}
                                onClick={() => setActivePhone(word.id, index)}
                                type="button"
                              >
                                <span>{phone}</span>
                                <span className="rounded-full bg-black/8 px-1 py-0.5 text-[10px] font-semibold">
                                  {annotation.phoneScores[index] ?? 2}
                                </span>
                              </button>
                            ))}
                          </div>
                        </div>
                      </MatrixDataCell>
                    ))}
                  </tr>

                  <tr className="bg-[#f6efe5]">
                    <MatrixLabelCell
                      description="Applies to the selected phone chip only. Unselected phones remain 2."
                      title="Phoneme score"
                    />
                    {wordColumns.map(({ annotation, selectedPhone, selectedPhoneIndex, word }) => (
                      <MatrixDataCell key={`phone-score-${word.id}`}>
                        <div className="space-y-2">
                          <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[#7a6d5d]">
                            Selected {selectedPhone}
                          </div>
                          <div className="flex gap-1">
                            {phoneScale.map((score) => (
                              <button
                                className={miniScoreClass(annotation.phoneScores[selectedPhoneIndex] ?? 2, score)}
                                key={`${word.id}-score-${score}`}
                                onClick={() => setPhoneScore(word.id, selectedPhoneIndex, score)}
                                type="button"
                              >
                                {score}
                              </button>
                            ))}
                          </div>
                        </div>
                      </MatrixDataCell>
                    ))}
                  </tr>

                  <tr className="bg-[#faf7f0]">
                    <MatrixLabelCell
                      description="Whole-word pronunciation quality, 0 to 10"
                      title="Word accuracy"
                    />
                    {wordColumns.map(({ annotation, word }) => (
                      <MatrixDataCell key={`word-accuracy-${word.id}`}>
                        <ValueSelect
                          onChange={(value) => setWordField(word.id, { accuracy: value })}
                          options={wordScale}
                          value={annotation.accuracy}
                        />
                      </MatrixDataCell>
                    ))}
                  </tr>

                  <tr className="bg-[#f6efe5]">
                    <MatrixLabelCell
                      description="10 if stress is correct, 5 if stress is wrong"
                      title="Stress"
                    />
                    {wordColumns.map(({ annotation, word }) => (
                      <MatrixDataCell key={`stress-${word.id}`}>
                        <div className="flex gap-1.5">
                          {stressScale.map((score) => (
                            <button
                              className={miniScoreClass(annotation.stress, score)}
                              key={`${word.id}-stress-${score}`}
                              onClick={() => setWordField(word.id, { stress: score })}
                              type="button"
                            >
                              {score}
                            </button>
                          ))}
                        </div>
                      </MatrixDataCell>
                    ))}
                  </tr>

                  <tr className="bg-[#fbf8f2]">
                    <MatrixLabelCell
                      description="Derived from accuracy and stress for consistency"
                      title="Total score"
                    />
                    {wordColumns.map(({ total, word }) => (
                      <MatrixDataCell key={`total-${word.id}`}>
                        <ReadonlyScoreBox value={total} />
                      </MatrixDataCell>
                    ))}
                  </tr>
                </tbody>
              </table>
            </div>
          </section>

          <section className="border-t border-[#d8ccbc] bg-[#f4ede2] px-4 py-4 sm:px-6">
            <div className="mb-4">
              <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-[#776857]">
                Sentence-level score
              </p>
              <h2 className="mt-1 font-[family:var(--font-display)] text-2xl text-[#241d15]">
                Whole-utterance judgment
              </h2>
            </div>
            <div className="overflow-x-auto rounded-[20px] border border-[#d8ccbc] bg-[#fbf8f2]">
              <table className="min-w-full border-collapse text-sm">
                <tbody>
                  <tr className="bg-[#f7f2e8]">
                    <SentenceLabelCell
                      description="Overall pronunciation quality of the whole sentence"
                      title="Accuracy"
                    />
                    <SentenceControlCell>
                      <SentenceScale
                        onSelect={(value) =>
                          setSentenceScores((current) => ({ ...current, accuracy: value }))
                        }
                        options={sentenceScale}
                        value={sentenceScores.accuracy}
                      />
                    </SentenceControlCell>
                  </tr>

                  <tr className="bg-[#fcfaf5]">
                    <SentenceLabelCell
                      description="Percentage of target words that were actually spoken, even if accented or imperfect"
                      title="Completeness / integrity"
                    />
                    <SentenceControlCell>
                      <div className="grid gap-3 md:grid-cols-3">
                        <label className="text-[12px] font-semibold uppercase tracking-[0.16em] text-[#7c6c59]">
                          Spoken words
                          <input
                            className="mt-1 h-10 w-full rounded-md border border-[#cdbfae] bg-white px-3 text-base font-semibold text-[#241d15] outline-none transition focus:border-[#8c6c3d] focus:ring-2 focus:ring-[#d2a96e]/35"
                            max={totalWords}
                            min={0}
                            onChange={(event) =>
                              setCompletenessFromSpokenWordCount(Number(event.target.value || 0))
                            }
                            type="number"
                            value={spokenWordCount}
                          />
                        </label>

                        <div className="text-[12px] font-semibold uppercase tracking-[0.16em] text-[#7c6c59]">
                          Target words
                          <div className="mt-1 flex h-10 items-center rounded-md border border-[#d6cab9] bg-[#f1ebe1] px-3 text-base font-semibold text-[#241d15]">
                            {totalWords}
                          </div>
                        </div>

                        <div className="text-[12px] font-semibold uppercase tracking-[0.16em] text-[#7c6c59]">
                          Integrity
                          <div className="mt-1 flex h-10 items-center rounded-md border border-[#d6cab9] bg-[#f1ebe1] px-3 text-base font-semibold text-[#241d15]">
                            {completenessPercent}%
                          </div>
                        </div>
                      </div>
                    </SentenceControlCell>
                  </tr>

                  <tr className="bg-[#f7f2e8]">
                    <SentenceLabelCell
                      description="Smoothness, pauses, repetition, and stammering"
                      title="Fluency"
                    />
                    <SentenceControlCell>
                      <SentenceScale
                        onSelect={(value) =>
                          setSentenceScores((current) => ({ ...current, fluency: value }))
                        }
                        options={sentenceScale}
                        value={sentenceScores.fluency}
                      />
                    </SentenceControlCell>
                  </tr>

                  <tr className="bg-[#fcfaf5]">
                    <SentenceLabelCell
                      description="Intonation, rhythm, and phrase-level soundness"
                      title="Prosody"
                    />
                    <SentenceControlCell>
                      <SentenceScale
                        onSelect={(value) =>
                          setSentenceScores((current) => ({ ...current, prosody: value }))
                        }
                        options={sentenceScale}
                        value={sentenceScores.prosody}
                      />
                    </SentenceControlCell>
                  </tr>

                  <tr className="bg-[#f1eadf]">
                    <SentenceLabelCell
                      description="Dev preview only. Not a paper-defined formula."
                      title="Total score"
                    />
                    <SentenceControlCell>
                      <ReadonlyScoreBox value={sentencePreviewTotal} />
                    </SentenceControlCell>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>

          <div className="border-t border-[#d8ccbc] bg-[#ece3d6] px-4 py-4 sm:px-6">
            <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
              <div className="text-sm text-[#564a3c]">
                {saveState === "saved"
                  ? "Annotation saved and queue advanced."
                  : saveState === "error"
                    ? "Save failed."
                    : "Raw votes are stored per reviewer. This screen is intentionally dense to keep the whole utterance visible at once."}
              </div>

              <Button
                className="h-11 rounded-full bg-[#1d1813] px-5 text-[#f7f0e4] hover:bg-[#0f0d0a]"
                disabled={!canSave || saveState === "saving"}
                onClick={saveAnnotation}
                size="lg"
              >
                {saveState === "saving" ? "Saving" : "Save and next"}
                <ChevronRight className="ml-2 size-4" />
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function ShellState({ heading, text }: { heading: string; text: string }) {
  return (
    <div className="mx-auto flex min-h-screen max-w-5xl items-center justify-center px-4 py-10 sm:px-6">
      <div className="w-full rounded-[28px] border border-[#cabca9] bg-[#ece4d6] p-8 text-[#1f1a14] shadow-[0_24px_80px_rgba(18,12,8,0.22)]">
        <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-[#776857]">
          SpeechOcean worksheet
        </p>
        <h1 className="mt-3 font-[family:var(--font-display)] text-4xl">{heading}</h1>
        <p className="mt-3 text-sm leading-6 text-[#5a4d40]">{text}</p>
      </div>
    </div>
  );
}

function StatusChip({
  label,
  tone,
}: {
  label: string;
  tone: "accent" | "danger" | "neutral";
}) {
  return (
    <span
      className={cn(
        "rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em]",
        tone === "accent" && "border-[#c99d64] bg-[#efe1c9] text-[#6a4a1f]",
        tone === "danger" && "border-[#d0a196] bg-[#f6dfda] text-[#7a2418]",
        tone === "neutral" && "border-[#d4c8b8] bg-[#f7f1e7] text-[#6d5f50]",
      )}
    >
      {label}
    </span>
  );
}

function MatrixLabelCell({
  description,
  title,
}: {
  description: string;
  title: string;
}) {
  return (
    <th className="sticky left-0 z-10 min-w-[240px] border-r border-[#d8ccbc] border-b border-[#d8ccbc] bg-inherit px-4 py-3 text-left align-top">
      <div className="font-semibold text-[#1f1a14]">{title}</div>
      <div className="mt-1 text-[12px] leading-5 text-[#6b5c4c]">{description}</div>
    </th>
  );
}

function MatrixDataCell({ children }: { children: ReactNode }) {
  return (
    <td className="min-w-[180px] border-r border-b border-[#d8ccbc] px-3 py-3 align-top last:border-r-0">
      {children}
    </td>
  );
}

function SentenceLabelCell({
  description,
  title,
}: {
  description: string;
  title: string;
}) {
  return (
    <th className="w-[320px] border-r border-b border-[#d8ccbc] bg-inherit px-4 py-4 text-left align-top">
      <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-[#776857]">
        {title}
      </div>
      <div className="mt-1 text-sm leading-6 text-[#5a4d40]">{description}</div>
    </th>
  );
}

function SentenceControlCell({ children }: { children: ReactNode }) {
  return <td className="border-b border-[#d8ccbc] px-4 py-4 align-top">{children}</td>;
}

function phoneChipClass(score: number, selected: boolean) {
  return cn(
    "inline-flex items-center gap-1 rounded-full border px-2 py-1 text-[12px] font-semibold transition-all",
    score >= 2 && "border-[#aebb8f] bg-[#edf2e3] text-[#30401f]",
    score === 1 && "border-[#d9b678] bg-[#fff0da] text-[#744d12]",
    score === 0 && "border-[#d8a093] bg-[#fbe2dd] text-[#77281b]",
    selected && "ring-2 ring-[#1f1a14]/25 ring-offset-1 ring-offset-[#faf7f0]",
  );
}

function miniScoreClass(value: number | null, selected: number) {
  return cn(
    "inline-flex h-8 min-w-8 items-center justify-center rounded-full border px-2 text-[12px] font-semibold transition-all",
    value === selected
      ? "border-[#1d1813] bg-[#1d1813] text-[#f7f0e4]"
      : "border-[#cabca8] bg-white/75 text-[#5e5144] hover:border-[#8c6c3d] hover:text-[#241d15]",
  );
}

function ValueSelect({
  options,
  value,
  onChange,
}: {
  options: number[];
  value: number | null;
  onChange: (value: number | null) => void;
}) {
  return (
    <select
      className="h-9 w-full rounded-md border border-[#cdbfae] bg-white/85 px-2 text-sm font-semibold text-[#241d15] outline-none transition focus:border-[#8c6c3d] focus:ring-2 focus:ring-[#d2a96e]/35"
      onChange={(event) =>
        onChange(event.target.value === "" ? null : Number(event.target.value))
      }
      value={value ?? ""}
    >
      <option value="">—</option>
      {options.map((option) => (
        <option key={option} value={option}>
          {option}
        </option>
      ))}
    </select>
  );
}

function ReadonlyScoreBox({ value }: { value: number | null }) {
  return (
    <div className="inline-flex min-w-[72px] items-center justify-center rounded-md border border-[#cdbfae] bg-white/88 px-3 py-2 text-base font-semibold text-[#241d15]">
      {value ?? "—"}
    </div>
  );
}

function SentenceScale({
  options,
  value,
  onSelect,
}: {
  options: number[];
  value: number | null;
  onSelect: (value: number) => void;
}) {
  return (
    <div className="flex flex-wrap gap-2">
      {options.map((score) => (
        <button
          className={cn(
            "inline-flex h-9 min-w-9 items-center justify-center rounded-full border px-2 text-sm font-semibold transition-all",
            value === score
              ? "border-[#1d1813] bg-[#1d1813] text-[#f7f0e4]"
              : "border-[#cabca8] bg-white/80 text-[#5f5245] hover:border-[#8c6c3d] hover:text-[#241d15]",
          )}
          key={score}
          onClick={() => onSelect(score)}
          type="button"
        >
          {score}
        </button>
      ))}
    </div>
  );
}
