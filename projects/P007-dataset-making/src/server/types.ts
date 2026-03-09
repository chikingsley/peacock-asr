export type SentenceMetricSet = {
  accuracy: number[];
  completeness: number[];
  fluency: number[];
  prosodic: number[];
  total: number[];
};

export type ReferenceWordScores = {
  accuracy: number[];
  stress: number[];
  total: number[];
  phones: string[];
};

export type DatasetWord = {
  text: string;
  refPhones: string;
  referencePhoneTokens: string[];
  candidatePhoneSequences: string[];
  scores: ReferenceWordScores;
};

export type DatasetUtterance = {
  id: string;
  split: "train" | "test";
  speakerId: string;
  speakerAge: number | null;
  speakerGender: string | null;
  transcript: string;
  audioRelativePath: string;
  audioSourcePath: string;
  referenceSentenceScores: SentenceMetricSet;
  words: DatasetWord[];
};

export type ReviewPhoneVote = {
  phoneIndex: number;
  canonicalPhone: string;
  label: "correct" | "accented" | "incorrect" | "unclear";
  observedPhone?: string | null;
  note?: string | null;
};

export type ReviewWordPayload = {
  wordPosition: number;
  accuracy: number | null;
  stress: number | null;
  total: number | null;
  selectedVariant: string | null;
  notes?: string | null;
  phoneVotes: ReviewPhoneVote[];
};

export type SaveReviewPayload = {
  reviewerId?: string;
  reviewerName?: string;
  utteranceId: string;
  status: "in_progress" | "completed";
  notes?: string | null;
  sentenceScores: {
    accuracy: number | null;
    completeness: number | null;
    fluency: number | null;
    prosody: number | null;
    total: number | null;
  };
  wordScores: ReviewWordPayload[];
};
