import fs from "node:fs";
import path from "node:path";
import { Database } from "bun:sqlite";
import { config } from "./config";
import type {
  DatasetUtterance,
  DatasetWord,
  SaveReviewPayload,
  SentenceMetricSet,
} from "./types";

type SplitName = (typeof config.splits)[number];

type SplitMetadata = {
  wavScp: Map<string, string>;
  text: Map<string, string>;
  utt2spk: Map<string, string>;
  spk2age: Map<string, number>;
  spk2gender: Map<string, string>;
};

let database: Database | null = null;

function readKeyValueFile(filePath: string): Map<string, string> {
  const rows = new Map<string, string>();
  const content = fs.readFileSync(filePath, "utf8");
  for (const line of content.split(/\r?\n/)) {
    if (!line.trim()) {
      continue;
    }
    const [key, ...valueParts] = line.trim().split(/\s+/);
    if (!key) {
      continue;
    }
    rows.set(key, valueParts.join(" "));
  }
  return rows;
}

function ensureDataRoot() {
  fs.mkdirSync(config.dataRoot, { recursive: true });
}

function loadSplitMetadata(split: SplitName): SplitMetadata {
  const splitRoot = path.join(config.referenceRoot, split);
  return {
    wavScp: readKeyValueFile(path.join(splitRoot, "wav.scp")),
    text: readKeyValueFile(path.join(splitRoot, "text")),
    utt2spk: readKeyValueFile(path.join(splitRoot, "utt2spk")),
    spk2age: new Map(
      Array.from(readKeyValueFile(path.join(splitRoot, "spk2age")).entries()).map(
        ([speakerId, age]) => [speakerId, Number(age)],
      ),
    ),
    spk2gender: readKeyValueFile(path.join(splitRoot, "spk2gender")),
  };
}

function normalizePhoneSequence(sequence: string): string {
  return sequence.replace(/[{}()[\]]/g, "").replace(/\s+/g, " ").trim();
}

function parsePhoneTokens(sequence: string): string[] {
  return normalizePhoneSequence(sequence).split(" ").filter(Boolean);
}

function uniqueSequences(sequences: string[]): string[] {
  return Array.from(
    new Set(
      sequences
        .map((value) => normalizePhoneSequence(value))
        .filter(Boolean),
    ),
  );
}

function loadReferenceDataset(): DatasetUtterance[] {
  const scoresPath = path.join(config.resourcePath, "scores-detail.json");
  const rawScores = JSON.parse(fs.readFileSync(scoresPath, "utf8")) as Record<
    string,
    {
      accuracy: number[];
      completeness: number[];
      fluency: number[];
      prosodic: number[];
      total: number[];
      text: string;
      words: Array<{
        text: string;
        "ref-phones": string;
        accuracy: number[];
        stress: number[];
        total: number[];
        phones: string[];
      }>;
    }
  >;

  const splitMetadata = new Map<SplitName, SplitMetadata>(
    config.splits.map((split) => [split, loadSplitMetadata(split)]),
  );

  const utterances: DatasetUtterance[] = [];

  for (const [utteranceId, detail] of Object.entries(rawScores)) {
    let selectedSplit: SplitName | null = null;
    let selectedMeta: SplitMetadata | null = null;
    for (const split of config.splits) {
      const candidate = splitMetadata.get(split)!;
      if (candidate.wavScp.has(utteranceId)) {
        selectedSplit = split;
        selectedMeta = candidate;
        break;
      }
    }

    if (!selectedSplit || !selectedMeta) {
      continue;
    }

    const speakerId = selectedMeta.utt2spk.get(utteranceId) ?? "";
    const speakerAge = selectedMeta.spk2age.get(speakerId) ?? null;
    const speakerGender = selectedMeta.spk2gender.get(speakerId) ?? null;
    const audioRelativePath = selectedMeta.wavScp.get(utteranceId) ?? "";

    const words: DatasetWord[] = detail.words.map((word) => ({
      text: word.text,
      refPhones: normalizePhoneSequence(word["ref-phones"]),
      referencePhoneTokens: parsePhoneTokens(word["ref-phones"]),
      candidatePhoneSequences: uniqueSequences([word["ref-phones"], ...word.phones]),
      scores: {
        accuracy: word.accuracy,
        stress: word.stress,
        total: word.total,
        phones: word.phones,
      },
    }));

    const referenceSentenceScores: SentenceMetricSet = {
      accuracy: detail.accuracy,
      completeness: detail.completeness,
      fluency: detail.fluency,
      prosodic: detail.prosodic,
      total: detail.total,
    };

    utterances.push({
      id: utteranceId,
      split: selectedSplit,
      speakerId,
      speakerAge,
      speakerGender,
      transcript: selectedMeta.text.get(utteranceId) ?? detail.text,
      audioRelativePath,
      audioSourcePath: path.join(config.referenceRoot, audioRelativePath),
      referenceSentenceScores,
      words,
    });
  }

  utterances.sort((left, right) => left.id.localeCompare(right.id));
  return utterances;
}

function createSchema(db: Database) {
  db.exec(`
    pragma busy_timeout = 3000;
    pragma foreign_keys = on;

    create table if not exists reviewers (
      id text primary key,
      display_name text not null,
      created_at text not null default current_timestamp
    );

    create table if not exists utterances (
      id text primary key,
      split text not null,
      speaker_id text not null,
      speaker_age integer,
      speaker_gender text,
      transcript text not null,
      audio_relative_path text not null,
      audio_source_path text not null,
      reference_sentence_scores_json text not null,
      source_corpus text not null default 'speechocean762'
    );

    create table if not exists utterance_words (
      id integer primary key autoincrement,
      utterance_id text not null references utterances(id) on delete cascade,
      position integer not null,
      text text not null,
      ref_phones text not null,
      reference_phone_tokens_json text not null,
      candidate_sequences_json text not null,
      reference_word_scores_json text not null,
      unique(utterance_id, position)
    );

    create table if not exists reviews (
      id integer primary key autoincrement,
      reviewer_id text not null references reviewers(id),
      utterance_id text not null references utterances(id) on delete cascade,
      status text not null check (status in ('in_progress', 'completed')),
      sentence_accuracy real,
      sentence_completeness real,
      sentence_fluency real,
      sentence_prosody real,
      sentence_total real,
      notes text,
      payload_json text not null,
      created_at text not null default current_timestamp,
      updated_at text not null default current_timestamp,
      unique(reviewer_id, utterance_id)
    );

    create table if not exists review_word_scores (
      id integer primary key autoincrement,
      review_id integer not null references reviews(id) on delete cascade,
      word_position integer not null,
      accuracy real,
      stress real,
      total real,
      selected_variant text,
      notes text,
      phone_votes_json text not null,
      unique(review_id, word_position)
    );

    create index if not exists idx_reviews_reviewer_status on reviews(reviewer_id, status);
    create index if not exists idx_utterance_words_utterance on utterance_words(utterance_id, position);
  `);
}

function seedReferenceData(db: Database) {
  const utteranceCount = db.query("select count(*) as count from utterances").get() as {
    count: number;
  };
  if (utteranceCount.count > 0) {
    return;
  }

  const utterances = loadReferenceDataset();
  const insertUtterance = db.query(`
    insert into utterances (
      id,
      split,
      speaker_id,
      speaker_age,
      speaker_gender,
      transcript,
      audio_relative_path,
      audio_source_path,
      reference_sentence_scores_json
    ) values (?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);
  const insertWord = db.query(`
    insert into utterance_words (
      utterance_id,
      position,
      text,
      ref_phones,
      reference_phone_tokens_json,
      candidate_sequences_json,
      reference_word_scores_json
    ) values (?, ?, ?, ?, ?, ?, ?)
  `);
  const insertReviewer = db.query(
    "insert or ignore into reviewers (id, display_name) values (?, ?)",
  );

  const seedTransaction = db.transaction(() => {
    insertReviewer.run(config.defaultReviewerId, config.defaultReviewerName);
    for (const utterance of utterances) {
      insertUtterance.run(
        utterance.id,
        utterance.split,
        utterance.speakerId,
        utterance.speakerAge,
        utterance.speakerGender,
        utterance.transcript,
        utterance.audioRelativePath,
        utterance.audioSourcePath,
        JSON.stringify(utterance.referenceSentenceScores),
      );

      utterance.words.forEach((word, index) => {
        insertWord.run(
          utterance.id,
          index,
          word.text,
          word.refPhones,
          JSON.stringify(word.referencePhoneTokens),
          JSON.stringify(word.candidatePhoneSequences),
          JSON.stringify(word.scores),
        );
      });
    }
  });

  seedTransaction();
}

export function getDb() {
  if (database) {
    return database;
  }
  ensureDataRoot();
  database = new Database(config.databasePath, { create: true });
  createSchema(database);
  seedReferenceData(database);
  return database;
}

export function ensureReviewer(reviewerId?: string, reviewerName?: string) {
  const db = getDb();
  const id = reviewerId?.trim() || config.defaultReviewerId;
  const displayName = reviewerName?.trim() || config.defaultReviewerName;
  db.query("insert or ignore into reviewers (id, display_name) values (?, ?)").run(
    id,
    displayName,
  );
  return { reviewerId: id, reviewerName: displayName };
}

export function listQueue(reviewerId?: string, limit = 25) {
  const db = getDb();
  const safeLimit = Math.max(1, Math.min(limit, 100));
  const reviewer = ensureReviewer(reviewerId);

  const queueRows = db
    .query(`
      select
        u.id,
        u.split,
        u.transcript,
        u.speaker_id as speakerId,
        u.speaker_age as speakerAge,
        u.speaker_gender as speakerGender,
        coalesce(r.status, 'unstarted') as reviewStatus,
        r.updated_at as updatedAt
      from utterances u
      left join reviews r
        on r.utterance_id = u.id
       and r.reviewer_id = ?
      order by
        case coalesce(r.status, 'unstarted')
          when 'in_progress' then 0
          when 'unstarted' then 1
          else 2
        end,
        u.id
      limit ?
    `)
    .all(reviewer.reviewerId, safeLimit) as Array<Record<string, unknown>>;

  const progress = db
    .query(`
      select
        count(*) as totalUtterances,
        sum(case when r.status = 'completed' then 1 else 0 end) as completed,
        sum(case when r.status = 'in_progress' then 1 else 0 end) as inProgress
      from utterances u
      left join reviews r
        on r.utterance_id = u.id
       and r.reviewer_id = ?
    `)
    .get(reviewer.reviewerId) as {
    totalUtterances: number;
    completed: number | null;
    inProgress: number | null;
  };

  return {
    reviewer,
    progress: {
      totalUtterances: progress.totalUtterances,
      completed: progress.completed ?? 0,
      inProgress: progress.inProgress ?? 0,
      remaining:
        progress.totalUtterances - (progress.completed ?? 0) - (progress.inProgress ?? 0),
    },
    queue: queueRows,
  };
}

export function getUtterance(utteranceId: string, reviewerId?: string) {
  const db = getDb();
  const reviewer = ensureReviewer(reviewerId);
  const utterance = db
    .query(`
      select
        id,
        split,
        speaker_id as speakerId,
        speaker_age as speakerAge,
        speaker_gender as speakerGender,
        transcript,
        audio_relative_path as audioRelativePath,
        audio_source_path as audioSourcePath,
        reference_sentence_scores_json as referenceSentenceScoresJson
      from utterances
      where id = ?
    `)
    .get(utteranceId) as Record<string, string | number | null> | null;

  if (!utterance) {
    return null;
  }

  const words = db
    .query(`
      select
        position,
        text,
        ref_phones as refPhones,
        reference_phone_tokens_json as referencePhoneTokensJson,
        candidate_sequences_json as candidateSequencesJson,
        reference_word_scores_json as referenceWordScoresJson
      from utterance_words
      where utterance_id = ?
      order by position
    `)
    .all(utteranceId) as Array<Record<string, string | number>>;

  const review = db
    .query(`
      select
        id,
        status,
        sentence_accuracy as accuracy,
        sentence_completeness as completeness,
        sentence_fluency as fluency,
        sentence_prosody as prosody,
        sentence_total as total,
        notes,
        payload_json as payloadJson,
        updated_at as updatedAt
      from reviews
      where utterance_id = ?
        and reviewer_id = ?
    `)
    .get(utteranceId, reviewer.reviewerId) as Record<string, string | number | null> | null;

  const reviewWordScores = review
    ? (db
        .query(`
          select
            word_position as wordPosition,
            accuracy,
            stress,
            total,
            selected_variant as selectedVariant,
            notes,
            phone_votes_json as phoneVotesJson
          from review_word_scores
          where review_id = ?
          order by word_position
        `)
        .all(Number(review.id)) as Array<Record<string, string | number | null>>)
    : [];

  return {
    reviewer,
    id: utterance.id,
    split: utterance.split,
    speakerId: utterance.speakerId,
    speakerAge: utterance.speakerAge,
    speakerGender: utterance.speakerGender,
    transcript: utterance.transcript,
    audioRelativePath: utterance.audioRelativePath,
    audioSourcePath: utterance.audioSourcePath,
    audioUrl: `/api/audio/${utteranceId}`,
    sentenceScores: JSON.parse(String(utterance.referenceSentenceScoresJson)),
    words: words.map((word) => ({
      position: word.position,
      text: word.text,
      refPhones: word.refPhones,
      referencePhoneTokens: JSON.parse(String(word.referencePhoneTokensJson)),
      candidateSequences: JSON.parse(String(word.candidateSequencesJson)),
      referenceWordScores: JSON.parse(String(word.referenceWordScoresJson)),
    })),
    existingAnnotation: review
      ? {
          id: review.id,
          status: review.status,
          updatedAt: review.updatedAt,
          notes: review.notes,
          sentenceScores: {
            accuracy: review.accuracy,
            completeness: review.completeness,
            fluency: review.fluency,
            prosody: review.prosody,
            total: review.total,
          },
          payload: JSON.parse(String(review.payloadJson)),
          wordScores: reviewWordScores.map((row) => ({
            wordPosition: row.wordPosition,
            accuracy: row.accuracy,
            stress: row.stress,
            total: row.total,
            selectedVariant: row.selectedVariant,
            notes: row.notes,
            phoneVotes: JSON.parse(String(row.phoneVotesJson)),
          })),
        }
      : null,
  };
}

export function getNextUtterance(
  reviewerId?: string,
  currentUtteranceId?: string | null,
) {
  const db = getDb();
  const reviewer = ensureReviewer(reviewerId);

  const currentClause = currentUtteranceId ? "and u.id > ?" : "";
  const nextRow = db
    .query(`
      select u.id
      from utterances u
      left join reviews r
        on r.utterance_id = u.id
       and r.reviewer_id = ?
      where coalesce(r.status, 'unstarted') != 'completed'
      ${currentClause}
      order by
        case coalesce(r.status, 'unstarted')
          when 'in_progress' then 0
          else 1
        end,
        u.id
      limit 1
    `)
    .get(
      ...(currentUtteranceId ? [reviewer.reviewerId, currentUtteranceId] : [reviewer.reviewerId]),
    ) as { id: string } | null;

  if (nextRow) {
    return getUtterance(nextRow.id, reviewer.reviewerId);
  }

  const fallbackRow = db
    .query(`
      select u.id
      from utterances u
      left join reviews r
        on r.utterance_id = u.id
       and r.reviewer_id = ?
      where coalesce(r.status, 'unstarted') != 'completed'
      order by u.id
      limit 1
    `)
    .get(reviewer.reviewerId) as { id: string } | null;

  return fallbackRow ? getUtterance(fallbackRow.id, reviewer.reviewerId) : null;
}

export function saveReview(input: SaveReviewPayload) {
  const db = getDb();
  const reviewer = ensureReviewer(input.reviewerId, input.reviewerName);
  const existing = db
    .query("select id from reviews where reviewer_id = ? and utterance_id = ?")
    .get(reviewer.reviewerId, input.utteranceId) as { id: number } | null;

  const payloadJson = JSON.stringify(input);

  const saveTransaction = db.transaction(() => {
    let reviewId = existing?.id ?? null;
    if (reviewId) {
      db.query(`
        update reviews
        set
          status = ?,
          sentence_accuracy = ?,
          sentence_completeness = ?,
          sentence_fluency = ?,
          sentence_prosody = ?,
          sentence_total = ?,
          notes = ?,
          payload_json = ?,
          updated_at = current_timestamp
        where id = ?
      `).run(
        input.status,
        input.sentenceScores.accuracy,
        input.sentenceScores.completeness,
        input.sentenceScores.fluency,
        input.sentenceScores.prosody,
        input.sentenceScores.total,
        input.notes ?? null,
        payloadJson,
        reviewId,
      );
      db.query("delete from review_word_scores where review_id = ?").run(reviewId);
    } else {
      const result = db.query(`
        insert into reviews (
          reviewer_id,
          utterance_id,
          status,
          sentence_accuracy,
          sentence_completeness,
          sentence_fluency,
          sentence_prosody,
          sentence_total,
          notes,
          payload_json
        ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).run(
        reviewer.reviewerId,
        input.utteranceId,
        input.status,
        input.sentenceScores.accuracy,
        input.sentenceScores.completeness,
        input.sentenceScores.fluency,
        input.sentenceScores.prosody,
        input.sentenceScores.total,
        input.notes ?? null,
        payloadJson,
      );
      reviewId = Number(result.lastInsertRowid);
    }

    const insertWord = db.query(`
      insert into review_word_scores (
        review_id,
        word_position,
        accuracy,
        stress,
        total,
        selected_variant,
        notes,
        phone_votes_json
      ) values (?, ?, ?, ?, ?, ?, ?, ?)
    `);

    for (const word of input.wordScores) {
      insertWord.run(
        reviewId,
        word.wordPosition,
        word.accuracy,
        word.stress,
        word.total,
        word.selectedVariant,
        word.notes ?? null,
        JSON.stringify(word.phoneVotes),
      );
    }

    return reviewId;
  });

  const reviewId = saveTransaction();
  return {
    reviewId,
    reviewerId: reviewer.reviewerId,
    utteranceId: input.utteranceId,
    status: input.status,
    next: getNextUtterance(reviewer.reviewerId, input.utteranceId),
  };
}

export function getAudioFile(utteranceId: string) {
  const db = getDb();
  const row = db
    .query("select audio_source_path as audioSourcePath from utterances where id = ?")
    .get(utteranceId) as { audioSourcePath: string } | null;
  return row ? Bun.file(row.audioSourcePath) : null;
}

export function initializeDatabase() {
  getDb();
  return {
    databasePath: config.databasePath,
    utteranceCount: (getDb().query("select count(*) as count from utterances").get() as {
      count: number;
    }).count,
  };
}
