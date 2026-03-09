-- SpeechLab Database Schema
-- Transcription annotation platform data model

CREATE TABLE IF NOT EXISTS corpus (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  file        TEXT    NOT NULL UNIQUE,
  audio_size  TEXT    NOT NULL,
  total_length TEXT   NOT NULL,
  transcript  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS corpus_words (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  corpus_id       INTEGER NOT NULL REFERENCES corpus(id) ON DELETE CASCADE,
  position        INTEGER NOT NULL,
  word            TEXT    NOT NULL,
  phonetic        TEXT    NOT NULL,
  phoneme_score   INTEGER NOT NULL CHECK (phoneme_score IN (0, 1)),
  phoneme_accuracy INTEGER NOT NULL CHECK (phoneme_accuracy BETWEEN 0 AND 10),
  stress_accuracy INTEGER NOT NULL CHECK (stress_accuracy IN (5, 10)),
  UNIQUE (corpus_id, position)
);

CREATE TABLE IF NOT EXISTS corpus_sentence_scores (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  corpus_id       INTEGER NOT NULL UNIQUE REFERENCES corpus(id) ON DELETE CASCADE,
  accuracy        INTEGER NOT NULL CHECK (accuracy BETWEEN 0 AND 10),
  fluency         INTEGER NOT NULL CHECK (fluency BETWEEN 0 AND 10),
  prosody         INTEGER NOT NULL CHECK (prosody BETWEEN 0 AND 10),
  integrity_words INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
  id          INTEGER PRIMARY KEY,
  project     INTEGER NOT NULL,
  batch       INTEGER NOT NULL,
  total       INTEGER NOT NULL,
  completed   INTEGER NOT NULL DEFAULT 0,
  audio_size  TEXT    NOT NULL,
  status      TEXT    NOT NULL CHECK (status IN ('completed', 'in-progress', 'pending')),
  due_date    TEXT    NOT NULL,
  lang        TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
  id          INTEGER PRIMARY KEY,
  title       TEXT    NOT NULL,
  lang        TEXT    NOT NULL,
  type        TEXT    NOT NULL,
  rate        TEXT    NOT NULL,
  tasks       INTEGER NOT NULL,
  deadline    TEXT    NOT NULL,
  difficulty  TEXT    NOT NULL CHECK (difficulty IN ('Easy', 'Medium', 'Hard')),
  description TEXT    NOT NULL,
  slots       INTEGER NOT NULL,
  applied     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS project_tags (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  tag        TEXT    NOT NULL,
  UNIQUE (project_id, tag)
);

CREATE TABLE IF NOT EXISTS project_breakdown (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  project   TEXT    NOT NULL UNIQUE,
  completed INTEGER NOT NULL,
  total     INTEGER NOT NULL,
  avg_score REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS user_stats (
  id    INTEGER PRIMARY KEY AUTOINCREMENT,
  label TEXT    NOT NULL,
  value TEXT    NOT NULL,
  sub   TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS weekly_activity (
  id    INTEGER PRIMARY KEY AUTOINCREMENT,
  day   TEXT    NOT NULL,
  tasks INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS monthly_scores (
  id   INTEGER PRIMARY KEY AUTOINCREMENT,
  week TEXT    NOT NULL,
  avg  REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS user_profile (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  name            TEXT    NOT NULL,
  email           TEXT    NOT NULL,
  user_id         TEXT    NOT NULL,
  timezone        TEXT    NOT NULL,
  primary_lang    TEXT    NOT NULL,
  proficiency     TEXT    NOT NULL,
  notify_new_task INTEGER NOT NULL DEFAULT 1,
  notify_score    INTEGER NOT NULL DEFAULT 1,
  notify_payment  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS user_secondary_langs (
  id      INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL REFERENCES user_profile(id) ON DELETE CASCADE,
  lang    TEXT    NOT NULL,
  UNIQUE (user_id, lang)
);
