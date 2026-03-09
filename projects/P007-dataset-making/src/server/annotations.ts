import { config } from "./config";
import {
  getAudioFile,
  getNextUtterance,
  getUtterance,
  initializeDatabase,
  listQueue,
  saveReview,
} from "./db";
import type { SaveReviewPayload } from "./types";

function json(data: unknown, init?: ResponseInit) {
  return Response.json(data, init);
}

function badRequest(message: string, status = 400) {
  return json({ error: message }, { status });
}

export function initializeAnnotatorBackend() {
  return initializeDatabase();
}

export function getAnnotationHealth() {
  return {
    ok: true,
    databasePath: config.databasePath,
  };
}

export function getAnnotationQueue(reviewerId?: string, limit = 25) {
  return listQueue(reviewerId, limit);
}

export function getAnnotationSession(reviewerId?: string, limit = 25) {
  const session = listQueue(reviewerId, limit);
  const current = getNextUtterance(reviewerId, null);
  return {
    reviewer: session.reviewer,
    queueSummary: session.progress,
    currentUtteranceId: current?.id ?? null,
  };
}

export function getAnnotationNext(reviewerId?: string, currentUtteranceId?: string | null) {
  return getNextUtterance(reviewerId, currentUtteranceId);
}

export function getUtteranceDetail(utteranceId: string, reviewerId?: string) {
  return getUtterance(utteranceId, reviewerId);
}

export async function getAnnotationAudioResponse(utteranceId: string) {
  const audioFile = getAudioFile(utteranceId);
  if (!audioFile || !(await audioFile.exists())) {
    return badRequest(`Unknown audio for utterance: ${utteranceId}`, 404);
  }

  return new Response(audioFile, {
    headers: {
      "content-type": audioFile.type || "audio/wav",
      "cache-control": "public, max-age=3600",
    },
  });
}

export function saveAnnotationReview(input: SaveReviewPayload) {
  const saved = saveReview(input);
  return {
    annotation: {
      reviewId: saved.reviewId,
      reviewerId: saved.reviewerId,
      utteranceId: saved.utteranceId,
      status: saved.status,
      sentenceScores: input.sentenceScores,
      wordCount: input.wordScores.length,
    },
    nextUtteranceId: saved.next?.id ?? null,
  };
}
