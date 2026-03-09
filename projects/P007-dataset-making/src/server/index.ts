import {
  getAnnotationAudioResponse,
  getAnnotationHealth,
  getAnnotationNext,
  getAnnotationQueue,
  getAnnotationSession,
  getUtteranceDetail,
  initializeAnnotatorBackend,
  saveAnnotationReview,
} from "./annotations";
import type { SaveReviewPayload } from "./types";
import app from "../index.html";

function json(data: unknown, init?: ResponseInit) {
  return Response.json(data, {
    headers: {
      "cache-control": "no-store",
      ...(init?.headers ?? {}),
    },
    ...init,
  });
}

function badRequest(message: string, status = 400) {
  return json({ error: message }, { status });
}

function parseLimit(value: string | null) {
  const parsed = value ? Number(value) : 25;
  return Number.isFinite(parsed) ? parsed : 25;
}

const isDev = process.env.NODE_ENV !== "production";
const initialized = initializeAnnotatorBackend();
const port = Number(process.env.PORT ?? 3000);

const server = Bun.serve({
  port,
  development: isDev
    ? {
        hmr: true,
        console: true,
      }
    : false,
  routes: {
    "/api/health": {
      GET: async () => json(getAnnotationHealth()),
    },
    "/api/queue": {
      GET: async (req) => {
        const reviewerId = new URL(req.url).searchParams.get("reviewerId") ?? undefined;
        const limit = parseLimit(new URL(req.url).searchParams.get("limit"));
        return json(getAnnotationQueue(reviewerId, limit));
      },
    },
    "/api/session/:reviewerId": {
      GET: async (req) => {
        const reviewerId = req.params.reviewerId || undefined;
        const limit = parseLimit(new URL(req.url).searchParams.get("limit"));
        return json(getAnnotationSession(reviewerId, limit));
      },
    },
    "/api/next": {
      GET: async (req) => {
        const url = new URL(req.url);
        const next = getAnnotationNext(
          url.searchParams.get("reviewerId") ?? undefined,
          url.searchParams.get("current"),
        );
        return next ? json(next) : badRequest("No remaining utterances", 404);
      },
    },
    "/api/utterances/:utteranceId": {
      GET: async (req) => {
        const url = new URL(req.url);
        const utteranceId = req.params.utteranceId;
        if (!utteranceId) {
          return badRequest("utteranceId is required");
        }
        const detail = getUtteranceDetail(
          utteranceId,
          url.searchParams.get("reviewerId") ?? undefined,
        );
        return detail ? json(detail) : badRequest(`Unknown utterance: ${utteranceId}`, 404);
      },
    },
    "/api/utterances/:utteranceId/next": {
      GET: async (req) => {
        const url = new URL(req.url);
        const utteranceId = req.params.utteranceId;
        if (!utteranceId) {
          return badRequest("utteranceId is required");
        }
        const next = getAnnotationNext(
          url.searchParams.get("reviewerId") ?? undefined,
          utteranceId,
        );
        return json({ nextUtteranceId: next?.id ?? null });
      },
    },
    "/api/audio/:utteranceId": {
      GET: async (req) => {
        const utteranceId = req.params.utteranceId;
        if (!utteranceId) {
          return badRequest("utteranceId is required");
        }
        return getAnnotationAudioResponse(utteranceId);
      },
    },
    "/audio/:utteranceId": {
      GET: async (req) => {
        const utteranceId = req.params.utteranceId;
        if (!utteranceId) {
          return badRequest("utteranceId is required");
        }
        return getAnnotationAudioResponse(utteranceId);
      },
    },
    "/api/annotations": {
      POST: async (req) => {
        try {
          const body = (await req.json()) as SaveReviewPayload;
          if (!body?.utteranceId) {
            return badRequest("utteranceId is required");
          }
          if (!Array.isArray(body.wordScores)) {
            return badRequest("wordScores must be an array");
          }
          return json(saveAnnotationReview(body));
        } catch (error) {
          return badRequest(
            error instanceof Error ? error.message : "Unable to save review",
            422,
          );
        }
      },
    },
    "/api/reviews": {
      POST: async (req) => {
        try {
          const body = (await req.json()) as SaveReviewPayload;
          if (!body?.utteranceId) {
            return badRequest("utteranceId is required");
          }
          if (!Array.isArray(body.wordScores)) {
            return badRequest("wordScores must be an array");
          }
          return json(saveAnnotationReview(body));
        } catch (error) {
          return badRequest(
            error instanceof Error ? error.message : "Unable to save review",
            422,
          );
        }
      },
    },
    "/api/*": () => badRequest("Not found", 404),
    "/*": app,
  },
});

console.log(
  `[annotator-api] listening on http://localhost:${server.port} using ${initialized.databasePath}`,
);
