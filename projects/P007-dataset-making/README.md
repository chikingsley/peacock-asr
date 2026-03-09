# P007 Dataset Making

Modern rebuild of the SpeechOcean uTrans annotation workflow on the `buncn` full-stack shape: Bun server, React Router client, Tailwind v4 styling, and SQLite for fast local iteration.

## What is here

- `src/`: app, routes, shared UI, and the Bun server runtime
- `server/data/`: local SQLite artifacts
- `projects/P007-dataset-making/references/speechocean762/`: local seed corpus used for the first working version

The current build follows the original review flow closely:

1. Play the whole utterance.
2. Review canonical phone sequence choices per word.
3. Score phones and word-level accuracy/stress.
4. Score sentence-level accuracy, completeness, fluency, and prosody.
5. Save and advance to the next utterance.

## Commands

```bash
bun install
bun run seed
bun run dev
```

Production and verification:

```bash
bun run typecheck
bun run build
bun run start
```

## Ports

- Default app server: `http://localhost:3000`

If `3000` is already busy on this machine, run with an override:

```bash
PORT=3317 bun run dev
```

## Project Shape

This project intentionally follows the same core conventions as `buncn`:

- `Bun.serve()` in [`src/server/index.ts`](/home/simon/github/peacock-asr/projects/P007-dataset-making/src/server/index.ts)
- SPA shell imported from [`src/index.html`](/home/simon/github/peacock-asr/projects/P007-dataset-making/src/index.html)
- React Router entry in [`src/main.tsx`](/home/simon/github/peacock-asr/projects/P007-dataset-making/src/main.tsx)
- App code under [`src/app`](/home/simon/github/peacock-asr/projects/P007-dataset-making/src/app)
- Tailwind + shadcn-style component setup via [`components.json`](/home/simon/github/peacock-asr/projects/P007-dataset-making/components.json)

The deliberate deviation is the database layer: this app uses SQLite instead of Postgres so annotation review stays simple to seed, snapshot, and run locally.
