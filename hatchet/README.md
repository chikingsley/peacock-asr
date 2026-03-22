# Hatchet Local Runtime

This folder contains the repo-local Hatchet runtime for `peacock-asr`.

## Reproducible Runtime

The Hatchet app runtime is owned by `docker-compose.yml` in this folder. That keeps
the public URL, cookie domain, ports, and database wiring in-repo instead of inside
an ephemeral CLI-generated container.

### Start the runtime

Copy the env example once:

```bash
cp hatchet/.env.example hatchet/.env
```

Start the stack:

```bash
docker compose --env-file hatchet/.env -f hatchet/docker-compose.yml up -d
```

Stop the stack:

```bash
docker compose --env-file hatchet/.env -f hatchet/docker-compose.yml down
```

The stack exposes:

- dashboard/app: `https://hatchet.peacockery.studio`
- local dashboard port: `http://localhost:8898`
- gRPC: `localhost:7077`

### Login

Default local admin credentials:

- email: `admin@example.com`
- password: `Admin123!!`

Those defaults come from the seeded Hatchet config volume created by the local
runtime. If you want different credentials, reset the Hatchet volumes and recreate the
stack.

## Worker

Run the repo worker from this directory:

```bash
hatchet worker dev --profile peacock-local --no-reload
```

The worker currently registers:

- `p003-compact-backbones-xlsr53-compare`: runs the XLSR-53 GOPT baseline with
  the Python scalar backend, prewarms `k2`, then runs the same baseline with
  `k2`, storing results under `projects/P003-compact-backbones/experiments`
- `p001-xlsr53-phase1-a3-gopt`: the older sweep-based XLSR-53 workflow

If you do not already have a `peacock-local` profile on this machine, create one after
logging into the Hatchet UI and generating an API token:

```bash
hatchet profile add
```

Use:

- server URL: `https://hatchet.peacockery.studio`
- token: API token from the Hatchet UI

If the worker fails with `invalid auth token` after recreating the Hatchet server,
generate a fresh API token in the UI and update the local profile:

```bash
hatchet profile update --name peacock-local --token <new-token>
```

## Trigger The Compare Run

Trigger the sequential compare workflow manually:

```bash
hatchet trigger manual \
  --profile peacock-local \
  --workflow p003-compact-backbones-xlsr53-compare \
  --json hatchet/p003-compact-backbones-input.json \
  --output json
```

That workflow will:

1. run one full XLSR-53 `--gopt` baseline with the Python scalar backend
2. prewarm the `k2` topology cache
3. run the same XLSR-53 `--gopt` baseline with `k2`
4. summarize elapsed time and metrics

## Public Access

To expose the dashboard at `https://hatchet.peacockery.studio`, see
`../tunnel/README.md` for Cloudflare Tunnel setup.
