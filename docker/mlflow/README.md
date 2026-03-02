# Local MLflow stack for peacock-asr

This mirrors the upstream MLflow `docker-compose` layout (Postgres + S3-compatible artifacts) and is the default local tracking setup for this repo.

## 1) Start MLflow

```bash
cd docker/mlflow
cp .env.example .env
docker compose up -d
docker compose ps
```

UI and API are at `http://localhost:15000`.

For external hostnames (for example behind Cloudflare Tunnel), set the documented MLflow server security vars in `.env`:
- `MLFLOW_SERVER_ALLOWED_HOSTS`
- `MLFLOW_SERVER_CORS_ALLOWED_ORIGINS`

## 2) Wire peacock-asr runs to MLflow

`peacock-asr run` auto-logs to MLflow when `MLFLOW_TRACKING_URI` is set.

```bash
export MLFLOW_TRACKING_URI=http://localhost:15000
export MLFLOW_EXPERIMENT_NAME=peacock-asr
```

Then run as normal:

```bash
uv run peacock-asr run --backend original --gopt --no-cache
```

What gets logged:
- run params (backend, mode, cache/device/workers, dataset revision, split sizes)
- eval metrics (`pcc`, CI, `mse`, `n_phones_eval`)
- per-phone PCC metrics
- `reports/*.json` evaluation artifact

## 3) Stop/reset

```bash
docker compose down
# full reset (metadata + artifacts):
docker compose down -v
```
