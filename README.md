# Streamlit + FastAPI + MLflow

A small, end-to-end ML serving demo: a **Streamlit** UI calls a **FastAPI** service, which serves the model
that the **MLflow** model registry marks as **champion**. Everything runs locally with one command:

```bash
docker compose up --build
```

| Open | URL |
|---|---|
| Streamlit app | http://localhost:8501 |
| FastAPI docs (Swagger) | http://localhost:8000/docs |
| MLflow UI | http://localhost:5001 |

## How it fits together

```
 Streamlit (:8501)  ──POST /predict──▶  FastAPI (:8000)  ──loads models:/iris-classifier@champion──▶  MLflow (:5001)
                                                                                                         ▲
                                                 trainer (one-off job): trains, logs, registers, sets the alias
```

| Service | What it does | Port |
|---|---|---|
| `mlflow` | Tracking server and model registry (SQLite + artifact store in a named volume) | 5001 |
| `trainer` | Trains 3 candidate models on iris, logs and tags them, registers the best, sets `@champion` (or `@challenger` if it doesn't beat the current champion), then exits | none |
| `backend` | FastAPI: loads the champion **once at startup**; `/health`, `/predict`, `/model-info`, `/reload`, `/message` | 8000 |
| `frontend` | Streamlit: a form that calls `/predict` and shows which model is serving | 8501 |

The frontend only knows the backend's URL (`BACKEND_URL`); the backend only knows the MLflow URL
(`MLFLOW_TRACKING_URI`) and the model name. Each part can be replaced, rewritten or scaled on its own.

**Why MLflow on 5001?** On macOS, AirPlay Receiver already listens on port 5000. If any port is taken on
your machine, set `MLFLOW_PORT`, `API_PORT` or `FRONTEND_PORT` in `.env` (see `.env.example`).

## API

| Method | Route | Returns |
|---|---|---|
| GET | `/health` | `200 {"status":"ok","model_loaded":true,"model_version":"1"}`, or `503` if the model isn't loaded |
| POST | `/predict` | `200` species and class probabilities · `422` invalid input (Pydantic) · `400` implausible flower (business rule) · `503` no model · `500` unexpected error |
| GET | `/model-info` | Model name, version, alias, run, metrics, expected features (in order) and classes |
| POST | `/reload` | Re-loads whichever version `@champion` points at now (after promoting or rolling back) |
| GET | `/message` | Simple connectivity check |

## Running it

See [RUNBOOK.md](RUNBOOK.md): Docker Compose (one command), plain `docker run`, or manually with `uv`,
plus how to retrain, promote and roll back.

## Layout

```
mlflow/      MLflow server image
trainer/     train.py: train, log, register, set the champion alias
backend/     FastAPI app
frontend/    Streamlit app
docker-compose.yml
RUNBOOK.md
```
