# Runbook

Three ways to run the same system: **Docker Compose** (recommended), plain **Docker**, or **manually with uv**.

| Service | URL | Port |
|---|---|---|
| Streamlit | http://localhost:8501 | 8501 |
| FastAPI (docs at `/docs`) | http://localhost:8000 | 8000 |
| MLflow UI | http://localhost:5001 | 5001 |

`BACKEND_URL` is how the frontend finds the backend, and `MLFLOW_TRACKING_URI` is how the backend and
trainer find MLflow. In Docker they're service names (`http://backend:8000`, `http://mlflow:5001`); once
deployed, set them to whatever address each service is reachable on (ALB DNS name, Cloud Map name, task IP).

---

## 1. Docker Compose

```bash
docker compose up --build
```

Start order is automatic: `mlflow` (healthy) → `trainer` (registers the champion, then exits) →
`backend` (healthy) → `frontend`.

**Port already taken?** Copy `.env.example` to `.env` and change `MLFLOW_PORT`, `API_PORT` or
`FRONTEND_PORT`, or set one inline, e.g. `API_PORT=8080 docker compose up --build`.

Teardown:

```bash
docker compose down          # keeps runs and models (named volume)
docker compose down -v       # also deletes all MLflow data
```

### Retrain, promote, roll back

```bash
docker compose run --rm trainer                              # new version; becomes champion only if it beats the current one
docker compose run --rm -e PROMOTION_MARGIN=0.01 trainer     # must win by more than 0.01 F1
curl -X POST http://localhost:8000/reload                    # serve whatever @champion points at now
```

Move the alias by hand (promote or roll back), then reload:

```bash
docker compose exec backend python -c "from mlflow import MlflowClient; MlflowClient('http://mlflow:5001').set_registered_model_alias('iris-classifier', 'champion', '1')"
curl -X POST http://localhost:8000/reload
```

---

## 2. Docker (no Compose)

```bash
docker network create appnet
docker volume create mlflow-data

docker build -t app-mlflow ./mlflow
docker build -t app-trainer ./trainer
docker build -t app-backend ./backend
docker build -t app-frontend ./frontend

docker run -d --name mlflow --network appnet -p 5001:5001 -v mlflow-data:/mlflow app-mlflow
docker run --rm --network appnet -e MLFLOW_TRACKING_URI=http://mlflow:5001 app-trainer
docker run -d --name backend --network appnet -p 8000:8000 -e MLFLOW_TRACKING_URI=http://mlflow:5001 app-backend
docker run -d --name frontend --network appnet -p 8501:8501 -e BACKEND_URL=http://backend:8000 \
  -e PUBLIC_API_URL=http://localhost:8000 -e MLFLOW_UI_URL=http://localhost:5001 app-frontend
```

Teardown:

```bash
docker rm -f frontend backend mlflow
docker network rm appnet
docker volume rm mlflow-data    # deletes all MLflow data
```

---

## 3. Manual (uv)

Four terminals, from the repo root. The defaults already point at these ports.

**1. MLflow server**

```bash
cd trainer
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5001
```

**2. Train and register the champion** (runs once and exits)

```bash
cd trainer
uv run python train.py
```

**3. Backend**

```bash
cd backend
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

**4. Frontend**

```bash
cd frontend
uv run streamlit run app.py
```

Stop with Ctrl+C in each terminal.

---

## Smoke test

```bash
curl http://localhost:8000/health
curl http://localhost:8000/model-info
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

Expected: `{"species":"setosa", ...}` with status 200. Send `"sepal_length": -1` to see a 422, or
`"petal_width": 2.0` with `"petal_length": 1.0` to see the 400 business-rule error.

---

## Gotchas this demo handles

- **MLflow 3 rejects unknown host names** (`403 Invalid Host header`). The server allows `mlflow:*`,
  `localhost:*` and `127.0.0.1:*` (see `mlflow/Dockerfile`).
- **Inside Docker, use service names**, not `localhost`: `http://mlflow:5001`, `http://backend:8000`.
  `localhost` inside a container means that container itself.
- **Bind to `0.0.0.0` inside containers**, otherwise the published port can't reach the app.
- **Load the model once at startup**, not per request. `/reload` swaps it when the alias moves.
- **Feature order:** the API builds the DataFrame with the same column names and order as training.
- **Port 5000 on macOS** is used by AirPlay Receiver, which is why MLflow runs on 5001.
