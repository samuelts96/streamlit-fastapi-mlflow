FROM python:3.12-slim

RUN pip install --no-cache-dir mlflow==3.17.1

# MLflow 3 rejects requests whose Host header it doesn't recognise (403 "Invalid Host header").
# Other containers reach this server as "mlflow:5001"; browsers as "localhost:5001".
ENV MLFLOW_ALLOWED_HOSTS="mlflow,mlflow:*,localhost,localhost:*,127.0.0.1,127.0.0.1:*"

EXPOSE 5001

# Runs and the registry live in SQLite; model files are served by this server (artifact proxy).
# /mlflow is a named volume, so everything survives restarts.
CMD mlflow server \
    --backend-store-uri sqlite:////mlflow/mlflow.db \
    --artifacts-destination /mlflow/artifacts \
    --host 0.0.0.0 --port 5001 \
    --allowed-hosts "$MLFLOW_ALLOWED_HOSTS"
