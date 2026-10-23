"""Train candidate models, log them to MLflow, register the best and point the "champion" alias at it.

Run once before starting the backend (Compose does this automatically). Running it again trains new
candidates: the best one becomes a new version, and it only takes over as champion if it beats the
current champion by MARGIN; otherwise it gets the "challenger" alias.
"""

import os
import time
import urllib.request

import mlflow
from mlflow import MlflowClient
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "iris-demo")
MODEL_NAME = os.getenv("MODEL_NAME", "iris-classifier")
MARGIN = float(os.getenv("PROMOTION_MARGIN", "0.0"))   # challenger must beat champion F1 by more than this
SEED = int(os.getenv("SEED", "42"))

# The API sends features in exactly this order and with these names.
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def wait_for_mlflow(timeout_s: int = 90) -> None:
    deadline = time.time() + timeout_s
    while True:
        try:
            urllib.request.urlopen(f"{TRACKING_URI}/health", timeout=3)
            return
        except OSError:
            if time.time() > deadline:
                raise SystemExit(f"MLflow server not reachable at {TRACKING_URI}")
            time.sleep(2)


def load_data():
    iris = load_iris(as_frame=True)
    X = iris.data.copy()
    X.columns = FEATURES
    y = iris.target.map(dict(enumerate(iris.target_names)))   # species names, so the model predicts names
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)


CANDIDATES = {
    "logreg-C0.1": ("logistic_regression", LogisticRegression(C=0.1, max_iter=1000)),
    "logreg-C1": ("logistic_regression", LogisticRegression(C=1.0, max_iter=1000)),
    "svm-rbf": ("svm", SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED)),
}


def train_candidates(X_train, X_test, y_train, y_test):
    results = []
    for run_name, (algorithm, estimator) in CANDIDATES.items():
        model = Pipeline([("scale", StandardScaler()), ("model", estimator)])
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags({"model_family": "ml", "algorithm": algorithm, "dataset": "iris", "trigger": "train.py"})
            mlflow.log_params({k: v for k, v in estimator.get_params().items() if k in {"C", "kernel", "max_iter"}})
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            # Rounded once here, so the value logged, stored on the model version and compared is identical.
            metrics = {"test_accuracy": round(accuracy_score(y_test, pred), 6),
                       "test_f1_macro": round(f1_score(y_test, pred, average="macro"), 6)}
            mlflow.log_metrics(metrics)
            info = mlflow.sklearn.log_model(model, name="model", input_example=X_train.head(2))
            results.append({"run_name": run_name, "run_id": run.info.run_id, "model_uri": info.model_uri, **metrics})
            print(f"{run_name:<12} accuracy {metrics['test_accuracy']:.3f}  F1 {metrics['test_f1_macro']:.3f}")
    return results


def champion_f1(client: MlflowClient):
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except mlflow.exceptions.MlflowException:
        return None, None
    return mv, float(mv.tags.get("test_f1_macro", "nan"))


def main() -> None:
    wait_for_mlflow()
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()

    results = train_candidates(*load_data())
    best = max(results, key=lambda r: (r["test_f1_macro"], r["test_accuracy"]))

    mv = mlflow.register_model(best["model_uri"], MODEL_NAME)
    for key in ("test_f1_macro", "test_accuracy"):
        client.set_model_version_tag(MODEL_NAME, mv.version, key, str(best[key]))
    client.set_model_version_tag(MODEL_NAME, mv.version, "run_name", best["run_name"])

    current, current_f1 = champion_f1(client)
    if current is None or best["test_f1_macro"] > current_f1 + MARGIN:
        client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
        print(f"Champion: {MODEL_NAME} v{mv.version} ({best['run_name']}, F1 {best['test_f1_macro']:.3f})")
    else:
        client.set_registered_model_alias(MODEL_NAME, "challenger", mv.version)
        print(f"Kept champion v{current.version} (F1 {current_f1:.3f}); v{mv.version} is the challenger")


if __name__ == "__main__":
    main()
