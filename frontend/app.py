import os

import pandas as pd
import requests
import streamlit as st

# How this container/process reaches the API (in Compose: http://backend:8000).
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
# Links shown to the user's browser, which can't resolve Compose service names.
PUBLIC_API_URL = os.getenv("PUBLIC_API_URL", BACKEND_URL)
MLFLOW_UI_URL = os.getenv("MLFLOW_UI_URL", "http://localhost:5001")

st.set_page_config(page_title="Iris predictor", page_icon="🌸")
st.title("Iris species predictor")
st.caption("Streamlit → FastAPI → MLflow model registry (champion alias)")

# --- Sidebar: which model is serving, and links to the other services -------------------------
with st.sidebar:
    st.header("Serving model")
    try:
        info = requests.get(f"{BACKEND_URL}/model-info", timeout=5)
        if info.status_code == 200:
            m = info.json()
            st.metric("Model", f"{m['model_name']} v{m['version']}")
            st.write(f"Alias: `@{m['alias']}` · run: `{m.get('run_name')}`")
            for k, v in m["metrics"].items():
                st.write(f"{k}: **{v:.3f}**")
        else:
            st.warning(f"API is up but the model isn't loaded ({info.status_code}).")
    except requests.RequestException:
        st.error(f"Can't reach the API at {BACKEND_URL}")
    st.divider()
    st.markdown(f"- [API docs (Swagger)]({PUBLIC_API_URL}/docs)\n- [MLflow UI]({MLFLOW_UI_URL})")
    st.caption(f"backend: {BACKEND_URL}")

# --- Prediction form ---------------------------------------------------------------------------
st.subheader("Measurements (cm)")
col1, col2 = st.columns(2)
sepal_length = col1.slider("Sepal length", 0.1, 10.0, 5.1, 0.1)
sepal_width = col2.slider("Sepal width", 0.1, 10.0, 3.5, 0.1)
petal_length = col1.slider("Petal length", 0.1, 10.0, 1.4, 0.1)
petal_width = col2.slider("Petal width", 0.1, 10.0, 0.2, 0.1)

if st.button("Predict", type="primary"):
    payload = {"sepal_length": sepal_length, "sepal_width": sepal_width,
               "petal_length": petal_length, "petal_width": petal_width}
    try:
        r = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=10)
    except requests.RequestException as exc:
        st.error(f"Request to backend failed: {exc}")
    else:
        if r.status_code == 200:
            result = r.json()
            st.success(f"Predicted species: **{result['species']}** (model v{result['model_version']})")
            probs = pd.Series(result["probabilities"], name="probability")
            st.bar_chart(probs)
        elif r.status_code == 422:
            st.warning("Invalid input (422): " + "; ".join(e["msg"] for e in r.json()["detail"]))
        elif r.status_code in (400, 503):
            st.warning(f"{r.status_code}: {r.json().get('detail')}")
        else:
            st.error(f"Server error ({r.status_code}). Please try again later.")

# --- The original connectivity check -----------------------------------------------------------
with st.expander("Connectivity check"):
    if st.button("Call backend"):
        try:
            response = requests.get(f"{BACKEND_URL}/message", timeout=5)
            response.raise_for_status()
            st.success(response.json()["message"])
        except requests.RequestException as exc:
            st.error(f"Request to backend failed: {exc}")
