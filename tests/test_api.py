"""End-to-end tests of the FastAPI backend (skipped until the notebooks have produced the models)."""

import io
import json

import pandas as pd
import pytest

from src import config

ARTEFACTS = [config.BEST_MODEL_PATH, config.MODEL_CONFIG_PATH, config.SCALER_PATH, config.ENCODER_PATH,
             config.FEATURE_COLUMNS_PATH, config.SHAP_BACKGROUND_PATH, config.SAMPLE_DEMO_PATH]
pytestmark = pytest.mark.skipif(not all(p.exists() for p in ARTEFACTS),
                                reason="model artefacts missing — run notebooks 01–07 first")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:      # `with` runs the lifespan (model loading)
        yield test_client


def _log_lines():
    path = config.LOGS_DIR / "predictions.jsonl"
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def test_health_and_config(client):
    health = client.get("/api/health").json()
    assert health["model_loaded"] is True
    cfg = client.get("/api/config").json()
    assert 0 < cfg["threshold"]["optimal"] < 1
    assert cfg["costs"]["c_lost"] > cfg["costs"]["c_offer"]


def test_dashboard_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200 and "Churn Early-Warning Dashboard" in response.text


def test_icons_and_static_assets_are_served(client):
    page = client.get("/").text
    for asset in ("/static/css/styles.css", "/static/js/app.js", "/static/img/favicon.svg",
                  "/static/img/apple-touch-icon.png"):
        assert asset in page
        assert client.get(asset).status_code == 200, asset
    favicon = client.get("/favicon.ico")
    assert favicon.status_code == 200 and favicon.headers["content-type"] == "image/x-icon"
    assert 'id="i-high"' in page and 'id="i-medium"' in page and 'id="i-low"' in page   # tier shape icons


def test_predict_sample_csv_is_ranked_and_logged(client):
    before = len(_log_lines())
    with open(config.SAMPLE_DEMO_PATH, "rb") as fh:
        response = client.post("/api/predict", files={"file": ("sample_demo.csv", fh, "text/csv")},
                               data={"threshold": "0.25"})
    assert response.status_code == 200, response.text
    body = response.json()
    probs = [c["churn_probability"] for c in body["customers"]]
    assert body["n_customers"] == 50 and probs == sorted(probs, reverse=True)
    assert {c["risk_tier"] for c in body["customers"]} <= {"High", "Medium", "Low"}
    assert all(c["flagged_for_offer"] == (c["churn_probability"] >= 0.25) for c in body["customers"])

    lines = _log_lines()
    assert len(lines) == before + 50
    record = json.loads(lines[-1])
    assert {"timestamp", "customer_id", "churn_probability", "risk_tier", "threshold_used"} <= record.keys()
    assert record["threshold_used"] == 0.25 and record["batch_id"] == body["batch_id"]


def test_label_column_is_ignored(client):
    raw = pd.read_csv(config.RAW_DATA_PATH).head(5)
    csv_bytes = raw.to_csv(index=False).encode()
    body = client.post("/api/predict", files={"file": ("with_label.csv", csv_bytes, "text/csv")}).json()
    assert body["n_customers"] == 5
    assert all("Churn" not in c["record"] for c in body["customers"])


@pytest.mark.parametrize("filename,content,status", [
    ("data.txt", b"a,b\n1,2\n", 400),
    ("empty.csv", b"", 400),
    ("missing.csv", b"customerID,tenure\nX,1\n", 422),
])
def test_bad_uploads(client, filename, content, status):
    response = client.post("/api/predict", files={"file": (filename, io.BytesIO(content), "text/csv")})
    assert response.status_code == status
    if status == 422:
        assert "Contract" in response.json()["detail"]["missing_columns"]


def test_threshold_out_of_range(client):
    with open(config.SAMPLE_DEMO_PATH, "rb") as fh:
        response = client.post("/api/predict", files={"file": ("s.csv", fh, "text/csv")}, data={"threshold": "1.5"})
    assert response.status_code == 422


def test_explain_returns_top_five_drivers(client):
    with open(config.SAMPLE_DEMO_PATH, "rb") as fh:
        customer = client.post("/api/predict", files={"file": ("s.csv", fh, "text/csv")}).json()["customers"][0]
    body = client.post("/api/explain", json={"record": customer["record"], "top_k": 5}).json()
    assert len(body["drivers"]) == 5
    assert body["churn_probability"] == pytest.approx(customer["churn_probability"], abs=1e-6)
    impacts = [abs(d["shap_value"]) for d in body["drivers"]]
    assert impacts == sorted(impacts, reverse=True)


def test_single_prediction_endpoint_is_logged(client):
    example = {
        "customerID": "TEST-SINGLE", "gender": "Male", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No",
        "tenure": 1, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 75.0, "TotalCharges": None,
    }
    body = client.post("/api/predict/single", json=example).json()
    assert body["risk_tier"] in {"High", "Medium"} and len(body["drivers"]) == 5
    assert json.loads(_log_lines()[-1])["customer_id"] == "TEST-SINGLE"
    invalid = client.post("/api/predict/single", json={**example, "Contract": "Forever"})
    assert invalid.status_code == 422


def test_logs_endpoint(client):
    body = client.get("/api/logs?limit=3").json()
    assert len(body["records"]) == 3 and body["stats"]["total_predictions"] >= 50


def test_clear_logs_endpoint(client, monkeypatch):
    assert len(_log_lines()) > 0
    monkeypatch.setenv("CHURN_ALLOW_LOG_CLEAR", "0")          # a public deployment can forbid it
    assert client.delete("/api/logs").status_code == 403
    assert len(_log_lines()) > 0

    monkeypatch.setenv("CHURN_ALLOW_LOG_CLEAR", "1")
    before = len(_log_lines())
    response = client.delete("/api/logs")
    assert response.status_code == 200 and response.json()["removed"] == before
    assert _log_lines() == []
    body = client.get("/api/logs").json()
    assert body["records"] == [] and body["stats"]["total_predictions"] == 0

    # Logging keeps working after a clear (JSONL and the SQLite table still exist).
    with open(config.SAMPLE_DEMO_PATH, "rb") as fh:
        client.post("/api/predict", files={"file": ("s.csv", fh, "text/csv")})
    assert len(_log_lines()) == 50
    import sqlite3
    with sqlite3.connect(config.LOGS_DIR / "predictions_log.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 50
