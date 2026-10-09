"""
FastAPI backend for the Telecom Churn Early-Warning Dashboard (Application Lead: Praveen).

Run from the project root::

    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

then open http://localhost:8000 (dashboard) or http://localhost:8000/docs (Swagger UI).

Endpoints
---------
=======  ======================  ==========================================================
GET      /                       HTML dashboard (Jinja2 template + Bootstrap + vanilla JS)
GET      /api/health             liveness + whether the model artefacts loaded
GET      /api/config             τ*, costs, risk tiers, strategy, test metrics (for the UI)
POST     /api/predict            multipart CSV upload → ranked risk table (logged)
POST     /api/predict/single     one customer as JSON → probability + top drivers (logged)
POST     /api/explain            one customer record → top-5 SHAP drivers (not logged:
                                 it re-explains a prediction that was already logged)
GET      /api/logs               most recent prediction-log records + totals
DELETE   /api/logs               clear the prediction log (disable with CHURN_ALLOW_LOG_CLEAR=0)
GET      /api/sample-csv         download ``data/sample_demo.csv`` for the demo
=======  ======================  ==========================================================

Design notes
------------
* All ML logic lives in ``src/`` (``ChurnPredictor``); this file only handles
  HTTP concerns: validation, error codes, JSON shaping and logging.
* CPU-heavy work (pandas parsing, PyTorch inference, SHAP) runs in a worker
  thread via ``run_in_threadpool`` so the async event loop stays responsive.
* If the model artefacts are missing the server still starts; prediction
  endpoints return **503** with instructions, and the UI shows a banner.
"""

from __future__ import annotations

import io
import logging
import math
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:          # allow `import src` however uvicorn is launched
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.data_preprocessing import ID_COL, RAW_FEATURE_COLS, SchemaError  # noqa: E402
from src.inference import ChurnPredictor, ModelNotReadyError, summarize_services  # noqa: E402

from app.logger import PredictionLogger, PredictionLoggingError, log_clear_enabled  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("churn-api")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024     # 10 MB ≈ 100k customers — generous for a demo
MAX_ROWS = 50_000

STATE: dict[str, Any] = {"predictor": None, "error": None, "logger": None}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load artefacts once at start-up (not per request) and open the prediction logger."""
    STATE["logger"] = PredictionLogger(config.LOGS_DIR, backend=os.environ.get("CHURN_LOG_BACKEND", "both"))
    try:
        STATE["predictor"] = ChurnPredictor(config.MODELS_DIR)
        STATE["error"] = None
        logger.info("Model loaded: %s (τ* = %.2f)", STATE["predictor"].model_version,
                    STATE["predictor"].default_threshold)
    except ModelNotReadyError as exc:
        STATE["predictor"] = None
        STATE["error"] = str(exc)
        logger.warning("Model not ready: %s", exc)
    yield


app = FastAPI(
    title="Telecom Churn Early-Warning API",
    version="1.0.0",
    description="Ranks customers by calibrated churn risk, explains each score with SHAP and "
                "recommends who should receive a retention offer based on business cost.",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
YesNo = Literal["Yes", "No"]
InternetAddOn = Literal["Yes", "No", "No internet service"]


class CustomerRecord(BaseModel):
    """One customer in the raw Telco format (used by ``POST /api/predict/single``)."""

    customerID: Optional[str] = Field(None, max_length=64)
    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(..., ge=0, le=1, description="1 = senior citizen, 0 = not")
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(..., ge=0, le=1200, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddOn
    OnlineBackup: InternetAddOn
    DeviceProtection: InternetAddOn
    TechSupport: InternetAddOn
    StreamingTV: InternetAddOn
    StreamingMovies: InternetAddOn
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)",
                           "Credit card (automatic)"]
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: Optional[float] = Field(None, ge=0, description="Blank for brand-new customers")

    model_config = ConfigDict(json_schema_extra={"examples": [{
        "customerID": "DEMO-0001", "gender": "Female", "SeniorCitizen": 0, "Partner": "No",
        "Dependents": "No", "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.1, "TotalCharges": 178.2,
    }]})


class ExplainRequest(BaseModel):
    """A customer record as returned in ``customers[i].record`` by ``/api/predict``."""

    record: dict[str, Any]
    top_k: int = Field(5, ge=1, le=len(RAW_FEATURE_COLS))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_predictor() -> ChurnPredictor:
    """Return the loaded predictor or raise 503 with the reason it is unavailable."""
    predictor = STATE.get("predictor")
    if predictor is None:
        raise HTTPException(status_code=503, detail=STATE.get("error") or "Model is not loaded yet.")
    return predictor


def _jsonable(value: Any) -> Any:
    """Convert NumPy/pandas scalars to plain Python; NaN/inf → None (JSON has no NaN)."""
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if value is None or (isinstance(value, str)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _validate_threshold(threshold: Optional[float]) -> Optional[float]:
    if threshold is None:
        return None
    if not (0.0 <= threshold <= 1.0) or not math.isfinite(threshold):
        raise HTTPException(status_code=422, detail="threshold must be between 0 and 1")
    return float(threshold)


def _log_predictions(rows: list[dict[str, Any]], threshold: float, batch_id: str, source: str,
                     model_version: str) -> Optional[str]:
    """Write one log record per prediction; return a warning string if logging failed."""
    plog: PredictionLogger = STATE["logger"]
    records = [plog.make_record(r["customer_id"], r["churn_probability"], r["risk_tier"], threshold,
                                batch_id=batch_id, source=source, model_version=model_version)
               for r in rows]
    try:
        plog.log(records)
        return None
    except PredictionLoggingError as exc:
        logger.error("Prediction logging failed: %s", exc)
        return f"Predictions were returned but could not be logged: {exc}"


def _read_csv_bytes(contents: bytes) -> pd.DataFrame:
    """Decode an uploaded CSV (UTF-8 with/without BOM, falling back to Latin-1) into a DataFrame."""
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            text = contents.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    try:
        # dtype=str keeps IDs like "0012-ABCD" intact; numeric columns are coerced during cleaning.
        return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=True)
    except pd.errors.EmptyDataError as exc:
        raise HTTPException(status_code=400, detail="The CSV file has no rows or no header.") from exc
    except pd.errors.ParserError as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse the CSV: {exc}") from exc


def _score(predictor: ChurnPredictor, df: pd.DataFrame, threshold: Optional[float]) -> dict[str, Any]:
    """Run the full pipeline on a DataFrame and shape the JSON response (runs in a worker thread)."""
    results, _, report = predictor.predict_dataframe(df, threshold)
    tau = predictor.default_threshold if threshold is None else threshold
    customers = []
    for row in results.to_dict(orient="records"):
        record = {col: _jsonable(row[col]) for col in [ID_COL] + RAW_FEATURE_COLS}
        customers.append({
            "rank": int(row["rank"]),
            "customer_id": str(row[ID_COL]),
            "churn_probability": round(float(row["churn_probability"]), 6),
            "risk_tier": row["risk_tier"],
            "flagged_for_offer": bool(row["flagged_for_offer"]),
            "key_services": {k: _jsonable(v) if not isinstance(v, list) else v
                             for k, v in summarize_services(row).items()},
            "record": record,
        })
    probs = results["churn_probability"].to_numpy() if len(results) else np.zeros(0)
    tiers = results["risk_tier"].value_counts().to_dict() if len(results) else {}
    return {
        "threshold_used": tau,
        "n_customers": len(customers),
        "customers": customers,
        "summary": {
            "by_tier": {t: int(tiers.get(t, 0)) for t in ("High", "Medium", "Low")},
            "n_flagged": int((probs >= tau).sum()),
            "mean_probability": float(probs.mean()) if probs.size else 0.0,
            "expected_churners": float(probs.sum()),
        },
        "cleaning": report.to_dict(),
        "warnings": list(report.warnings),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard(request: Request) -> HTMLResponse:
    """Serve the single-page dashboard."""
    return templates.TemplateResponse(request, "index.html", {"app_version": app.version})


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> FileResponse:
    """Legacy favicon path that browsers request automatically (the page itself links the SVG icon)."""
    return FileResponse(APP_DIR / "static" / "img" / "favicon.ico", media_type="image/x-icon")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    """Liveness probe; ``model_loaded`` is False until the notebooks have produced the artefacts."""
    predictor = STATE.get("predictor")
    return {
        "status": "ok" if predictor else "model_not_ready",
        "model_loaded": predictor is not None,
        "model_version": predictor.model_version if predictor else None,
        "detail": STATE.get("error"),
        "log_backends": STATE["logger"].active_backends if STATE.get("logger") else [],
    }


@app.get("/api/config")
async def get_config() -> dict[str, Any]:
    """Everything the UI needs to initialise: τ*, costs, tiers, strategy, test metrics."""
    return get_predictor().metadata()


@app.post("/api/predict")
async def predict_csv(
    file: UploadFile = File(..., description="CSV with the 19 Telco feature columns (+ optional customerID)"),
    threshold: Optional[float] = Form(None, description="Decision threshold τ; default = validation-optimal τ*"),
) -> dict[str, Any]:
    """
    Score an uploaded CSV and return customers ranked by churn probability (highest first).

    Error codes: 400 bad/empty file · 413 too large · 422 missing columns or bad threshold ·
    503 model not trained yet.
    """
    predictor = get_predictor()
    threshold = _validate_threshold(threshold)
    filename = file.filename or "upload.csv"
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    if not contents.strip():
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    df = await run_in_threadpool(_read_csv_bytes, contents)
    if len(df) == 0:
        raise HTTPException(status_code=400, detail="The CSV has a header but no customer rows.")
    if len(df) > MAX_ROWS:
        raise HTTPException(status_code=413, detail=f"At most {MAX_ROWS:,} rows per upload.")

    try:
        payload = await run_in_threadpool(_score, predictor, df, threshold)
    except SchemaError as exc:
        raise HTTPException(status_code=422, detail={"message": str(exc), "missing_columns": exc.missing,
                                                     "required_columns": RAW_FEATURE_COLS}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    batch_id = uuid.uuid4().hex
    warning = _log_predictions(payload["customers"], payload["threshold_used"], batch_id,
                               "csv_upload", predictor.model_version)
    if warning:
        payload["warnings"].append(warning)
    payload.update({"batch_id": batch_id, "filename": filename, "model_version": predictor.model_version})
    return payload


@app.post("/api/predict/single")
async def predict_single(customer: CustomerRecord,
                         threshold: Optional[float] = Query(None, ge=0.0, le=1.0)) -> dict[str, Any]:
    """Score one customer (JSON body) and return probability, tier, decision and top-5 drivers."""
    predictor = get_predictor()
    record = customer.model_dump()
    tau = predictor.default_threshold if threshold is None else float(threshold)
    try:
        result = await run_in_threadpool(predictor.explain_record, record, 5)
    except (ValueError, ModelNotReadyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result["flagged_for_offer"] = result["churn_probability"] >= tau
    result["threshold_used"] = tau
    batch_id = uuid.uuid4().hex
    warning = _log_predictions([result], tau, batch_id, "api_single", predictor.model_version)
    result.update({"batch_id": batch_id, "model_version": predictor.model_version})
    if warning:
        result["warnings"] = [*result.get("warnings", []), warning]
    return result


@app.post("/api/explain")
async def explain(request: ExplainRequest) -> dict[str, Any]:
    """Top-k SHAP drivers for one customer record (already scored and logged by /api/predict)."""
    predictor = get_predictor()
    try:
        return await run_in_threadpool(predictor.explain_record, request.record, request.top_k)
    except SchemaError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/logs")
async def get_logs(limit: int = Query(100, ge=1, le=1000)) -> dict[str, Any]:
    """Most recent prediction-log records (newest first) plus totals."""
    plog: PredictionLogger = STATE["logger"]
    records = await run_in_threadpool(plog.recent, limit)
    stats = await run_in_threadpool(plog.stats)
    return {"records": records, "stats": stats}


@app.delete("/api/logs")
async def clear_logs() -> dict[str, Any]:
    """
    Permanently delete all logged predictions (JSONL + SQLite) — the "Clear log" button.

    Returns 403 when the deployment has disabled it with ``CHURN_ALLOW_LOG_CLEAR=0``.
    """
    if not log_clear_enabled():
        raise HTTPException(status_code=403, detail="Clearing the prediction log is disabled on this deployment.")
    plog: PredictionLogger = STATE["logger"]
    try:
        removed = await run_in_threadpool(plog.clear)
    except PredictionLoggingError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    logger.info("Prediction log cleared: %d record(s) removed", removed)
    return {"removed": removed}


@app.get("/api/sample-csv", include_in_schema=True)
async def sample_csv() -> FileResponse:
    """Download the 50-customer unlabelled demo file produced by notebook 02."""
    if not config.SAMPLE_DEMO_PATH.exists():
        raise HTTPException(status_code=404, detail="data/sample_demo.csv not found — run notebook 02.")
    return FileResponse(config.SAMPLE_DEMO_PATH, media_type="text/csv", filename="sample_demo.csv")
