"""Tests for cleaning, splitting and leakage-free encoding."""

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import (
    RAW_FEATURE_COLS,
    SchemaError,
    clean_dataframe,
    fit_preprocessors,
    get_feature_names,
    get_feature_origins,
    prepare_inference_features,
    split_data,
    transform_features,
)


def _row(**overrides):
    base = {
        "customerID": "0001-A", "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 5, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
        "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 50.0, "TotalCharges": "250.0", "Churn": "No",
    }
    base.update(overrides)
    return base


def _frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rows.append(_row(
            customerID=f"C{i:04d}", tenure=int(rng.integers(0, 72)),
            Contract=rng.choice(["Month-to-month", "One year", "Two year"]),
            InternetService=rng.choice(["DSL", "Fiber optic", "No"]),
            MonthlyCharges=float(rng.uniform(20, 110)), TotalCharges=str(float(rng.uniform(0, 5000))),
            Churn="Yes" if i % 4 == 0 else "No",
        ))
    return pd.DataFrame(rows)


def test_blank_total_charges_with_zero_tenure_becomes_zero():
    df = pd.DataFrame([_row(tenure=0, TotalCharges=" ")])
    clean, report = clean_dataframe(df, require_target=True)
    assert clean["TotalCharges"].iloc[0] == 0.0
    assert report.total_charges_blank_filled == 1
    assert clean["Churn"].iloc[0] == 0


def test_blank_total_charges_with_positive_tenure_is_imputed_with_warning():
    df = pd.DataFrame([_row(tenure=3, MonthlyCharges=40.0, TotalCharges="")])
    clean, report = clean_dataframe(df, require_target=False)
    assert clean["TotalCharges"].iloc[0] == pytest.approx(120.0)
    assert report.total_charges_imputed == 1 and report.warnings


def test_missing_columns_are_all_reported():
    df = pd.DataFrame([_row()]).drop(columns=["tenure", "Contract"])
    with pytest.raises(SchemaError) as excinfo:
        clean_dataframe(df, require_target=False)
    assert set(excinfo.value.missing) == {"tenure", "Contract"}


def test_invalid_numeric_rows_are_dropped_and_reported():
    df = pd.DataFrame([_row(), _row(customerID="BAD", tenure="abc")])
    clean, report = clean_dataframe(df, require_target=False)
    assert len(clean) == 1
    assert report.dropped_rows[0]["customer_id"] == "BAD"


def test_inference_path_drops_label_and_generates_ids():
    df = pd.DataFrame([_row()]).drop(columns=["customerID"])
    clean, report = clean_dataframe(df, require_target=False)
    assert "Churn" not in clean.columns
    assert clean["customerID"].iloc[0] == "ROW-0001" and report.generated_ids == 1


def test_split_is_disjoint_stratified_and_60_20_20():
    clean, _ = clean_dataframe(_frame(200), require_target=True)
    train, val, test = split_data(clean)
    assert (len(train), len(val), len(test)) == (120, 40, 40)
    assert not (set(train.customerID) & set(val.customerID) | set(train.customerID) & set(test.customerID))
    for part in (train, val, test):
        assert part["Churn"].mean() == pytest.approx(0.25, abs=0.03)


def test_scaler_is_fitted_on_train_only():
    clean, _ = clean_dataframe(_frame(200), require_target=True)
    train, val, _ = split_data(clean)
    scaler, encoder = fit_preprocessors(train)
    X_train = transform_features(train, scaler, encoder)
    np.testing.assert_allclose(X_train[:, :3].mean(axis=0), 0, atol=1e-6)
    np.testing.assert_allclose(scaler.mean_, train[["tenure", "MonthlyCharges", "TotalCharges"]].mean(), rtol=1e-9)
    assert len(get_feature_names(encoder)) == len(get_feature_origins(encoder)) == X_train.shape[1]


def test_unseen_category_is_encoded_as_zeros_with_warning():
    clean, _ = clean_dataframe(_frame(100), require_target=True)
    scaler, encoder = fit_preprocessors(clean)
    upload = pd.DataFrame([_row(Contract="Three year")])[RAW_FEATURE_COLS]
    X, clean_upload, report = prepare_inference_features(upload, scaler, encoder)
    names = get_feature_names(encoder)
    contract_cols = [i for i, n in enumerate(names) if n.startswith("Contract_")]
    assert X[0, contract_cols].sum() == 0
    assert report.unknown_categories == {"Contract": ["Three year"]}
