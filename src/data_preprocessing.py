"""
Data loading, cleaning, stratified splitting and leakage-free feature encoding.

Pipeline overview (Part 2 of the brief)
---------------------------------------
1. ``load_raw_data``     – read the CSV and check that every expected column exists.
2. ``clean_dataframe``   – strip whitespace, fix the blank ``TotalCharges`` strings,
                            normalise ``SeniorCitizen``, encode ``Churn`` → {0, 1}.
3. ``split_data``        – stratified 60 / 20 / 20 train / validation / test split.
4. ``fit_preprocessors`` – fit ``StandardScaler`` + ``OneHotEncoder`` on **train only**.
5. ``transform_features``– apply the *already fitted* transformers to any split
                            (or to a CSV uploaded to the web app).

Data-leakage rule (viva critical)
---------------------------------
The scaler learns a mean μ and standard deviation σ for each numeric column and
the encoder learns the list of categories. If those statistics were computed on
validation/test rows, information about the "future" would leak into training
and our reported metrics would be optimistic. Therefore ``fit_preprocessors``
receives the training frame only, and every other split is passed through
``transform_features`` which calls ``.transform()`` — never ``.fit()``.

Standardisation maps each numeric feature x to

    z = (x − μ_train) / σ_train

so all numeric inputs have mean 0 and variance 1 *on the training set*. This
keeps the scale of ``TotalCharges`` (up to ~8,700) from dominating the gradient
updates relative to 0/1 one-hot features.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


# ---------------------------------------------------------------------------
# Column schema of the IBM / Kaggle Telco Customer Churn dataset
# ---------------------------------------------------------------------------
ID_COL: str = "customerID"
TARGET_COL: str = "Churn"

#: Continuous columns → standardised with StandardScaler.
NUMERIC_COLS: list[str] = ["tenure", "MonthlyCharges", "TotalCharges"]

#: Discrete columns → one-hot encoded. ``SeniorCitizen`` is stored as 0/1 in the
#: raw file but is conceptually a Yes/No category, so it is mapped to "No"/"Yes".
CATEGORICAL_COLS: list[str] = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

#: The 19 raw model inputs, in the order they appear in the source CSV.
RAW_FEATURE_COLS: list[str] = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]

_YES_NO_MAP: dict[Any, str] = {
    0: "No", 1: "Yes", "0": "No", "1": "Yes",
    "no": "No", "yes": "Yes", "false": "No", "true": "Yes",
    False: "No", True: "Yes",
}
_TARGET_MAP: dict[Any, int] = {
    "yes": 1, "no": 0, "1": 1, "0": 0, 1: 1, 0: 0, True: 1, False: 0,
    "true": 1, "false": 0,
}
UNKNOWN_CATEGORY: str = "Unknown"


class SchemaError(ValueError):
    """Raised when an input table is missing columns the model needs."""

    def __init__(self, missing: list[str], context: str = "input data") -> None:
        self.missing = list(missing)
        super().__init__(
            f"{context} is missing {len(missing)} required column(s): {', '.join(missing)}"
        )


@dataclass
class CleaningReport:
    """
    Human-readable audit trail of everything ``clean_dataframe`` changed.

    The web app returns this to the UI so a business user can see, for example,
    that three uploaded rows were skipped because ``tenure`` was not a number.
    """

    n_input_rows: int = 0
    n_output_rows: int = 0
    total_charges_blank_filled: int = 0      # blank TotalCharges with tenure == 0 → 0.0
    total_charges_imputed: int = 0           # blank TotalCharges with tenure > 0 → tenure × MonthlyCharges
    generated_ids: int = 0                   # rows without a customerID
    dropped_rows: list[dict[str, Any]] = field(default_factory=list)
    missing_categorical: dict[str, int] = field(default_factory=dict)
    unknown_categories: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return asdict(self)


@dataclass
class DataSplits:
    """
    Container for the processed arrays of each split.

    ``X_test``/``y_test`` stay ``None`` unless the caller explicitly asks for
    them (``load_splits(include_test=True)``) — a guard-rail that makes it hard
    to peek at the test set accidentally while developing (notebooks 03–06).
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    ids_train: np.ndarray
    ids_val: np.ndarray
    feature_names: list[str]
    X_test: np.ndarray | None = None
    y_test: np.ndarray | None = None
    ids_test: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Loading & validation
# ---------------------------------------------------------------------------
def load_raw_data(path: str | Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw Telco CSV and verify its schema.

    Parameters
    ----------
    path : str or Path
        Location of ``Telco-Customer-Churn.csv``.

    Returns
    -------
    pd.DataFrame
        The raw, *uncleaned* table (7,043 rows × 21 columns for the full dataset).

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist (the message explains how to download it).
    SchemaError
        If any feature column, the ID column or the target column is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download it with:\n"
            f"  wget -O {path} '{config.DATASET_URL}'"
        )
    df = pd.read_csv(path)
    validate_columns(df, require_target=True, require_id=True, context=str(path.name))
    return df


def validate_columns(
    df: pd.DataFrame,
    *,
    require_target: bool = False,
    require_id: bool = False,
    context: str = "input data",
) -> None:
    """
    Check that ``df`` contains every raw feature column (and optionally ID/target).

    Raises
    ------
    SchemaError
        Listing every missing column, so the user can fix the file in one go.
    """
    required = list(RAW_FEATURE_COLS)
    if require_id:
        required.append(ID_COL)
    if require_target:
        required.append(TARGET_COL)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise SchemaError(missing, context)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def _strip_strings(series: pd.Series) -> pd.Series:
    """Strip surrounding whitespace from string cells, leaving non-strings untouched."""
    return series.map(lambda v: v.strip() if isinstance(v, str) else v)


def _to_numeric(series: pd.Series) -> pd.Series:
    """
    Convert a column to float, turning blanks/garbage into NaN.

    ``" "`` (a single space, as found in ``TotalCharges``) becomes ``""`` after
    stripping, which ``pd.to_numeric(errors="coerce")`` maps to NaN.
    """
    return pd.to_numeric(_strip_strings(series).replace("", np.nan), errors="coerce")


def _normalise_yes_no(value: Any) -> Any:
    """Map 0/1, True/False and case variants of yes/no to canonical "No"/"Yes"."""
    if isinstance(value, float) and np.isnan(value):
        return value
    key = value.strip().lower() if isinstance(value, str) else value
    if isinstance(key, float) and key.is_integer():
        key = int(key)
    return _YES_NO_MAP.get(key, value)


def encode_target(series: pd.Series) -> pd.Series:
    """
    Encode the ``Churn`` column as integers: Yes → 1, No → 0.

    Raises
    ------
    ValueError
        If any value cannot be interpreted as yes/no (e.g. a typo such as "Yse").
    """
    def _map(v: Any) -> Any:
        key = v.strip().lower() if isinstance(v, str) else v
        return _TARGET_MAP.get(key, np.nan)

    encoded = series.map(_map)
    if encoded.isna().any():
        bad = sorted({str(v) for v in series[encoded.isna()]})
        raise ValueError(f"Unrecognised {TARGET_COL} values: {bad}")
    return encoded.astype(int)


def clean_dataframe(
    df: pd.DataFrame,
    *,
    require_target: bool,
    known_categories: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Clean a raw Telco table so it can be encoded.

    Cleaning rules
    --------------
    * **Whitespace** is stripped from all text cells.
    * **TotalCharges** is converted to float. The raw file contains 11 blank
      strings (``" "``). Every one of those rows has ``tenure == 0``: they are
      brand-new customers who have not yet received a first bill, so the
      amount they have been charged so far is genuinely **0.0**. (For uploaded
      data where a blank appears with ``tenure > 0`` we impute
      ``tenure × MonthlyCharges`` — the best available estimate — and warn.)
    * **SeniorCitizen** 0/1 → "No"/"Yes" so it is one-hot encoded like the
      other binary flags.
    * **Missing categorical values** become "Unknown", which the fitted encoder
      maps to an all-zero block (``handle_unknown="ignore"``).
    * Rows whose ``tenure`` or ``MonthlyCharges`` is missing/negative cannot be
      scored and are **dropped** and listed in the report (never silently).
    * **Churn** (if present) → 1/0. **customerID** is kept for identification
      but never used as a feature; rows lacking one get ``ROW-0001``-style IDs.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data (training CSV or a user upload). It is not modified in place.
    require_target : bool
        ``True`` for training data (Churn must exist and be valid),
        ``False`` for inference data (any Churn column is dropped).
    known_categories : dict, optional
        Categories seen by the fitted encoder, ``{column: [levels...]}``.
        When given, values outside this vocabulary are reported as warnings.

    Returns
    -------
    (pd.DataFrame, CleaningReport)
        The cleaned frame — columns ``[customerID] + RAW_FEATURE_COLS (+ Churn)``,
        original index preserved so dropped rows can be traced back — and the report.
    """
    validate_columns(df, require_target=require_target)
    report = CleaningReport(n_input_rows=int(len(df)))
    out = df.copy()

    # --- identifier ---------------------------------------------------------
    if ID_COL in out.columns:
        ids = _strip_strings(out[ID_COL]).astype(object)
        missing_id = ids.isna() | (ids.astype(str).str.len() == 0)
    else:
        ids = pd.Series([np.nan] * len(out), index=out.index, dtype=object)
        missing_id = pd.Series(True, index=out.index)
    if missing_id.any():
        positions = np.flatnonzero(missing_id.to_numpy())
        ids.loc[missing_id] = [f"ROW-{p + 1:04d}" for p in positions]
        report.generated_ids = int(missing_id.sum())
        report.warnings.append(
            f"{report.generated_ids} row(s) had no {ID_COL}; generated IDs like 'ROW-0001'."
        )
    out[ID_COL] = ids.astype(str)

    # --- numeric columns ----------------------------------------------------
    for col in NUMERIC_COLS:
        out[col] = _to_numeric(out[col])

    # Rows that cannot be scored: tenure / MonthlyCharges missing or negative.
    invalid_mask = pd.Series(False, index=out.index)
    for col in ("tenure", "MonthlyCharges"):
        bad = out[col].isna() | (out[col] < 0)
        for idx in out.index[bad & ~invalid_mask]:
            report.dropped_rows.append({
                "row": int(out.index.get_loc(idx)) + 1,
                "customer_id": str(out.at[idx, ID_COL]),
                "reason": f"'{col}' is missing, non-numeric or negative",
            })
        invalid_mask |= bad
    negative_total = out["TotalCharges"] < 0
    for idx in out.index[negative_total & ~invalid_mask]:
        report.dropped_rows.append({
            "row": int(out.index.get_loc(idx)) + 1,
            "customer_id": str(out.at[idx, ID_COL]),
            "reason": "'TotalCharges' is negative",
        })
    invalid_mask |= negative_total
    out = out.loc[~invalid_mask].copy()

    # TotalCharges blanks: tenure == 0 → 0.0 (no bill yet); tenure > 0 → estimate.
    blank_total = out["TotalCharges"].isna()
    new_customer = blank_total & (out["tenure"] == 0)
    out.loc[new_customer, "TotalCharges"] = 0.0
    report.total_charges_blank_filled = int(new_customer.sum())
    needs_estimate = blank_total & (out["tenure"] > 0)
    if needs_estimate.any():
        out.loc[needs_estimate, "TotalCharges"] = (
            out.loc[needs_estimate, "tenure"] * out.loc[needs_estimate, "MonthlyCharges"]
        )
        report.total_charges_imputed = int(needs_estimate.sum())
        report.warnings.append(
            f"{report.total_charges_imputed} row(s) had blank TotalCharges with tenure > 0; "
            "estimated as tenure × MonthlyCharges."
        )

    # --- categorical columns ------------------------------------------------
    for col in CATEGORICAL_COLS:
        values = _strip_strings(out[col])
        if col == "SeniorCitizen":
            values = values.map(_normalise_yes_no)
        missing = values.isna() | (values.astype(str).str.len() == 0)
        if missing.any():
            report.missing_categorical[col] = int(missing.sum())
            values = values.where(~missing, UNKNOWN_CATEGORY)
        out[col] = values.astype(str)
    if report.missing_categorical:
        report.warnings.append(
            "Missing categorical values were set to 'Unknown' (encoded as all zeros): "
            + ", ".join(f"{c} ({n})" for c, n in report.missing_categorical.items())
        )

    if known_categories:
        for col, levels in known_categories.items():
            if col not in out.columns:
                continue
            unseen = sorted(set(out[col].unique()) - set(levels) - {UNKNOWN_CATEGORY})
            if unseen:
                report.unknown_categories[col] = unseen
        if report.unknown_categories:
            report.warnings.append(
                "Values never seen during training (encoded as all zeros): "
                + "; ".join(f"{c}: {v}" for c, v in report.unknown_categories.items())
            )

    # --- target -------------------------------------------------------------
    columns = [ID_COL] + RAW_FEATURE_COLS
    if require_target:
        out[TARGET_COL] = encode_target(out[TARGET_COL])
        columns.append(TARGET_COL)
    elif TARGET_COL in out.columns:
        # Never let a ground-truth label reach the inference path.
        out = out.drop(columns=[TARGET_COL])

    if report.dropped_rows:
        report.warnings.append(f"{len(report.dropped_rows)} row(s) could not be scored and were skipped.")
    report.n_output_rows = int(len(out))
    return out[columns], report


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def split_data(
    df: pd.DataFrame,
    *,
    val_size: float = config.VAL_SIZE,
    test_size: float = config.TEST_SIZE,
    seed: int = config.SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train / validation / test split (default 60 / 20 / 20).

    Two calls to ``train_test_split`` are used:

    1. full → (train+val 80 %, test 20 %)
    2. train+val → (train 75 %, val 25 %), and 0.75 × 0.80 = 0.60, 0.25 × 0.80 = 0.20.

    ``stratify=y`` keeps the churn rate (≈26.5 %) identical in all three
    splits, so validation and test metrics are measured on the same class
    balance the model was trained on.

    Returns
    -------
    (train_df, val_df, test_df)
        Disjoint frames (checked by :func:`assert_disjoint_splits`).
    """
    if TARGET_COL not in df.columns:
        raise SchemaError([TARGET_COL], "split_data input")
    if not (0 < val_size < 1 and 0 < test_size < 1 and val_size + test_size < 1):
        raise ValueError(
            f"val_size ({val_size}) and test_size ({test_size}) must be in (0, 1) and sum to < 1"
        )
    if df[TARGET_COL].value_counts().min() < 3:
        raise ValueError("Each class needs at least 3 rows to stratify into three splits.")

    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET_COL], random_state=seed
    )
    relative_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val, stratify=train_val[TARGET_COL], random_state=seed
    )
    assert_disjoint_splits(train, val, test)
    return train, val, test


def assert_disjoint_splits(*frames: pd.DataFrame) -> None:
    """
    Guarantee no customer appears in more than one split (a leakage check).

    Raises
    ------
    AssertionError
        If any index value or customerID is shared between splits.
    """
    seen_index: set[Any] = set()
    seen_ids: set[str] = set()
    for frame in frames:
        idx = set(frame.index)
        if seen_index & idx:
            raise AssertionError("Row index overlap between splits — data leakage!")
        seen_index |= idx
        if ID_COL in frame.columns:
            ids = set(frame[ID_COL].astype(str))
            if seen_ids & ids:
                raise AssertionError("customerID overlap between splits — data leakage!")
            seen_ids |= ids


# ---------------------------------------------------------------------------
# Encoding & scaling (fit on train only!)
# ---------------------------------------------------------------------------
def make_one_hot_encoder() -> OneHotEncoder:
    """
    Build the categorical encoder in a way that works across scikit-learn versions.

    * ``handle_unknown="ignore"`` – an unseen category at inference time becomes
      an all-zero block instead of crashing the web app.
    * ``drop="if_binary"`` – a Yes/No column produces one 0/1 feature
      (e.g. ``Partner_Yes``) instead of two perfectly collinear ones; columns
      with ≥3 levels keep every level so no information is lost.
    * Dense output (``sparse_output=False``; called ``sparse`` before sklearn 1.2).
    """
    kwargs: dict[str, Any] = {"handle_unknown": "ignore", "drop": "if_binary", "dtype": np.float64}
    try:
        return OneHotEncoder(sparse_output=False, **kwargs)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(sparse=False, **kwargs)


def fit_preprocessors(train_df: pd.DataFrame) -> tuple[StandardScaler, OneHotEncoder]:
    """
    Fit the scaler and encoder on the **training split only**.

    Parameters
    ----------
    train_df : pd.DataFrame
        Cleaned training rows. Passing validation or test rows here would
        leak their statistics into the model — do not.

    Returns
    -------
    (StandardScaler, OneHotEncoder)
        Fitted transformers, later saved to ``models/scaler.pkl`` / ``encoder.pkl``.
    """
    if len(train_df) == 0:
        raise ValueError("Cannot fit preprocessors on an empty training frame.")
    scaler = StandardScaler().fit(train_df[NUMERIC_COLS].to_numpy(dtype=np.float64))
    encoder = make_one_hot_encoder().fit(train_df[CATEGORICAL_COLS].astype(object))
    return scaler, encoder


def transform_features(
    df: pd.DataFrame, scaler: StandardScaler, encoder: OneHotEncoder
) -> np.ndarray:
    """
    Apply the *already-fitted* scaler and encoder (``.transform`` only).

    Output column order = ``NUMERIC_COLS`` (scaled) followed by the one-hot
    columns, exactly matching :func:`get_feature_names`.

    Returns
    -------
    np.ndarray, dtype float32, shape (n_rows, n_features)
        float32 because that is PyTorch's default tensor precision.
    """
    if len(df) == 0:
        return np.zeros((0, len(get_feature_names(encoder))), dtype=np.float32)
    numeric = scaler.transform(df[NUMERIC_COLS].to_numpy(dtype=np.float64))
    with warnings.catch_warnings():
        # Unseen categories are already detected and reported by clean_dataframe(); silence
        # scikit-learn's duplicate console warning so server logs stay readable.
        warnings.filterwarnings("ignore", message="Found unknown categories", category=UserWarning)
        categorical = encoder.transform(df[CATEGORICAL_COLS].astype(object))
    if hasattr(categorical, "toarray"):  # defensive: sparse output on odd configs
        categorical = categorical.toarray()
    return np.hstack([numeric, categorical]).astype(np.float32)


def get_feature_names(encoder: OneHotEncoder) -> list[str]:
    """Ordered names of the encoded features, e.g. ``['tenure', ..., 'Contract_Two year', ...]``."""
    return list(NUMERIC_COLS) + [str(n) for n in encoder.get_feature_names_out(CATEGORICAL_COLS)]


def get_feature_origins(encoder: OneHotEncoder) -> list[str]:
    """
    For each encoded feature, the raw column it came from.

    Used by the explainability module to **sum SHAP values of one-hot columns
    back into their original feature** (SHAP values are additive, so the sum
    is the total contribution of, e.g., ``Contract``).

    Built from ``encoder.categories_`` and ``encoder.drop_idx_`` rather than by
    parsing names, so it is robust to category labels containing underscores.
    """
    origins: list[str] = list(NUMERIC_COLS)
    drop_idx = getattr(encoder, "drop_idx_", None)
    for i, (col, cats) in enumerate(zip(CATEGORICAL_COLS, encoder.categories_)):
        dropped = None if drop_idx is None else drop_idx[i]
        for j, _ in enumerate(cats):
            if dropped is not None and j == dropped:
                continue
            origins.append(col)
    expected = len(get_feature_names(encoder))
    if len(origins) != expected:
        raise RuntimeError(f"Feature-origin map has {len(origins)} entries, expected {expected}")
    return origins


def get_known_categories(encoder: OneHotEncoder) -> dict[str, list[str]]:
    """``{column: [categories seen during training]}`` from a fitted encoder."""
    return {col: [str(c) for c in cats] for col, cats in zip(CATEGORICAL_COLS, encoder.categories_)}


def prepare_inference_features(
    raw_df: pd.DataFrame, scaler: StandardScaler, encoder: OneHotEncoder
) -> tuple[np.ndarray, pd.DataFrame, CleaningReport]:
    """
    Full inference-time preprocessing for an uploaded CSV / API payload.

    Drops any ``Churn`` column, cleans with the same rules as training, reports
    unseen categories, and transforms with the saved (train-fitted) objects.

    Returns
    -------
    (X, clean_df, report)
        ``X`` is float32 ``(n_valid_rows, n_features)``, row-aligned with ``clean_df``.
    """
    clean_df, report = clean_dataframe(
        raw_df, require_target=False, known_categories=get_known_categories(encoder)
    )
    X = transform_features(clean_df, scaler, encoder)
    return X, clean_df, report


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_preprocessors(
    scaler: StandardScaler,
    encoder: OneHotEncoder,
    feature_columns: list[str],
    models_dir: str | Path = config.MODELS_DIR,
) -> None:
    """Persist the fitted scaler, encoder and ordered feature names with joblib."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, models_dir / config.SCALER_PATH.name)
    joblib.dump(encoder, models_dir / config.ENCODER_PATH.name)
    joblib.dump(list(feature_columns), models_dir / config.FEATURE_COLUMNS_PATH.name)


def load_preprocessors(
    models_dir: str | Path = config.MODELS_DIR,
) -> tuple[StandardScaler, OneHotEncoder, list[str]]:
    """
    Load the scaler, encoder and feature list, and check they are mutually consistent.

    Raises
    ------
    FileNotFoundError
        If any artefact is missing (run ``02_preprocessing.ipynb`` first).
    RuntimeError
        If the saved feature list does not match what the encoder produces.
    """
    models_dir = Path(models_dir)
    paths = [models_dir / config.SCALER_PATH.name, models_dir / config.ENCODER_PATH.name,
             models_dir / config.FEATURE_COLUMNS_PATH.name]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing preprocessing artefacts (run notebooks/02_preprocessing.ipynb): " + ", ".join(missing)
        )
    scaler, encoder, feature_columns = (joblib.load(p) for p in paths)
    if list(feature_columns) != get_feature_names(encoder):
        raise RuntimeError("feature_columns.pkl does not match the saved encoder — re-run preprocessing.")
    return scaler, encoder, list(feature_columns)


def save_splits(
    splits: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    feature_names: list[str],
    processed_dir: str | Path = config.PROCESSED_DIR,
) -> Path:
    """
    Save the encoded arrays so every later notebook uses *identical* splits.

    Parameters
    ----------
    splits : dict
        ``{"train": (X, y, ids), "val": (...), "test": (...)}``.
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {"feature_names": np.array(feature_names, dtype=object)}
    for name, (X, y, ids) in splits.items():
        arrays[f"X_{name}"] = np.asarray(X, dtype=np.float32)
        arrays[f"y_{name}"] = np.asarray(y, dtype=np.int64)
        arrays[f"ids_{name}"] = np.asarray(ids, dtype=object)
    path = processed_dir / "splits.npz"
    np.savez_compressed(path, **arrays)
    return path


def load_splits(
    processed_dir: str | Path = config.PROCESSED_DIR, *, include_test: bool = False
) -> DataSplits:
    """
    Load the arrays written by :func:`save_splits`.

    Parameters
    ----------
    include_test : bool, default False
        The test split is only returned when explicitly requested. Notebooks
        03–06 (model development) leave this ``False``; only the final
        evaluation notebook 07 sets it to ``True`` — "report metrics on the
        test set only once".
    """
    path = Path(processed_dir) / "splits.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run notebooks/02_preprocessing.ipynb first.")
    data = np.load(path, allow_pickle=True)  # object arrays hold string IDs / names
    splits = DataSplits(
        X_train=data["X_train"], y_train=data["y_train"],
        X_val=data["X_val"], y_val=data["y_val"],
        ids_train=data["ids_train"], ids_val=data["ids_val"],
        feature_names=[str(n) for n in data["feature_names"]],
    )
    if include_test:
        splits.X_test, splits.y_test, splits.ids_test = data["X_test"], data["y_test"], data["ids_test"]
    return splits
