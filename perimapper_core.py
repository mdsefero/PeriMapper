import os

import pandas as pd


HOUSTON_ZIP_PREFIXES = ("770", "772", "773", "774", "775")


def resolve_existing_path(*candidates: str) -> str:
    """Return the first existing path, falling back to the first candidate."""
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return candidates[0]


def ensure_directory(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def is_binary(series: pd.Series) -> bool:
    unique_vals = series.dropna().unique()
    return len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})


def effective_counts(df: pd.DataFrame) -> pd.Series:
    """Return filter counts that better reflect mappable analytical value."""
    result = {}
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        n_unique = s.nunique()
        if n_unique <= 1:
            result[col] = 0
        elif n_unique == 2:
            result[col] = int(s.value_counts().min())
        else:
            result[col] = len(s)
    return pd.Series(result)


def normalize_zip_series(series: pd.Series) -> pd.Series:
    """Normalize ZIP codes to five-character strings while preserving index."""
    numeric = pd.to_numeric(series, errors="coerce")
    normalized = pd.Series(pd.NA, index=series.index, dtype="object")
    valid = numeric.notna()
    normalized.loc[valid] = (
        numeric.loc[valid]
        .astype(int)
        .astype(str)
        .str.zfill(5)
    )
    return normalized


def is_plausible_zip(zip_code: str) -> bool:
    return len(zip_code) == 5 and zip_code.isdigit() and zip_code.startswith(HOUSTON_ZIP_PREFIXES)


def filter_plausible_zips(series: pd.Series) -> pd.Series:
    return series.where(series.astype(str).map(is_plausible_zip))


def aggregate_for_map(
    joined: pd.DataFrame,
    selected_var: str,
    selected_zip_col: str,
    min_zip_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, bool, str]:
    """Aggregate a selected variable by ZIP for mapping."""
    work = joined[[selected_var, selected_zip_col]].copy()
    work.columns = ["value", "zip_raw"]
    work["zip"] = filter_plausible_zips(normalize_zip_series(work["zip_raw"]))
    work = work.dropna(subset=["zip"])
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work = work.dropna(subset=["value"])

    if work.empty:
        return pd.DataFrame(), pd.DataFrame(), False, "Mean value"

    binary = is_binary(work["value"])
    value_label = "Prevalence (%)" if binary else "Mean value"

    agg = work.groupby("zip").agg(
        value=("value", "mean"),
        count=("value", "size"),
    ).reset_index()

    if binary:
        agg["value"] = agg["value"] * 100

    excluded = agg[agg["count"] < min_zip_n].copy()
    agg = agg[agg["count"] >= min_zip_n].copy()
    return agg, excluded, binary, value_label
