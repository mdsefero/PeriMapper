"""
PMap_build_db.py — Build the filtered pickle for PeriMapper.

Reads the full PeriMiner pickle (path from .env PICKLE_PATH), drops columns
that are likely poorly represented / imputed:

  * Binary (0/1) columns with fewer than 10 positive rows are dropped.
  * Constant columns (only one unique value) are dropped.

The result is saved as  DB_for_PMap.pkl  in the **same directory** as the
source pickle.  If DB_for_PMap.pkl already exists the script exits early.

Usage:
    python PMap_build_db.py            # uses PICKLE_PATH from .env
    python PMap_build_db.py --force    # rebuild even if DB_for_PMap.pkl exists
"""

import os
import pickle
import sys

import pandas as pd
from dotenv import load_dotenv

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_SCRIPT_DIR, ".env"))

MIN_ONES = 10  # minimum count of 1s for a binary column to be kept


def _is_binary(series: pd.Series) -> bool:
    unique_vals = series.dropna().unique()
    return len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})


def build(source_path: str, dest_path: str) -> None:
    print(f"Loading source pickle: {source_path}")
    with open(source_path, "rb") as f:
        df = pickle.load(f)
    print(f"  {df.shape[0]:,} rows x {df.shape[1]:,} columns")

    cols_to_drop = []
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        n_unique = s.dropna().nunique()

        # Drop constant columns (zero variance)
        if n_unique <= 1:
            cols_to_drop.append(col)
            continue

        # Drop binary columns with fewer than MIN_ONES ones
        if _is_binary(s):
            n_ones = int((s == 1).sum())
            if n_ones < MIN_ONES:
                cols_to_drop.append(col)

    print(f"  Dropping {len(cols_to_drop):,} columns "
          f"({len(cols_to_drop)}/{df.shape[1]} = "
          f"{len(cols_to_drop)/df.shape[1]*100:.1f}%)")

    df = df.drop(columns=cols_to_drop)
    print(f"  Remaining: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

    with open(dest_path, "wb") as f:
        pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved: {dest_path}")


def main() -> None:
    source = os.getenv("PICKLE_PATH")
    if not source:
        sys.exit("ERROR: PICKLE_PATH not set in .env")
    if not os.path.isabs(source):
        source = os.path.join(_SCRIPT_DIR, source)

    dest = os.path.join(os.path.dirname(source), "DB_for_PMap.pkl")

    force = "--force" in sys.argv
    if os.path.exists(dest) and not force:
        print(f"DB_for_PMap.pkl already exists at:\n  {dest}")
        print("Use --force to rebuild.")
        return

    if not os.path.exists(source):
        sys.exit(f"ERROR: Source pickle not found: {source}")

    build(source, dest)


if __name__ == "__main__":
    main()
