"""
extract_zips.py — PeriMapper Step 1

Extracts zip code columns from PBDBfinal.txt and produces a lookup CSV
aligned to the Pregnancy IDs present in the final PeriMiner ML pickle.

Usage:
    python PMap_extract_zips.py \
        --pbdb   PBDBfinal.txt \
        --pickle PBDBfinal_ready_forML_IHCP_paper3.pkl \
        --out    data/zip_lookup.csv
"""

import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Extract zip codes from PBDBfinal.txt")
    parser.add_argument("--pbdb",   default="PBDBfinal.txt",
                        help="Path to PBDBfinal.txt (pipe-delimited, DB_1 output)")
    parser.add_argument("--pickle", default="PBDBfinal_ready_forML_IHCP_paper3.pkl",
                        help="Path to final ML pickle (provides valid Pregnancy ID set)")
    parser.add_argument("--out",    default="data/zip_lookup.csv",
                        help="Output CSV path")
    args = parser.parse_args()
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Stream PBDBfinal.txt and pull out Pregnancy ID + zip columns
    # ------------------------------------------------------------------
    print(f"Reading {args.pbdb} ...")
    reader = pd.read_csv(args.pbdb, sep="|", dtype=str, chunksize=10_000)

    zip_cols = None
    keep = None
    chunks = []

    for chunk in reader:
        if zip_cols is None:
            # Detect zip columns once from the header
            zip_cols = [c for c in chunk.columns if "ip code" in c]
            if not zip_cols:
                raise ValueError(
                    "No zip code columns found in PBDBfinal.txt. "
                    "Expected columns containing 'ip code' (e.g. 'Home zip code at preconception')."
                )
            keep = ["Pregnancy ID"] + zip_cols
            print(f"  Found {len(zip_cols)} zip column(s): {zip_cols}")
        chunks.append(chunk[keep])

    df_zip = pd.concat(chunks, ignore_index=True)
    df_zip = df_zip.set_index("Pregnancy ID")
    df_zip.index = pd.to_numeric(df_zip.index, errors="coerce")
    df_zip = df_zip[df_zip.index.notna()]
    df_zip.index = df_zip.index.astype(int)
    print(f"  {len(df_zip):,} rows with valid Pregnancy ID in PBDBfinal.txt")

    # ------------------------------------------------------------------
    # 2. Load the ML pickle and get the surviving Pregnancy IDs
    # ------------------------------------------------------------------
    print(f"Loading {args.pickle} ...")
    ml_index = pd.read_pickle(args.pickle).index
    print(f"  {len(ml_index):,} Pregnancy IDs in ML pickle")

    # ------------------------------------------------------------------
    # 3. Filter zip table to IDs that made it through the pipeline
    # ------------------------------------------------------------------
    df_zip = df_zip.loc[df_zip.index.isin(ml_index)]
    print(f"  {len(df_zip):,} matched rows after inner join")

    # ------------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------------
    df_zip.to_csv(args.out)
    print(f"Saved → {args.out}  ({len(df_zip):,} rows × {len(zip_cols)} zip columns)")


if __name__ == "__main__":
    main()
