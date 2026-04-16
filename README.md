# PeriMapper

PeriMapper is a companion tool to [PeriMiner](https://github.com/mdsefero/PeriMiner) — a Streamlit app for exploring ZIP-level geographic patterns in a perinatal dataset. It joins a filtered machine-learning pickle to ZIP lookup columns, aggregates a selected variable by ZIP, and renders an interactive Houston-area choropleth with a downloadable summary table.

## Repository Layout

- `PMap_dashboard.py`: Streamlit app entry point.
- `PMap_build_db.py`: Builds `DB_for_PMap.pkl` from the source pickle by dropping constant and very sparse binary columns.
- `PMap_extract_zips.py`: Extracts ZIP code columns into `data/zip_lookup.csv`.
- `perimapper_core.py`: Shared analytical helpers used by the dashboard and tests.
- `Zip_Codes/`: Static shapefile assets.
- `data/`: Generated outputs such as logs, lookup CSVs, and geometry caches.

## Expected Inputs

PeriMapper expects:

- `PICKLE_PATH` in `.env`, pointing to the source perinatal pickle. See `.env.example` for the full list of supported variables.
- A ZIP lookup CSV at `data/zip_lookup.csv` by default.
- Houston ZIP shapefile assets under `Zip_Codes/`.

The dashboard derives `DB_for_PMap.pkl` from the directory containing `PICKLE_PATH`.

### Shapefile dependency

The ZIP geometry comes from the U.S. Census Bureau's 2023 TIGER/Line ZCTA shapefile, which is too large to ship in the repo. Download `tl_2023_us_zcta520.zip` from the Census Bureau and extract it into `Zip_Codes/` so that `Zip_Codes/tl_2023_us_zcta520.shp` (and its sidecar `.dbf`, `.shx`, `.prj`, `.cpg` files) exist:

<https://www2.census.gov/geo/tiger/TIGER2023/ZCTA520/>

Override the location via `SHAPEFILE_PATH` in `.env` if you place it elsewhere.

## Setup

This project currently depends on an older pandas/geopandas pairing because the source pickle was serialized with an older pandas version.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Build the ZIP lookup if needed:

```bash
python PMap_extract_zips.py --pbdb /path/to/PBDBfinal.txt --pickle /path/to/PBDBfinal_ready_forML_IHCP_paper3.pkl
```

Build the filtered dashboard pickle:

```bash
python PMap_build_db.py --force
```

Start the app:

```bash
streamlit run PMap_dashboard.py
```

## Analytical Notes

- Binary variables are mapped as prevalence percentages by ZIP.
- Continuous variables are mapped as ZIP-level means.
- `Min effective count` uses the minority-class count for binary variables and total non-null rows for continuous variables.
- `Min records per zip` hides low-volume ZIPs from the map but keeps them visible in the summary table.
