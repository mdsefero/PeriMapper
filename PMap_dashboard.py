# PeriMapper Dashboard  —  streamlit run PMap_dashboard.py
#
# Interactive choropleth of Houston zip codes, shaded by any variable from the
# PeriMiner ML pickle. Pick a variable, pick a zip column, adjust minimum
# sample size, and explore geographic patterns.

import csv
import json
import logging
import os
import pickle as _pkl
import threading
from datetime import datetime, timezone

import geopandas as gpd
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from perimapper_core import (
    aggregate_for_map,
    effective_counts,
    ensure_directory,
    resolve_existing_path,
)

logger = logging.getLogger(__name__)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = ensure_directory(os.path.join(_SCRIPT_DIR, "data"))
_CACHE_DIR = ensure_directory(os.path.join(_DATA_DIR, "cache"))
load_dotenv(os.path.join(_SCRIPT_DIR, ".env"))


# ---------------------------------------------------------------------------
# Usage tracking
# ---------------------------------------------------------------------------
_LOG_FILE = resolve_existing_path(
    os.path.join(_DATA_DIR, "usage_log.csv"),
    os.path.join(_SCRIPT_DIR, "usage_log.csv"),
)
_LOG_LOCK = threading.Lock()
_LOG_COLS = ["timestamp", "event", "variable", "zip_column", "min_total", "min_zip_n", "user_hash"]


def _get_user_token() -> str:
    try:
        hdrs = st.context.headers
        forwarded = hdrs.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    except Exception:
        pass
    return "unknown"


def _log_event(
    event: str,
    variable: str = "",
    zip_column: str = "",
    min_total: int = 0,
    min_zip_n: int = 0,
) -> None:
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "variable": variable,
        "zip_column": zip_column,
        "min_total": min_total,
        "min_zip_n": min_zip_n,
        "user_hash": _get_user_token(),
    }
    write_header = not os.path.exists(_LOG_FILE)
    try:
        with _LOG_LOCK:
            with open(_LOG_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_LOG_COLS)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
    except Exception as exc:
        logger.warning("Unable to write usage log %s: %s", _LOG_FILE, exc)


def _resolve(path: str) -> str:
    """Resolve a path relative to the script directory if not absolute."""
    if os.path.isabs(path):
        return path
    return os.path.join(_SCRIPT_DIR, path)


# ---------------------------------------------------------------------------
# Defaults (override via .env)
# ---------------------------------------------------------------------------
_RAW_PICKLE = os.getenv("PICKLE_PATH", "PBDBfinal_ready_forML_IHCP_paper3.pkl")
if not os.path.isabs(_RAW_PICKLE):
    _RAW_PICKLE = os.path.join(_SCRIPT_DIR, _RAW_PICKLE)

_DEFAULT_PICKLE = os.path.join(os.path.dirname(_RAW_PICKLE), "DB_for_PMap.pkl")
_DEFAULT_ZIP_LOOKUP = resolve_existing_path(
    _resolve(os.getenv("ZIP_LOOKUP_PATH", "data/zip_lookup.csv")),
    os.path.join(_SCRIPT_DIR, "zip_lookup.csv"),
)
_DEFAULT_SHAPEFILE = _resolve(os.getenv("SHAPEFILE_PATH", "Zip_Codes/tl_2023_us_zcta520.shp"))

_HOUSTON_LAT = 29.76
_HOUSTON_LON = -95.37
_HOUSTON_ZOOM = 9

_ZIP_FRIENDLY = {
    "MatInfo__Home zip code at preconception": "Maternal Home - Preconception",
    "MatInfo__Home zip code at 1st trim": "Maternal Home - 1st Trimester",
    "MatInfo__Home zip code at 2nd/3rd trim": "Maternal Home - 2nd/3rd Trimester",
    "MatInfo__Work zip code at preconception": "Maternal Work - Preconception",
    "MatInfo__Work zip code at 1st trim": "Maternal Work - 1st Trimester",
    "MatInfo__Work zip code at 2nd/3rd trim": "Maternal Work - 2nd/3rd Trimester",
    "Harvey__Zip code during Harvey": "Harvey - During Storm",
    "Harvey__Zip code of displacement": "Harvey - Displacement",
    "PatInfo__Home zip code at preconception": "Paternal Home - Preconception",
    "PatInfo__Home zip code at 1st trim": "Paternal Home - 1st Trimester",
    "PatInfo__Home zip code at 2nd/3rd trim": "Paternal Home - 2nd/3rd Trimester",
    "PatInfo__Work zip code at preconception": "Paternal Work - Preconception",
    "PatInfo__Work zip code at 1st trim": "Paternal Work - 1st Trimester",
    "PatInfo__Work zip code at 2nd/3rd trim": "Paternal Work - 2nd/3rd Trimester",
}

st.set_page_config(page_title="PeriMapper", page_icon="map", layout="wide")

if "session_logged" not in st.session_state:
    _log_event("page_visit")
    st.session_state["session_logged"] = True


@st.cache_resource(show_spinner="Loading ML pickle...")
def _load_pickle(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        return _pkl.load(f)


@st.cache_data(show_spinner="Loading zip lookup...")
def _load_zip_lookup(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df["Pregnancy ID"] = pd.to_numeric(df["Pregnancy ID"], errors="coerce")
    df = df.dropna(subset=["Pregnancy ID"])
    df["Pregnancy ID"] = df["Pregnancy ID"].astype(int)
    return df.set_index("Pregnancy ID")


@st.cache_resource(show_spinner="Loading shapefile...")
def _load_shapefile(path: str) -> tuple[gpd.GeoDataFrame, dict]:
    stem = os.path.splitext(os.path.basename(path))[0]
    parquet_cache = os.path.join(_CACHE_DIR, f"{stem}_houston_cache.parquet")
    geojson_cache = os.path.join(_CACHE_DIR, f"{stem}_houston_cache.geojson")

    if os.path.exists(parquet_cache) and os.path.exists(geojson_cache):
        gdf = gpd.read_parquet(parquet_cache)
        with open(geojson_cache, encoding="utf-8") as f:
            geojson = json.load(f)
        return gdf, geojson

    houston_bbox = (-96.5, 28.9, -94.2, 30.6)
    gdf = gpd.read_file(path, bbox=houston_bbox).to_crs(epsg=4326)
    if "ZCTA5CE20" in gdf.columns and "ZIP_CODE" not in gdf.columns:
        gdf = gdf.rename(columns={"ZCTA5CE20": "ZIP_CODE"})
    gdf["ZIP_CODE"] = gdf["ZIP_CODE"].astype(str).str.strip().str.zfill(5)
    gdf["geometry"] = gdf["geometry"].simplify(0.001, preserve_topology=True)
    gdf = gdf[["ZIP_CODE", "geometry"]]
    geojson = json.loads(gdf.to_json())

    try:
        gdf.to_parquet(parquet_cache)
        with open(geojson_cache, "w", encoding="utf-8") as f:
            json.dump(geojson, f)
    except Exception as exc:
        logger.warning("Unable to write geometry cache in %s: %s", _CACHE_DIR, exc)

    return gdf, geojson


@st.cache_data(show_spinner="Joining pickle to zip lookup...")
def _join_data(pickle_df: pd.DataFrame, zip_df: pd.DataFrame) -> pd.DataFrame:
    return pickle_df.join(zip_df, how="inner")


@st.cache_data(show_spinner=False)
def _col_counts(pickle_df: pd.DataFrame) -> pd.Series:
    return pickle_df.notna().sum()


def _build_sidebar(
    pickle_df: pd.DataFrame,
    zip_df: pd.DataFrame,
    joined: pd.DataFrame,
) -> tuple[list[str], str | None, int, int, str | None]:
    def _normalize_search_text(value: str) -> str:
        return "".join(ch.lower() for ch in value if ch.isalnum())

    pickle_cols = list(pickle_df.columns)
    zip_cols = [c for c in zip_df.columns if c in _ZIP_FRIENDLY]

    zip_display = [_ZIP_FRIENDLY.get(c, c) for c in zip_cols]
    default_zip = "Maternal Home - 2nd/3rd Trimester"
    default_idx = zip_display.index(default_zip) if default_zip in zip_display else None

    selected_zip_display = st.selectbox(
        "Zip code column",
        options=zip_display,
        index=default_idx,
        placeholder="Select a zip column...",
    )
    selected_zip_col = dict(zip(zip_display, zip_cols)).get(selected_zip_display)

    if selected_zip_col is not None:
        mappable = joined[joined[selected_zip_col].notna()]
        eff_counts = effective_counts(mappable[pickle_cols])
        total_counts = _col_counts(mappable[pickle_cols])
    else:
        eff_counts = effective_counts(joined[pickle_cols])
        total_counts = _col_counts(joined[pickle_cols])

    slider_max = min(max(int(total_counts.max()), 10), 5000)
    min_total = st.slider(
        "Min effective count to list a variable",
        min_value=10,
        max_value=max(slider_max, 1000),
        value=min(100, slider_max),
        step=10,
        help=(
            "Binary variables use the rarer class count. Continuous variables use total "
            "non-null rows after filtering to records with a valid ZIP in the selected column."
        ),
    )
    min_zip_n = st.slider(
        "Min records per zip to map",
        min_value=1,
        max_value=200,
        value=20,
        help="ZIPs with fewer records for the selected variable remain in the summary but are hidden from the map.",
    )

    eligible_cols = sorted(c for c in pickle_cols if int(eff_counts.get(c, 0)) >= min_total)
    st.caption(f"{len(eligible_cols):,} of {len(pickle_cols):,} variables meet the threshold")

    search_term = st.text_input("Search variables", placeholder="Type to filter...")
    if search_term:
        normalized_search = _normalize_search_text(search_term)
        filtered_cols = [
            c for c in eligible_cols if normalized_search in _normalize_search_text(c)
        ]
    else:
        filtered_cols = eligible_cols
    st.caption(f"{len(filtered_cols):,} variable(s) match")

    st.caption(
        "Map values are ZIP-level means for continuous variables and prevalence for binary variables."
    )

    current_selected = st.session_state.get("selected_var")
    if current_selected not in filtered_cols:
        st.session_state.pop("selected_var", None)

    selected_var = st.selectbox(
        "Variable to map",
        options=filtered_cols,
        index=None,
        placeholder="Select a variable...",
        key="selected_var",
    )
    return pickle_cols, selected_zip_col, min_total, min_zip_n, selected_var


def _render_metrics(agg: pd.DataFrame, excluded: pd.DataFrame, binary: bool, value_label: str) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Zips mapped", f"{len(agg):,}")
    c2.metric("Total pregnancies", f"{agg['count'].sum():,}")
    overall = (agg["value"] * agg["count"]).sum() / agg["count"].sum()
    c3.metric(f"Overall {value_label.lower()}", f"{overall:.2f}{'%' if binary else ''}")
    c4.metric("Zips excluded (low N)", f"{len(excluded):,}")


def _render_map(map_gdf: gpd.GeoDataFrame, geojson: dict, value_label: str) -> None:
    fig = px.choropleth_mapbox(
        map_gdf,
        geojson=geojson,
        locations="ZIP_CODE",
        featureidkey="properties.ZIP_CODE",
        color="value",
        color_continuous_scale="YlOrRd",
        mapbox_style="carto-positron",
        center={"lat": _HOUSTON_LAT, "lon": _HOUSTON_LON},
        zoom=_HOUSTON_ZOOM,
        hover_data={"ZIP_CODE": True, "value": ":.2f", "count": True},
        labels={"value": value_label, "count": "N pregnancies"},
    )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=600)
    st.plotly_chart(fig, use_container_width=True)


def _render_table(agg: pd.DataFrame, value_label: str, selected_var: str, selected_zip_col: str) -> None:
    st.subheader("Per-zip summary")
    display_df = agg.rename(columns={"value": value_label, "count": "N pregnancies"}).sort_values(
        value_label, ascending=False
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    if st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name=f"perimapper_{selected_var}_{selected_zip_col}.csv",
        mime="text/csv",
    ):
        _log_event("csv_download", variable=selected_var, zip_column=selected_zip_col)


def main() -> None:
    st.title("PeriMapper")

    with st.sidebar:
        st.header("PeriMapper")
        st.caption("Geographic patterns in perinatal data")

        if not os.path.exists(_DEFAULT_PICKLE):
            st.error(
                "Filtered pickle **DB_for_PMap.pkl** not found.\n\n"
                "Run `python PMap_build_db.py` first to generate it."
            )
            st.stop()

        try:
            pickle_df = _load_pickle(_DEFAULT_PICKLE)
            zip_df = _load_zip_lookup(_DEFAULT_ZIP_LOOKUP)
            shp_gdf, geojson = _load_shapefile(_DEFAULT_SHAPEFILE)
            joined = _join_data(pickle_df, zip_df)
        except Exception as exc:
            st.error(f"Failed to load data: {exc}")
            st.stop()

        pickle_cols, selected_zip_col, min_total, min_zip_n, selected_var = _build_sidebar(
            pickle_df, zip_df, joined
        )

    if selected_var is None or selected_zip_col is None:
        st.info("Select a variable and a zip column in the sidebar to generate the map.")
        return

    agg, excluded, binary, value_label = aggregate_for_map(joined, selected_var, selected_zip_col, min_zip_n)
    if agg.empty:
        st.warning(
            "No ZIP codes remain after filtering the selected variable. "
            "Try lowering the minimum records per zip or choose another variable."
        )
        return

    agg = agg.rename(columns={"zip": "ZIP_CODE"})
    map_gdf = shp_gdf.merge(agg, on="ZIP_CODE", how="inner")

    map_key = f"logged_map_{selected_var}_{selected_zip_col}"
    if map_key not in st.session_state:
        _log_event(
            "map_generated",
            variable=selected_var,
            zip_column=selected_zip_col,
            min_total=min_total,
            min_zip_n=min_zip_n,
        )
        st.session_state[map_key] = True

    _render_metrics(agg, excluded, binary, value_label)
    _render_map(map_gdf, geojson, value_label)
    _render_table(agg, value_label, selected_var, selected_zip_col)

    st.divider()
    st.caption(
        "PeriMapper · Geographic exploration of perinatal variables across Houston-area ZIP codes."
    )


main()
