#!/usr/bin/env python3
"""Interactive Washington Mapillary and KartaView coverage dashboard."""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "assets"
STATEWIDE_CSV = DATA_DIR / "coverage_by_year_statewide.csv"
ROAD_CLASS_CSV = DATA_DIR / "coverage_by_year_and_road_class.csv"
CUMULATIVE_CSV = DATA_DIR / "cumulative_coverage.csv"
OVERLAP_CSV = DATA_DIR / "cumulative_overlap.csv"
TRACKS_GPKG = DATA_DIR / "imagery_tracks_by_year.gpkg"
MAP_COVERAGE_FILES = {
    "Mapillary": DATA_DIR / "mapillary_coverage_2020_2026.fgb",
    "KartaView": DATA_DIR / "kartaview_coverage_2020_2026.fgb",
}
WASHINGTON_BOUNDARY = DATA_DIR / "washington_state_boundary.geojson"
STATE_JSON = DATA_DIR / "statewide_analysis_state.json"
KARTAVIEW_CACHE = DATA_DIR / "kartaview_cache.sqlite"
YEARS = list(range(2020, 2027))
PROVIDER_COLORS = {"Mapillary": "#168f75", "KartaView": "#7856c8"}
ROAD_CLASS_ORDER = [
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "residential",
    "living_street",
    "unclassified",
    "service",
    "unknown",
]
STATEWIDE_COORDINATE_GRID_DEGREES = 0.00005
STATEWIDE_SIMPLIFY_TOLERANCE_METERS = 50.0


st.set_page_config(
    page_title="Washington imagery coverage",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
      --ink: #17231f;
      --muted: #5f6e68;
      --paper: #f7f7f2;
      --line: #d9dfda;
      --mapillary: #168f75;
      --kartaview: #7856c8;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stSidebar"] { background: #edf1ec; }
    [data-testid="stMetric"] {
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 0.85rem 1rem;
      box-shadow: 0 2px 8px rgba(30, 45, 38, 0.04);
    }
    .coverage-kicker {
      color: var(--muted);
      font-size: 0.76rem;
      font-weight: 700;
      letter-spacing: 0.11em;
      text-transform: uppercase;
      margin-bottom: 0.25rem;
    }
    .coverage-title {
      font-family: Georgia, 'Times New Roman', serif;
      font-size: clamp(2.1rem, 5vw, 4.2rem);
      font-weight: 500;
      line-height: 0.98;
      letter-spacing: -0.045em;
      margin: 0 0 0.7rem 0;
    }
    .coverage-subtitle {
      color: var(--muted);
      font-size: 1.05rem;
      max-width: 62rem;
      line-height: 1.55;
      margin-bottom: 1.2rem;
    }
    .status-chip {
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 99px;
      background: #fff;
      padding: 0.28rem 0.65rem;
      color: var(--muted);
      font-size: 0.78rem;
      font-weight: 650;
      margin: 0 0.35rem 0.35rem 0;
    }
    .provider-dot {
      display: inline-block;
      width: 0.58rem;
      height: 0.58rem;
      border-radius: 50%;
      margin-right: 0.35rem;
    }
    .section-rule { border-top: 1px solid var(--line); margin: 1.8rem 0 1rem; }
    .method-note {
      background: #eef2ed;
      border-left: 4px solid #91a399;
      padding: 0.85rem 1rem;
      color: #46554f;
      font-size: 0.9rem;
      line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def file_version(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return 0


@st.cache_data(show_spinner=False)
def load_csv(path: str, version: int) -> pd.DataFrame:
    del version
    frame = pd.read_csv(path)
    numeric = [
        "capture_year",
        "total_road_km",
        "covered_road_km",
        "road_km_with_any_coverage",
        "segments_with_coverage",
        "coverage_pct",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["provider"] = frame["provider"].astype(str)
    frame["road_class"] = frame["road_class"].astype(str)
    return frame


def load_progress() -> dict:
    result = {
        "queries": {},
        "sequences": {},
        "coverage": {},
        "photos": 0,
        "phase": "not_started",
        "updated_at": None,
    }
    if KARTAVIEW_CACHE.is_file():
        uri = f"file:{KARTAVIEW_CACHE}?mode=ro"
        try:
            with sqlite3.connect(uri, uri=True, timeout=3) as con:
                for table, key in (
                    ("coverage_tiles", "coverage"),
                    ("queries", "queries"),
                    ("sequences", "sequences"),
                ):
                    result[key] = dict(
                        con.execute(
                            f"SELECT status, COUNT(*) FROM {table} GROUP BY status"
                        ).fetchall()
                    )
                result["photos"] = int(
                    con.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
                )
        except sqlite3.Error as exc:
            result["read_error"] = str(exc)
    if STATE_JSON.is_file():
        try:
            state = json.loads(STATE_JSON.read_text(encoding="utf-8"))
            result["phase"] = state.get("phase", result["phase"])
            result["updated_at"] = state.get("updated_at")
        except (OSError, json.JSONDecodeError):
            pass
    return result


def format_phase(phase: str) -> str:
    return phase.replace("_", " ").strip().title()


def parse_coordinate(value: str) -> tuple[float, float]:
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", value)
    if len(numbers) != 2:
        raise ValueError("Enter one latitude and longitude pair, such as 47.6062, -122.3321.")
    latitude, longitude = map(float, numbers)
    if not (45.3 <= latitude <= 49.2 and -125.0 <= longitude <= -116.8):
        raise ValueError(
            "The coordinate must be in or immediately around Washington and use "
            "latitude, longitude order."
        )
    return latitude, longitude


def latest_data_time() -> str:
    versions = [file_version(STATEWIDE_CSV), file_version(ROAD_CLASS_CSV)]
    newest = max(versions)
    if not newest:
        return "not available"
    return datetime.fromtimestamp(newest / 1_000_000_000).astimezone().strftime(
        "%b %d, %Y · %H:%M %Z"
    )


def provider_legend(providers: list[str]) -> None:
    items = []
    for provider in providers:
        color = PROVIDER_COLORS.get(provider, "#6b756f")
        items.append(
            f'<span class="status-chip"><span class="provider-dot" '
            f'style="background:{color}"></span>{provider}</span>'
        )
    st.markdown("".join(items), unsafe_allow_html=True)


@st.fragment(run_every=30)
def live_collection_status() -> None:
    progress = load_progress()
    queries = progress["queries"]
    complete = int(queries.get("complete", 0))
    failed = int(queries.get("failed", 0))
    pending = int(queries.get("pending", 0))
    total = complete + failed + pending
    fraction = complete / total if total else 0.0

    st.markdown("#### Live KartaView collection")
    if total:
        st.progress(
            fraction,
            text=f"{complete:,} of {total:,} discovery queries complete ({fraction:.1%})",
        )
        first_row = st.columns(2)
        first_row[0].metric("Queries remaining", f"{pending:,}")
        first_row[1].metric("Photos cached", f"{progress['photos']:,}")
        second_row = st.columns(2)
        second_row[0].metric(
            "Sequences found",
            f"{sum(progress['sequences'].values()):,}",
        )
        second_row[1].metric("Retry queue", f"{failed:,}")
        if failed:
            st.caption(
                "Failed requests are retained in the resumable cache and retried "
                "by the statewide supervisor."
            )
    else:
        st.info("KartaView collection has not started or its cache is unavailable.")


def annual_chart(frame: pd.DataFrame) -> alt.Chart:
    providers = list(dict.fromkeys(frame["provider"].astype(str)))
    base = alt.Chart(frame).encode(
        x=alt.X(
            "capture_year:O",
            title="Capture year",
            axis=alt.Axis(labelAngle=0),
        ),
        color=alt.Color(
            "provider:N",
            title=None,
            scale=alt.Scale(
                domain=providers,
                range=[PROVIDER_COLORS.get(p, "#6b756f") for p in providers],
            ),
        ),
        tooltip=[
            alt.Tooltip("provider:N", title="Provider"),
            alt.Tooltip("capture_year:O", title="Year"),
            alt.Tooltip("covered_road_km:Q", title="Covered road", format=",.1f"),
            alt.Tooltip("coverage_pct:Q", title="Coverage", format=".2f"),
            alt.Tooltip(
                "segments_with_coverage:Q", title="Segments touched", format=",.0f"
            ),
        ],
    )
    lines = base.mark_line(point=alt.OverlayMarkDef(size=70), strokeWidth=3).encode(
        y=alt.Y("coverage_pct:Q", title="Road length covered (%)")
    )
    return lines.properties(height=360)


@st.cache_data(show_spinner=False)
def load_cumulative_coverage(path: str, version: int) -> pd.DataFrame:
    """Load deployment-ready cumulative coverage summaries."""
    del version
    frame = pd.read_csv(path)
    frame["cutoff_year"] = pd.to_numeric(frame["cutoff_year"]).astype(int)
    return frame


@st.cache_data(max_entries=8, show_spinner=False)
def load_cumulative_overlap(
    path: str,
    version: int,
    cutoff_year: int,
) -> pd.DataFrame:
    """Load provider overlap for one deployment-ready cutoff year."""
    del version
    frame = pd.read_csv(path)
    return frame[pd.to_numeric(frame["cutoff_year"]).eq(int(cutoff_year))].copy()


def cumulative_chart(frame: pd.DataFrame) -> alt.Chart:
    providers = list(dict.fromkeys(frame["provider"].astype(str)))
    return (
        alt.Chart(frame)
        .mark_line(point=alt.OverlayMarkDef(size=70), strokeWidth=3)
        .encode(
            x=alt.X(
                "cutoff_year:O",
                title="Coverage through year",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "touched_pct:Q",
                title="Cumulative road network touched (%)",
                scale=alt.Scale(zero=True),
            ),
            color=alt.Color(
                "provider:N",
                title=None,
                scale=alt.Scale(
                    domain=providers,
                    range=[PROVIDER_COLORS.get(p, "#6b756f") for p in providers],
                ),
            ),
            tooltip=[
                alt.Tooltip("provider:N", title="Provider"),
                alt.Tooltip("cutoff_year:O", title="Through year"),
                alt.Tooltip("touched_road_km:Q", title="Road touched (km)", format=",.0f"),
                alt.Tooltip("touched_pct:Q", title="Network touched", format=".2f"),
                alt.Tooltip("segments_touched:Q", title="Segments", format=",.0f"),
            ],
        )
        .properties(height=360)
    )


def comparison_class_chart(frame: pd.DataFrame) -> alt.Chart:
    providers = list(dict.fromkeys(frame["provider"].astype(str)))
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            y=alt.Y(
                "road_class:N",
                title=None,
                sort=[x for x in ROAD_CLASS_ORDER if x in set(frame["road_class"])],
            ),
            x=alt.X("touched_pct:Q", title="Cumulative road network touched (%)"),
            yOffset="provider:N",
            color=alt.Color(
                "provider:N",
                title=None,
                scale=alt.Scale(
                    domain=providers,
                    range=[PROVIDER_COLORS.get(p, "#6b756f") for p in providers],
                ),
            ),
            tooltip=[
                alt.Tooltip("provider:N", title="Provider"),
                alt.Tooltip("road_class:N", title="Road class"),
                alt.Tooltip("touched_road_km:Q", title="Road touched (km)", format=",.0f"),
                alt.Tooltip("touched_pct:Q", title="Network touched", format=".2f"),
            ],
        )
        .properties(height=360)
    )


def overlap_chart(frame: pd.DataFrame) -> alt.Chart:
    order = ["Mapillary only", "Both", "KartaView only", "Neither"]
    colors = [PROVIDER_COLORS["Mapillary"], "#33443d", PROVIDER_COLORS["KartaView"], "#c8ceca"]
    return (
        alt.Chart(frame)
        .mark_bar(size=42)
        .encode(
            x=alt.X(
                "road_km:Q",
                stack="zero",
                axis=alt.Axis(
                    labels=False,
                    ticks=False,
                    domain=False,
                    title=None,
                ),
            ),
            color=alt.Color(
                "availability:N",
                title=None,
                sort=order,
                scale=alt.Scale(domain=order, range=colors),
                legend=alt.Legend(orient="bottom", direction="horizontal"),
            ),
            order=alt.Order("order:Q"),
            tooltip=[
                alt.Tooltip("availability:N", title="Availability"),
                alt.Tooltip("road_km:Q", title="Road network", format=",.0f"),
                alt.Tooltip("share_pct:Q", title="Share", format=".2f"),
            ],
        )
        .properties(height=110)
    )


def road_class_chart(frame: pd.DataFrame) -> alt.Chart:
    providers = list(dict.fromkeys(frame["provider"].astype(str)))
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            y=alt.Y(
                "road_class:N",
                title=None,
                sort=[x for x in ROAD_CLASS_ORDER if x in set(frame["road_class"])],
            ),
            x=alt.X("coverage_pct:Q", title="Road length covered (%)"),
            color=alt.Color(
                "provider:N",
                title=None,
                scale=alt.Scale(
                    domain=providers,
                    range=[PROVIDER_COLORS.get(p, "#6b756f") for p in providers],
                ),
            ),
            row=alt.Row(
                "capture_year:O",
                title=None,
                header=alt.Header(labelOrient="top", labelFontSize=14),
            ),
            tooltip=[
                alt.Tooltip("provider:N", title="Provider"),
                alt.Tooltip("capture_year:O", title="Year"),
                alt.Tooltip("road_class:N", title="Road class"),
                alt.Tooltip("covered_road_km:Q", title="Covered km", format=",.1f"),
                alt.Tooltip("total_road_km:Q", title="Road km", format=",.1f"),
                alt.Tooltip("coverage_pct:Q", title="Coverage", format=".2f"),
            ],
        )
        .properties(height=260)
        .resolve_scale(y="independent")
    )


@st.cache_data(show_spinner=False)
def load_boundary_geojson(path: str, version: int) -> dict:
    del version
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(max_entries=5, show_spinner=False)
def load_local_base_roads(
    path: str,
    version: int,
    latitude: float,
    longitude: float,
    radius_km: float,
) -> tuple[dict, dict]:
    del version
    import geopandas as gpd
    import shapely
    from pyproj import Transformer

    latitude_pad = radius_km / 110.574 * 1.15
    longitude_pad = radius_km / (111.320 * max(0.2, abs(__import__("math").cos(__import__("math").radians(latitude))))) * 1.15
    bbox = (
        longitude - longitude_pad,
        latitude - latitude_pad,
        longitude + longitude_pad,
        latitude + latitude_pad,
    )
    roads = gpd.read_file(
        path,
        layer="washington_overture_roads_qgis",
        bbox=bbox,
        engine="pyogrio",
    )
    to_analysis = Transformer.from_crs(4326, 5070, always_xy=True)
    to_wgs84 = Transformer.from_crs(5070, 4326, always_xy=True)
    center = shapely.Point(*to_analysis.transform(longitude, latitude))
    search_area = center.buffer(radius_km * 1000.0)
    if roads.empty:
        return {"type": "FeatureCollection", "features": []}, shapely.geometry.mapping(
            shapely.transform(search_area, to_wgs84.transform, interleaved=False)
        )
    roads = roads.to_crs(5070)
    roads = roads[roads.geometry.intersects(search_area)].copy()
    roads["geometry"] = roads.geometry.intersection(search_area)
    roads = roads[~roads.geometry.is_empty].to_crs(4326)
    roads["geometry"] = roads.geometry.simplify(0.000015, preserve_topology=True)
    road_geojson = json.loads(roads[["geometry"]].to_json())
    area_geojson = shapely.geometry.mapping(
        shapely.transform(search_area, to_wgs84.transform, interleaved=False)
    )
    return road_geojson, area_geojson


@st.cache_data(max_entries=12, show_spinner="Loading original imagery tracks…")
def load_local_provider_tracks(
    provider: str,
    years: tuple[int, ...],
    latitude: float,
    longitude: float,
    radius_km: float,
    combined_path: str,
    combined_version: int,
) -> tuple[dict, int]:
    del combined_version
    import geopandas as gpd
    from pyproj import Transformer
    import shapely

    latitude_pad = radius_km / 110.574 * 1.15
    longitude_pad = radius_km / (
        111.320
        * max(0.2, abs(__import__("math").cos(__import__("math").radians(latitude))))
    ) * 1.15
    bbox = (
        longitude - longitude_pad,
        latitude - latitude_pad,
        longitude + longitude_pad,
        latitude + latitude_pad,
    )
    path = Path(combined_path)
    if not path.is_file():
        return {}, 0
    frame = gpd.read_file(path, layer="imagery_tracks_by_year", bbox=bbox)
    frame = frame.loc[
        frame["provider"].astype(str).eq(provider)
        & pd.to_numeric(frame["capture_year"], errors="coerce").isin(years)
    ].copy()
    frame["imagery_type"] = "Sequence track"
    if frame.empty:
        return {}, 0
    if frame.crs is None:
        frame = frame.set_crs(4326)
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].to_crs(5070)
    center = shapely.Point(
        *Transformer.from_crs(4326, 5070, always_xy=True).transform(longitude, latitude)
    )
    search_area = center.buffer(radius_km * 1000.0)
    frame = frame[frame.geometry.intersects(search_area)].copy()
    if frame.empty:
        return {}, 0
    frame["geometry"] = frame.geometry.intersection(search_area)
    frame = frame[~frame.geometry.is_empty].to_crs(4326)
    frame["geometry"] = frame.geometry.simplify(0.000008, preserve_topology=True)
    frame["layer_type"] = "Original provider track"
    frame["road_class"] = ""
    frame["coverage_bucket"] = ""
    return json.loads(
        frame[
            [
                "layer_type",
                "imagery_type",
                "road_class",
                "coverage_bucket",
                "geometry",
            ]
        ].to_json()
    ), len(frame)


@st.cache_data(max_entries=12, show_spinner="Loading covered-road map…")
def load_covered_road_geojson(
    coverage_path: str,
    coverage_version: int,
    years: tuple[int, ...],
    road_classes: tuple[str, ...],
    focus: tuple[float, float, float] | None,
) -> tuple[dict[str, dict], int]:
    """Read the compact provider layer prepared for cloud deployment."""
    del coverage_version
    import geopandas as gpd
    import shapely
    from pyproj import Transformer

    bbox = None
    search_area = None
    if focus is not None:
        latitude, longitude, radius_km = focus
        to_analysis = Transformer.from_crs(4326, 5070, always_xy=True)
        center = shapely.Point(*to_analysis.transform(longitude, latitude))
        search_area = center.buffer(radius_km * 1000.0)
        bbox = search_area.bounds

    frame = gpd.read_file(coverage_path, bbox=bbox, engine="pyogrio")
    year_columns = [f"y{year}" for year in years if f"y{year}" in frame.columns]
    if not year_columns:
        return {}, 0
    year_mask = frame[year_columns].fillna(0).astype(bool).any(axis=1)
    frame = frame.loc[
        year_mask & frame["road_class"].astype(str).isin(road_classes)
    ].copy()
    if frame.empty:
        return {}, 0

    if frame.crs is None:
        frame = frame.set_crs(5070)
    if search_area is not None:
        frame = frame[frame.geometry.intersects(search_area)].copy()
        frame["geometry"] = frame.geometry.intersection(search_area)
        frame = frame[~frame.geometry.is_empty]
    if frame.empty:
        return {}, 0

    segment_count = int(pd.to_numeric(frame["segment_count"], errors="coerce").sum())
    features = []
    transformer = Transformer.from_crs(frame.crs, 4326, always_xy=True)
    for road_class in road_classes:
        geometries = frame.loc[
            frame["road_class"].astype(str).eq(road_class), "geometry"
        ].array
        if len(geometries) == 0:
            continue
        merged = shapely.line_merge(shapely.union_all(geometries))
        if focus is None:
            merged = shapely.simplify(
                merged,
                STATEWIDE_SIMPLIFY_TOLERANCE_METERS,
                preserve_topology=False,
            )
        merged = shapely.transform(merged, transformer.transform, interleaved=False)
        if focus is None:
            merged = shapely.set_precision(
                merged,
                STATEWIDE_COORDINATE_GRID_DEGREES,
                mode="valid_output",
            )
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "layer_type": "Derived covered road",
                    "road_class": road_class,
                    "coverage_bucket": "available",
                    "imagery_type": "",
                },
                "geometry": shapely.geometry.mapping(merged),
            }
        )
    if not features:
        return {}, 0
    return {"full": {"type": "FeatureCollection", "features": features}}, segment_count


def render_covered_roads_map(
    providers_to_show: list[str],
    years_to_show: list[int],
    time_label: str,
    road_classes: list[str],
    focus: tuple[float, float, float] | None,
    show_source_tracks: bool,
) -> None:
    st.subheader("Roads with imagery coverage")
    if not WASHINGTON_BOUNDARY.is_file():
        st.error("The packaged Washington boundary is missing.")
        return
    if not road_classes:
        st.info("Select at least one road class to draw the map.")
        return

    coverage_paths = {
        provider: MAP_COVERAGE_FILES[provider]
        for provider in providers_to_show
        if provider in MAP_COVERAGE_FILES and MAP_COVERAGE_FILES[provider].is_file()
    }
    available_providers = [
        provider for provider, path in coverage_paths.items() if path.is_file()
    ]
    missing_providers = [
        provider for provider in providers_to_show if provider not in available_providers
    ]
    if not available_providers:
        st.info(
            f"Coverage geometry for {time_label} is not available for the selected "
            "provider layers yet."
        )
        return

    try:
        import pydeck as pdk

        boundary = load_boundary_geojson(
            str(WASHINGTON_BOUNDARY), file_version(WASHINGTON_BOUNDARY)
        )
        bucket_style = {
            "low": (75, 1.0),
            "partial": (120, 1.4),
            "high": (165, 1.9),
            "full": (210, 2.5),
        }
        layers = [
            pdk.Layer(
                "GeoJsonLayer",
                boundary,
                stroked=True,
                filled=True,
                get_fill_color=[247, 247, 242, 12],
                get_line_color=[36, 49, 43, 230],
                get_line_width=2.2,
                line_width_min_pixels=1.8,
                pickable=False,
            )
        ]
        if focus is not None:
            latitude, longitude, radius_km = focus

        segment_counts: dict[str, int] = {}
        track_counts: dict[str, int] = {}
        for provider in available_providers:
            coverage_layers, segment_counts[provider] = load_covered_road_geojson(
                str(coverage_paths[provider]),
                file_version(coverage_paths[provider]),
                tuple(years_to_show),
                tuple(road_classes),
                focus,
            )
            color_hex = PROVIDER_COLORS.get(provider, "#56635d").lstrip("#")
            rgb = [int(color_hex[i : i + 2], 16) for i in (0, 2, 4)]
            for bucket in ("low", "partial", "high", "full"):
                if bucket not in coverage_layers:
                    continue
                for feature in coverage_layers[bucket]["features"]:
                    feature["properties"]["provider"] = provider
                alpha, width = bucket_style[bucket]
                layers.append(
                    pdk.Layer(
                        "GeoJsonLayer",
                        coverage_layers[bucket],
                        stroked=True,
                        filled=False,
                        get_line_color=rgb + [alpha],
                        get_line_width=width,
                        line_width_min_pixels=1.2,
                        pickable=True,
                    )
                )

            track_counts[provider] = 0
            if focus is not None and show_source_tracks:
                source_tracks, track_counts[provider] = load_local_provider_tracks(
                    provider,
                    tuple(years_to_show),
                    latitude,
                    longitude,
                    radius_km,
                    str(TRACKS_GPKG),
                    file_version(TRACKS_GPKG),
                )
                if source_tracks:
                    for feature in source_tracks["features"]:
                        feature["properties"]["provider"] = provider
                    layers.append(
                        pdk.Layer(
                            "GeoJsonLayer",
                            source_tracks,
                            stroked=True,
                            filled=False,
                            get_line_color=rgb + [255],
                            get_line_width=1.15,
                            line_width_min_pixels=1.7,
                            pickable=True,
                        )
                    )

        if focus is not None:
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    [
                        {
                            "position": [longitude, latitude],
                            "provider": "",
                            "layer_type": "Searched coordinate",
                            "imagery_type": "",
                            "road_class": "",
                            "coverage_bucket": "",
                        }
                    ],
                    get_position="position",
                    get_fill_color=[255, 255, 255, 255],
                    get_line_color=[24, 31, 28, 255],
                    get_radius=12,
                    radius_min_pixels=5,
                    radius_max_pixels=8,
                    stroked=True,
                    line_width_min_pixels=2,
                    pickable=True,
                )
            )
            zoom_by_radius = {
                1.0: 13.0,
                2.0: 12.0,
                5.0: 10.7,
                10.0: 9.7,
                20.0: 8.7,
            }
            view = pdk.ViewState(
                latitude=latitude,
                longitude=longitude,
                zoom=zoom_by_radius.get(float(radius_km), 10.7),
            )
        else:
            view = pdk.ViewState(latitude=47.35, longitude=-120.75, zoom=5.45)

        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view,
            map_style="light",
            tooltip={
                "text": "{provider}\n{layer_type}\n"
                "{imagery_type}{road_class}\n{coverage_bucket}"
            },
        )
        provider_key = "-".join(available_providers)
        years_key = "-".join(str(year) for year in years_to_show)
        st.pydeck_chart(
            deck,
            use_container_width=True,
            height=560,
            key=(
                f"covered-road-map-v5-{provider_key}-{years_key}-"
                f"{focus if focus is not None else 'statewide'}-{show_source_tracks}"
            ),
        )
        count_summary = "; ".join(
            f"{provider}: {segment_counts[provider]:,} covered road segments"
            + (
                f", {track_counts.get(provider, 0):,} source track fragments"
                if focus is not None and show_source_tracks
                else ""
            )
            for provider in available_providers
        )
        location_summary = (
            "statewide"
            if focus is None
            else (
                f"within {focus[2]:g} km of {focus[0]:.5f}, "
                f"{focus[1]:.5f}"
            )
        )
        st.caption(
            f"{time_label} · {location_summary} · {count_summary}. "
            "Mapillary is teal; KartaView is purple. Wide translucent lines are "
            "derived covered roads; narrow solid lines are original provider tracks."
            + (
                " Statewide geometry is generalized to map scale; coordinate searches "
                "retain full detail."
                if focus is None
                else ""
            )
        )
        if missing_providers:
            st.info(
                f"{', '.join(missing_providers)} {time_label} geometry is still pending. "
                "It will join this map after processing finishes and Refresh data is pressed."
            )
    except Exception as exc:  # keep the analytical charts usable if map I/O fails
        st.warning(f"Covered-road map is temporarily unavailable: {exc}")


def render_explore_tab(
    _statewide: pd.DataFrame,
    available_years: list[int],
    selected_classes: list[str],
    complete_comparison: bool,
) -> None:
    latest_year = max(available_years)
    year_col, cumulative_col = st.columns([1.6, 1])
    focus_year = year_col.select_slider(
        "Imagery year",
        options=available_years,
        value=latest_year,
        help="Choose the capture year used as the end of the map period.",
    )
    cumulative_mode = cumulative_col.toggle(
        "Cumulative coverage",
        value=True,
        key="explore_cumulative_mode",
        help=(
            "On: show imagery from 2020 through the selected year. "
            "Off: show imagery captured only in the selected year."
        ),
    )
    if cumulative_mode:
        years_to_show = [year for year in available_years if year <= focus_year]
        time_label = f"2020–{focus_year} cumulative"
        cumulative_col.caption(f"Showing imagery captured from 2020 through {focus_year}.")
    else:
        years_to_show = [focus_year]
        time_label = str(focus_year)
        cumulative_col.caption(f"Showing imagery captured in {focus_year} only.")
    provider_toggle_cols = st.columns(2)
    show_mapillary = provider_toggle_cols[0].checkbox(
        "Mapillary layer",
        value=True,
        key="show_mapillary_layer",
    )
    show_kartaview = provider_toggle_cols[1].checkbox(
        "KartaView layer",
        value=True,
        key="show_kartaview_layer",
    )
    providers_to_show = [
        provider
        for provider, visible in (
            ("Mapillary", show_mapillary),
            ("KartaView", show_kartaview),
        )
        if visible
    ]
    focus_location = None
    if "focus_latitude" in st.session_state and "focus_longitude" in st.session_state:
        focus_location = (
            float(st.session_state["focus_latitude"]),
            float(st.session_state["focus_longitude"]),
            float(st.session_state.get("radius_km", 5)),
        )
    show_source_tracks = bool(st.session_state.get("show_source_tracks", True))
    if cumulative_mode:
        st.caption(
            "Cumulative modes union unique road segments touched in any included "
            "year. Exact annual buffered-length metrics remain in Trends → Annual source metrics."
        )
    if not complete_comparison:
        st.caption("KartaView statewide collection is still in progress; its comparison will update after refresh.")

    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    render_covered_roads_map(
        providers_to_show,
        years_to_show,
        time_label,
        selected_classes,
        focus_location,
        show_source_tracks,
    )

    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.subheader("Zoom to a coordinate")
    with st.form("coordinate_search"):
        coordinate_col, radius_col, submit_col = st.columns([2.2, 1, 0.8])
        coordinate_value = coordinate_col.text_input(
            "Latitude, longitude",
            value=st.session_state.get("coordinate_value", ""),
            placeholder="47.6062, -122.3321",
            help="Enter decimal degrees in latitude, longitude order.",
        )
        radius_value = radius_col.select_slider(
            "Search radius",
            options=[1, 2, 5, 10, 20],
            value=int(st.session_state.get("radius_km", 5)),
            format_func=lambda value: f"{value} km",
        )
        submit_col.markdown("&nbsp;", unsafe_allow_html=True)
        zoom_submitted = submit_col.form_submit_button(
            "Zoom",
            use_container_width=True,
            type="primary",
        )
    if zoom_submitted:
        try:
            latitude, longitude = parse_coordinate(coordinate_value)
            st.session_state["coordinate_value"] = coordinate_value
            st.session_state["focus_latitude"] = latitude
            st.session_state["focus_longitude"] = longitude
            st.session_state["radius_km"] = radius_value
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    if focus_location is not None:
        location_col, clear_col = st.columns([4, 1])
        location_col.caption(
            f"Viewing {focus_location[2]:g} km around "
            f"{focus_location[0]:.5f}, {focus_location[1]:.5f}. "
            f"Showing {', '.join(providers_to_show) or 'no provider'} imagery "
            f"for {time_label}."
        )
        if clear_col.button("Statewide view", use_container_width=True):
            for key in ("focus_latitude", "focus_longitude"):
                st.session_state.pop(key, None)
            st.rerun()
        st.checkbox(
            "Show original provider tracks",
            value=True,
            key="show_source_tracks",
            help=(
                "Narrow provider-colored lines are source sequence geometry. "
                "Wide translucent lines are derived Overture-road coverage."
            ),
        )
        st.caption(
            "Map layers: light basemap roads · teal/purple derived road coverage · "
            "narrow provider-colored source tracks · white dot searched coordinate."
        )


def cumulative_view(
    cumulative: pd.DataFrame,
    by_class: pd.DataFrame,
    selected_classes: list[str],
) -> tuple[pd.DataFrame, dict[str, float], float]:
    denominator_by_class = (
        by_class.loc[~by_class["road_class"].eq("ALL")]
        .drop_duplicates("road_class")
        .set_index("road_class")["total_road_km"]
        .astype(float)
        .to_dict()
    )
    selected_denominator = sum(
        denominator_by_class.get(road_class, 0.0) for road_class in selected_classes
    )
    frame = cumulative[cumulative["road_class"].isin(selected_classes)].copy()
    frame = (
        frame.groupby(["provider", "cutoff_year"], as_index=False)
        .agg(
            touched_road_km=("touched_road_km", "sum"),
            segments_touched=("segments_touched", "sum"),
        )
    )
    frame["touched_pct"] = np.where(
        selected_denominator > 0,
        frame["touched_road_km"] / selected_denominator * 100.0,
        0.0,
    )
    return frame, denominator_by_class, selected_denominator


def render_trends_tab(
    cumulative: pd.DataFrame,
    statewide: pd.DataFrame,
    by_class: pd.DataFrame,
    available_years: list[int],
    selected_classes: list[str],
) -> None:
    if not selected_classes:
        st.info("Select at least one road class in the sidebar.")
        return
    trend, _, denominator = cumulative_view(cumulative, by_class, selected_classes)
    if trend.empty or denominator <= 0:
        st.info("Cumulative coverage is not available for the selected road classes.")
        return

    selected_cutoff = st.select_slider(
        "Coverage through year",
        options=available_years,
        value=max(available_years),
        key="trend_cutoff_year",
    )
    selected = trend[trend["cutoff_year"].eq(selected_cutoff)].set_index("provider")
    metric_cols = st.columns(2)
    for column, provider in zip(metric_cols, ("Mapillary", "KartaView")):
        if provider not in selected.index:
            column.metric(provider, "Pending")
            continue
        row = selected.loc[provider]
        column.metric(
            f"{provider} through {selected_cutoff} · {row['touched_pct']:.2f}%",
            f"{row['touched_road_km']:,.0f} km",
        )
    st.subheader("Cumulative coverage trend")
    st.altair_chart(cumulative_chart(trend), use_container_width=True)
    st.caption(
        "A road segment enters the cumulative total the first year that provider "
        "touches it. Repeated coverage in later years is counted once."
    )

    with st.expander("Annual source metrics"):
        st.caption(
            "These are the independently calculated annual buffered-length metrics, "
            "retained for audit and recency analysis. They are not cumulative."
        )
        detail = statewide[
            [
                "provider",
                "capture_year",
                "covered_road_km",
                "road_km_with_any_coverage",
                "segments_with_coverage",
                "coverage_pct",
            ]
        ].sort_values(["capture_year", "provider"], ascending=[False, True])
        st.dataframe(
            detail,
            hide_index=True,
            use_container_width=True,
            column_config={
                "provider": "Provider",
                "capture_year": st.column_config.NumberColumn("Year", format="%d"),
                "covered_road_km": st.column_config.NumberColumn(
                    "Exact covered km", format="%.0f"
                ),
                "road_km_with_any_coverage": st.column_config.NumberColumn(
                    "Road touched (km)", format="%.0f"
                ),
                "segments_with_coverage": st.column_config.NumberColumn(
                    "Segments", format=",%d"
                ),
                "coverage_pct": st.column_config.ProgressColumn(
                    "Exact annual coverage",
                    min_value=0.0,
                    max_value=100.0,
                    format="%.2f%%",
                ),
            },
        )
        st.download_button(
            "Download annual source metrics",
            detail.to_csv(index=False).encode("utf-8"),
            file_name="washington_imagery_annual_source_metrics.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_compare_tab(
    cumulative: pd.DataFrame,
    overlap: pd.DataFrame,
    by_class: pd.DataFrame,
    available_years: list[int],
    selected_classes: list[str],
    cutoff_year: int,
) -> None:
    if not selected_classes:
        st.info("Select at least one road class in the sidebar.")
        return
    _, denominator_by_class, denominator = cumulative_view(
        cumulative, by_class, selected_classes
    )
    provider_class = cumulative[
        cumulative["cutoff_year"].eq(cutoff_year)
        & cumulative["road_class"].isin(selected_classes)
    ].copy()
    provider_class["total_road_km"] = provider_class["road_class"].map(
        denominator_by_class
    )
    provider_class["touched_pct"] = np.where(
        provider_class["total_road_km"] > 0,
        provider_class["touched_road_km"] / provider_class["total_road_km"] * 100.0,
        0.0,
    )
    totals = (
        provider_class.groupby("provider", as_index=False)
        .agg(touched_road_km=("touched_road_km", "sum"))
    )
    totals["touched_pct"] = np.where(
        denominator > 0, totals["touched_road_km"] / denominator * 100.0, 0.0
    )
    totals = totals.set_index("provider")

    selected_overlap = overlap[overlap["road_class"].isin(selected_classes)].copy()
    mix = (
        selected_overlap.groupby("availability", as_index=False)
        .agg(road_km=("road_km", "sum"), segments=("segments", "sum"))
    )
    union_km = float(mix["road_km"].sum()) if not mix.empty else 0.0
    neither_km = max(0.0, denominator - union_km)
    mix = pd.concat(
        [
            mix,
            pd.DataFrame(
                [{"availability": "Neither", "road_km": neither_km, "segments": np.nan}]
            ),
        ],
        ignore_index=True,
    )
    order = {name: index for index, name in enumerate(
        ["Mapillary only", "Both", "KartaView only", "Neither"]
    )}
    mix["order"] = mix["availability"].map(order)
    mix["share_pct"] = np.where(
        denominator > 0, mix["road_km"] / denominator * 100.0, 0.0
    )
    shared = float(
        mix.loc[mix["availability"].eq("Both"), "road_km"].sum()
    )
    map_pct = float(totals.loc["Mapillary", "touched_pct"]) if "Mapillary" in totals.index else 0.0
    karta_pct = float(totals.loc["KartaView", "touched_pct"]) if "KartaView" in totals.index else 0.0

    metric_cols = st.columns(4)
    metric_cols[0].metric(
        f"Mapillary · {map_pct:.2f}%",
        f"{float(totals.loc['Mapillary', 'touched_road_km']):,.0f} km"
        if "Mapillary" in totals.index else "Pending",
    )
    metric_cols[1].metric(
        f"KartaView · {karta_pct:.2f}%",
        f"{float(totals.loc['KartaView', 'touched_road_km']):,.0f} km"
        if "KartaView" in totals.index else "Pending",
    )
    metric_cols[2].metric(
        "Shared network",
        f"{shared:,.0f} km",
    )
    lead = map_pct - karta_pct
    metric_cols[3].metric(
        f"{'Mapillary' if lead >= 0 else 'KartaView'} ahead",
        f"{abs(lead):.2f} points",
    )

    st.subheader("Provider availability across the selected road network")
    availability = mix.set_index("availability")
    availability_cols = st.columns(4)
    for column, category in zip(
        availability_cols,
        ("Mapillary only", "Both", "KartaView only", "Neither"),
    ):
        if category in availability.index:
            row = availability.loc[category]
            column.metric(category, f"{float(row['road_km']):,.0f} km")
            column.caption(f"{float(row['share_pct']):.2f}% of selected roads")
        else:
            column.metric(category, "0 km")
            column.caption("0.00% of selected roads")
    st.altair_chart(overlap_chart(mix), use_container_width=True)
    st.caption(
        "Teal: Mapillary only · charcoal: both · purple: KartaView only · gray: neither. "
        "Availability uses unique Overture segments of at most 50 metres; “both” "
        "means each provider touched the segment by the selected cutoff year."
    )
    st.subheader("Comparison by road class")
    if provider_class.empty:
        st.info("No cumulative coverage is available for these classes.")
    else:
        st.altair_chart(comparison_class_chart(provider_class), use_container_width=True)


def main() -> None:
    st.markdown('<div class="coverage-kicker">Washington State · 2020–2026</div>', unsafe_allow_html=True)
    st.markdown('<div class="coverage-title">Street-view coverage<br>through time</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="coverage-subtitle">
        Compare cumulative Mapillary and KartaView imagery availability against
        the same 240,157 km Overture road denominator. Trends count unique road
        segments touched through each year; exact annual metrics remain available
        for audit and recency analysis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not STATEWIDE_CSV.is_file() or not ROAD_CLASS_CSV.is_file():
        st.error("Annual coverage CSVs are not available yet.")
        live_collection_status()
        st.stop()

    statewide = load_csv(str(STATEWIDE_CSV), file_version(STATEWIDE_CSV))
    by_class = load_csv(str(ROAD_CLASS_CSV), file_version(ROAD_CLASS_CSV))
    providers = sorted(
        statewide["provider"].dropna().unique(),
        key=lambda p: list(PROVIDER_COLORS).index(p) if p in PROVIDER_COLORS else 99,
    )
    available_years = [
        int(year)
        for year in sorted(statewide["capture_year"].dropna().astype(int).unique())
    ]
    complete_comparison = set(PROVIDER_COLORS).issubset(providers)

    status = "Two-provider results ready" if complete_comparison else "KartaView collection in progress"
    st.markdown(
        f'<span class="status-chip">{status}</span>'
        f'<span class="status-chip">Data updated {latest_data_time()}</span>',
        unsafe_allow_html=True,
    )
    provider_legend(providers)

    with st.sidebar:
        st.header("View controls")
        road_options = [
            rc
            for rc in ROAD_CLASS_ORDER
            if rc in set(by_class.loc[~by_class["road_class"].eq("ALL"), "road_class"])
        ]
        selected_classes = st.multiselect(
            "Road classes",
            road_options,
            default=road_options,
        )
        if st.button("Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.caption(
            "Road classes filter the map, cumulative trends, and provider comparison. "
            "Refresh after source data changes."
        )

    if not CUMULATIVE_CSV.is_file() or not OVERLAP_CSV.is_file():
        st.error("The packaged cumulative comparison tables are missing.")
        st.stop()
    cumulative = load_cumulative_coverage(
        str(CUMULATIVE_CSV), file_version(CUMULATIVE_CSV)
    )

    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    explore_tab, trends_tab, compare_tab = st.tabs(
        ["Explore", "Trends", "Compare"]
    )
    with explore_tab:
        render_explore_tab(
            statewide,
            available_years,
            selected_classes,
            complete_comparison,
        )
    with trends_tab:
        render_trends_tab(
            cumulative,
            statewide,
            by_class,
            available_years,
            selected_classes,
        )
    with compare_tab:
        compare_year = st.select_slider(
            "Compare coverage through year",
            options=available_years,
            value=max(available_years),
            key="compare_cutoff_year",
        )
        overlap = load_cumulative_overlap(
            str(OVERLAP_CSV),
            file_version(OVERLAP_CSV),
            compare_year,
        )
        render_compare_tab(
            cumulative,
            overlap,
            by_class,
            available_years,
            selected_classes,
            compare_year,
        )

    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="method-note">
        <strong>How to read this dashboard.</strong> Trends and Compare count each
        canonical road segment once, beginning with the first year in which a
        provider touches it. The annual source table retains the exact
        per-year buffered-length calculation. Neither measure indicates image
        quality, field of view, or current accessibility. KartaView tracks are
        reconstructed from complete ordered sequence photos; gaps greater than
        100 metres are not joined.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
