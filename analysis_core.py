"""Core analysis for the Newfoundland tropical-cyclone recurvature study.

The publication workflow has four defining features:

1. Every track is placed on a regular 6-hour UTC grid.
2. Recurvature windows are defined in hours and require a west-to-east
   transition with continued poleward motion.
3. Proximity is measured from the post-turn track polyline to the
   Newfoundland island polygon.
4. Frequency, pathway rate, recurvature location, and proximity are each
   quantified with effect estimates and uncertainty.
"""

from __future__ import annotations

import json
import math
import os
import urllib.request
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy import stats
from scipy.spatial.distance import cdist, pdist
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import nearest_points, transform as shapely_transform
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
import xarray as xr


R_EARTH_KM = 6371.0
YEAR_MIN = 1950
YEAR_MAX = 2023
SATELLITE_START = 1979

IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r00/access/netcdf/IBTrACS.ALL.v04r00.nc"
)


@dataclass(frozen=True)
class DetectorConfig:
    grid_hours: int = 6
    window_hours: int = 24
    latitude_gate_deg_n: float = 25.0
    pre_east_max_kmh: float = 0.0
    post_east_min_kmh: float = 5.0
    post_north_min_kmh: float = 0.0
    min_eastward_acceleration_kmh: float = 5.0
    maximum_source_gap_hours: float = 12.0

    @property
    def window_steps(self) -> int:
        steps = self.window_hours / self.grid_hours
        if not float(steps).is_integer():
            raise ValueError("window_hours must be divisible by grid_hours")
        return int(steps)


@dataclass(frozen=True)
class RegionConfig:
    proximity_threshold_km: float = 600.0
    projection_epsg: int = 3347


BASELINE_DETECTOR = DetectorConfig()
BASELINE_REGION = RegionConfig()

LEGACY_NL_LAT_MIN = 44.0
LEGACY_NL_LAT_MAX = 54.5
LEGACY_NL_LON_MIN = -61.5
LEGACY_NL_LON_MAX = -50.0
LEGACY_PROXY_POINTS = np.array(
    [
        [47.56, -52.71],
        [48.95, -57.95],
        [48.70, -53.11],
        [49.18, -55.74],
        [51.45, -56.00],
    ],
    dtype=float,
)


def decode_text(value) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="ignore").strip(" \x00")
    if hasattr(value, "tobytes") and getattr(value, "dtype", None) is not None:
        if value.dtype.kind == "S":
            return value.tobytes().decode("utf-8", errors="ignore").strip(" \x00")
    return str(value).strip()


def wrap_longitude(lon):
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0


def download_if_needed(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, destination)
    return destination


def load_ibtracs(path: Path) -> xr.Dataset:
    return xr.open_dataset(path, decode_times=True)


def north_atlantic_indices(
    ds: xr.Dataset,
    year_min: int = YEAR_MIN,
    year_max: int = YEAR_MAX,
) -> np.ndarray:
    season = ds["season"].values.astype(int)
    basin = ds["basin"].values
    has_na = (basin == b"NA").any(axis=1)
    return np.where((season >= year_min) & (season <= year_max) & has_na)[0]


def tropical_origin_indices(
    ds: xr.Dataset,
    storm_indices: Iterable[int],
) -> np.ndarray:
    """Retain storms coded tropical (IBTrACS nature code TS) at least once."""

    if "nature" not in ds.variables:
        raise KeyError("IBTrACS nature is required to define tropical origin")
    retained = []
    for k in storm_indices:
        k = int(k)
        if any(decode_text(value) == "TS" for value in ds["nature"].values[k]):
            retained.append(k)
    return np.asarray(retained, dtype=int)


def _round_times_to_minute(values: np.ndarray) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(values)).round("min")


def _contiguous_source_segments(
    frame: pd.DataFrame,
    maximum_gap_hours: float,
) -> list[pd.DataFrame]:
    if frame.empty:
        return []
    gaps = frame["time"].diff().dt.total_seconds().div(3600.0)
    segment_id = (gaps > maximum_gap_hours).cumsum()
    return [g.copy() for _, g in frame.groupby(segment_id, sort=True)]


def resample_track_six_hourly(
    times: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    natures: np.ndarray | None = None,
    statuses: np.ndarray | None = None,
    grid_hours: int = 6,
    maximum_source_gap_hours: float = 12.0,
) -> pd.DataFrame:
    """Interpolate a track to a regular UTC grid without bridging long gaps."""

    valid = np.isfinite(lats) & np.isfinite(lons) & ~pd.isna(times)
    if valid.sum() < 2:
        return pd.DataFrame()

    frame = pd.DataFrame(
        {
            "time": _round_times_to_minute(times[valid]),
            "lat": np.asarray(lats, dtype=float)[valid],
            "lon": wrap_longitude(np.asarray(lons, dtype=float)[valid]),
        }
    )
    if natures is not None:
        frame["nature"] = [decode_text(v) for v in np.asarray(natures)[valid]]
    if statuses is not None:
        frame["status"] = [decode_text(v) for v in np.asarray(statuses)[valid]]

    frame = (
        frame.sort_values("time")
        .drop_duplicates("time", keep="first")
        .reset_index(drop=True)
    )
    if len(frame) < 2:
        return pd.DataFrame()

    pieces = []
    for source in _contiguous_source_segments(frame, maximum_source_gap_hours):
        if len(source) < 2:
            continue
        start = source["time"].iloc[0].ceil(f"{grid_hours}h")
        end = source["time"].iloc[-1].floor(f"{grid_hours}h")
        if start > end:
            continue
        grid = pd.date_range(start, end, freq=f"{grid_hours}h")
        if len(grid) < 2:
            continue

        source_seconds = source["time"].astype("int64").to_numpy(dtype=float) / 1e9
        grid_seconds = grid.astype("int64").to_numpy(dtype=float) / 1e9
        unwrapped_lon = np.degrees(
            np.unwrap(np.radians(source["lon"].to_numpy(dtype=float)))
        )

        out = pd.DataFrame(
            {
                "time": grid,
                "lat": np.interp(
                    grid_seconds, source_seconds, source["lat"].to_numpy(dtype=float)
                ),
                "lon_unwrapped": np.interp(
                    grid_seconds, source_seconds, unwrapped_lon
                ),
            }
        )
        out["lon"] = wrap_longitude(out["lon_unwrapped"])

        nearest = np.abs(
            source_seconds[:, None] - grid_seconds[None, :]
        ).argmin(axis=0)
        if "nature" in source:
            out["nature"] = source["nature"].to_numpy()[nearest]
        if "status" in source:
            out["status"] = source["status"].to_numpy()[nearest]
        pieces.append(out)

    if not pieces:
        return pd.DataFrame()

    out = (
        pd.concat(pieces, ignore_index=True)
        .sort_values("time")
        .drop_duplicates("time", keep="first")
        .reset_index(drop=True)
    )
    dt = out["time"].diff().dt.total_seconds().div(3600.0)
    out["segment"] = (dt > grid_hours * 1.01).cumsum()
    return out


def displacement_components_km(track: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    lat1 = np.radians(track["lat"].to_numpy(dtype=float)[:-1])
    lat2 = np.radians(track["lat"].to_numpy(dtype=float)[1:])
    lon_unwrapped = np.radians(track["lon_unwrapped"].to_numpy(dtype=float))
    dlon = np.diff(lon_unwrapped)
    dlat = lat2 - lat1
    mean_lat = 0.5 * (lat1 + lat2)
    dx = R_EARTH_KM * np.cos(mean_lat) * dlon
    dy = R_EARTH_KM * dlat
    return dx, dy


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    value = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * R_EARTH_KM * np.arcsin(np.sqrt(value))


def detect_legacy_increment_recurvature(
    lat: np.ndarray,
    lon: np.ndarray,
    latitude_gate_deg_n: float = 25.0,
    window_steps: int = 4,
    post_zonal_km_per_increment: float = 30.0,
    pre_zonal_max_km_per_increment: float = 30.0,
) -> dict | None:
    """Recovered increment-based detector retained only for comparison."""

    if len(lat) < 2 * window_steps + 3:
        return None
    lon_unwrapped = np.degrees(np.unwrap(np.radians(lon)))
    dlon = np.radians(np.diff(lon_unwrapped))
    mean_lat = 0.5 * (lat[:-1] + lat[1:])
    zonal_km = R_EARTH_KM * np.cos(np.radians(mean_lat)) * dlon
    for j in range(window_steps, len(zonal_km) - window_steps):
        i_star = j + 1
        if lat[i_star] < latitude_gate_deg_n:
            continue
        pre = float(np.mean(zonal_km[j - window_steps : j]))
        post = float(np.mean(zonal_km[j : j + window_steps]))
        if (
            pre <= pre_zonal_max_km_per_increment
            and post > post_zonal_km_per_increment
        ):
            return {
                "recurv_index": int(i_star),
                "recurv_lat": float(lat[i_star]),
                "recurv_lon": float(lon[i_star]),
                "legacy_pre_zonal_km_per_increment": pre,
                "legacy_post_zonal_km_per_increment": post,
            }
    return None


def classify_legacy_method(
    ds: xr.Dataset,
    storm_indices: Iterable[int],
    distance_threshold_km: float = 600.0,
    minimum_track_points: int = 25,
) -> pd.DataFrame:
    """Apply the diagnostic native-cadence workflow comparator."""

    rows = []
    for k in storm_indices:
        k = int(k)
        lat_raw = ds["lat"].values[k].astype(float)
        lon_raw = wrap_longitude(ds["lon"].values[k].astype(float))
        time_raw = ds["time"].values[k]
        valid = np.isfinite(lat_raw) & np.isfinite(lon_raw)
        lat = lat_raw[valid]
        lon = lon_raw[valid]
        original_indices = np.flatnonzero(valid)
        if len(lat) < minimum_track_points:
            continue
        result = detect_legacy_increment_recurvature(lat, lon)
        if result is None:
            continue
        i_star = result["recurv_index"]
        post_lat = lat[i_star:]
        post_lon = lon[i_star:]
        box_hit = bool(
            np.any(
                (post_lat >= LEGACY_NL_LAT_MIN)
                & (post_lat <= LEGACY_NL_LAT_MAX)
                & (post_lon >= LEGACY_NL_LON_MIN)
                & (post_lon <= LEGACY_NL_LON_MAX)
            )
        )
        min_distance = min(
            float(np.nanmin(haversine_km(post_lat, post_lon, point[0], point[1])))
            for point in LEGACY_PROXY_POINTS
        )
        relevant = box_hit or min_distance <= distance_threshold_km
        if not relevant:
            continue
        original_index = int(original_indices[i_star])
        rows.append(
            {
                "dataset_index": k,
                "sid": decode_text(ds["sid"].values[k]),
                "season": int(ds["season"].values[k]),
                "name": decode_text(ds["name"].values[k]),
                "recurv_time": pd.Timestamp(time_raw[original_index]).round("min"),
                "box_hit": box_hit,
                "minimum_proxy_distance_km": min_distance,
                **result,
            }
        )
    return pd.DataFrame(rows).sort_values(["season", "sid"]).reset_index(drop=True)


def detect_recurvature_velocity(
    track: pd.DataFrame,
    config: DetectorConfig = BASELINE_DETECTOR,
) -> dict | None:
    """Detect the first sustained west-to-east, poleward trajectory turn."""

    win = config.window_steps
    if len(track) < 2 * win + 1:
        return None

    dx, dy = displacement_components_km(track)
    dt_hours = float(config.grid_hours)
    u = dx / dt_hours
    v = dy / dt_hours
    segments = track["segment"].to_numpy(dtype=int)

    for boundary in range(win, len(track) - win):
        # The candidate position is the first point in the post-turn window.
        if track["lat"].iloc[boundary] < config.latitude_gate_deg_n:
            continue
        if segments[boundary - win] != segments[boundary + win]:
            continue

        pre_u = float(np.mean(u[boundary - win : boundary]))
        post_u = float(np.mean(u[boundary : boundary + win]))
        post_v = float(np.mean(v[boundary : boundary + win]))
        delta_u = post_u - pre_u

        if (
            pre_u <= config.pre_east_max_kmh
            and post_u >= config.post_east_min_kmh
            and post_v > config.post_north_min_kmh
            and delta_u >= config.min_eastward_acceleration_kmh
        ):
            row = track.iloc[boundary]
            return {
                "recurv_index": int(boundary),
                "recurv_time": pd.Timestamp(row["time"]),
                "recurv_lat": float(row["lat"]),
                "recurv_lon": float(row["lon"]),
                "pre_u_kmh": pre_u,
                "post_u_kmh": post_u,
                "post_v_kmh": post_v,
                "delta_u_kmh": delta_u,
                "nature_at_recurvature": str(row.get("nature", "")),
                "status_at_recurvature": str(row.get("status", "")),
            }
    return None


def detect_recurvature_heading(
    track: pd.DataFrame,
    config: DetectorConfig = BASELINE_DETECTOR,
    minimum_heading_change_deg: float = 30.0,
) -> dict | None:
    """Alternative detector based on local-displacement headings."""

    win = config.window_steps
    if len(track) < 2 * win + 1:
        return None
    dx, dy = displacement_components_km(track)
    segments = track["segment"].to_numpy(dtype=int)
    angles = np.degrees(np.arctan2(dx, dy)) % 360.0
    east_component = np.sin(np.radians(angles))
    north_component = np.cos(np.radians(angles))

    for boundary in range(win, len(track) - win):
        if track["lat"].iloc[boundary] < config.latitude_gate_deg_n:
            continue
        if segments[boundary - win] != segments[boundary + win]:
            continue

        pre_e = float(np.mean(east_component[boundary - win : boundary]))
        post_e = float(np.mean(east_component[boundary : boundary + win]))
        post_n = float(np.mean(north_component[boundary : boundary + win]))
        pre_angle = math.degrees(
            math.atan2(
                float(np.mean(dx[boundary - win : boundary])),
                float(np.mean(dy[boundary - win : boundary])),
            )
        ) % 360.0
        post_angle = math.degrees(
            math.atan2(
                float(np.mean(dx[boundary : boundary + win])),
                float(np.mean(dy[boundary : boundary + win])),
            )
        ) % 360.0
        signed_change = ((post_angle - pre_angle + 180.0) % 360.0) - 180.0

        if (
            pre_e <= 0.0
            and post_e > 0.10
            and post_n > 0.0
            and signed_change >= minimum_heading_change_deg
        ):
            row = track.iloc[boundary]
            return {
                "recurv_index": int(boundary),
                "recurv_time": pd.Timestamp(row["time"]),
                "recurv_lat": float(row["lat"]),
                "recurv_lon": float(row["lon"]),
                "pre_heading_deg": pre_angle,
                "post_heading_deg": post_angle,
                "heading_change_deg": signed_change,
                "nature_at_recurvature": str(row.get("nature", "")),
                "status_at_recurvature": str(row.get("status", "")),
            }
    return None


def has_detection_opportunity(
    track: pd.DataFrame,
    config: DetectorConfig = BASELINE_DETECTOR,
) -> bool:
    """Whether a track contains a complete fixed-duration window poleward of the gate."""

    win = config.window_steps
    if len(track) < 2 * win + 1:
        return False
    segments = track["segment"].to_numpy(dtype=int)
    latitude = track["lat"].to_numpy(dtype=float)
    for boundary in range(win, len(track) - win):
        if (
            latitude[boundary] >= config.latitude_gate_deg_n
            and segments[boundary - win] == segments[boundary + win]
        ):
            return True
    return False


def load_newfoundland_island_polygon(
    cartopy_data_dir: Path,
) -> tuple[Polygon | MultiPolygon, Polygon | MultiPolygon]:
    """Return Newfoundland island in lon/lat and EPSG:3347 coordinates."""

    cartopy_data_dir.mkdir(parents=True, exist_ok=True)
    cartopy.config["data_dir"] = str(cartopy_data_dir)
    shp = shapereader.natural_earth(
        resolution="10m",
        category="cultural",
        name="admin_1_states_provinces",
    )
    province = None
    for record in shapereader.Reader(shp).records():
        attr = record.attributes
        if attr.get("admin") == "Canada" and attr.get("postal") == "NL":
            province = record.geometry
            break
    if province is None:
        raise RuntimeError("Newfoundland and Labrador polygon was not found")

    components = list(province.geoms) if isinstance(province, MultiPolygon) else [province]
    island_components = [
        geom
        for geom in components
        if geom.bounds[1] < 52.0 and geom.bounds[3] < 52.5
    ]
    if not island_components:
        raise RuntimeError("Newfoundland island component was not identified")
    island_lonlat = (
        island_components[0]
        if len(island_components) == 1
        else MultiPolygon(island_components)
    )
    project = projected_transformer(3347).transform
    island_projected = shapely_transform(project, island_lonlat)
    return island_lonlat, island_projected


@lru_cache(maxsize=8)
def projected_transformer(projection_epsg: int) -> Transformer:
    return Transformer.from_crs(
        "EPSG:4326", f"EPSG:{projection_epsg}", always_xy=True
    )


def post_track_distance_to_island_km(
    track: pd.DataFrame,
    recurv_index: int,
    island_projected,
    projection_epsg: int = 3347,
) -> float:
    return post_track_distance_diagnostics(
        track,
        recurv_index,
        island_projected,
        projection_epsg=projection_epsg,
    )["minimum_distance_to_newfoundland_km"]


def post_track_distance_diagnostics(
    track: pd.DataFrame,
    recurv_index: int,
    island_projected,
    projection_epsg: int = 3347,
    endpoint_tolerance_km: float = 1.0,
) -> dict:
    """Return projected distance and potential end-of-track censoring diagnostics."""

    transformer = projected_transformer(projection_epsg)
    return post_track_distance_diagnostics_with_transformer(
        track,
        recurv_index,
        island_projected,
        transformer,
        endpoint_tolerance_km=endpoint_tolerance_km,
    )


def post_track_distance_diagnostics_with_transformer(
    track: pd.DataFrame,
    recurv_index: int,
    island_projected,
    transformer: Transformer,
    endpoint_tolerance_km: float = 1.0,
) -> dict:
    """Distance diagnostics using a supplied lon/lat-to-projected transformer."""

    post = track.iloc[recurv_index:]
    x, y = transformer.transform(
        post["lon"].to_numpy(dtype=float),
        post["lat"].to_numpy(dtype=float),
    )
    if len(x) == 1:
        geometry = Point(float(x[0]), float(y[0]))
    else:
        geometry = LineString(np.column_stack([x, y]))
    distance_km = float(geometry.distance(island_projected) / 1000.0)

    if isinstance(geometry, LineString) and geometry.length > 0:
        nearest_on_track, _ = nearest_points(geometry, island_projected)
        along_track_m = float(geometry.project(nearest_on_track))
        closest_fraction = along_track_m / float(geometry.length)
        closest_at_endpoint = bool(
            float(geometry.length) - along_track_m
            <= endpoint_tolerance_km * 1000.0
        )
    else:
        closest_fraction = 1.0
        closest_at_endpoint = True

    duration_hours = float(
        (
            pd.Timestamp(post["time"].iloc[-1])
            - pd.Timestamp(post["time"].iloc[0])
        ).total_seconds()
        / 3600.0
    )
    return {
        "minimum_distance_to_newfoundland_km": distance_km,
        "closest_fraction_along_post_track": closest_fraction,
        "closest_at_track_endpoint": closest_at_endpoint,
        "post_recurvature_duration_hours": duration_hours,
    }


def _storm_track_from_dataset(
    ds: xr.Dataset,
    k: int,
    config: DetectorConfig,
) -> pd.DataFrame:
    natures = ds["nature"].values[k] if "nature" in ds.variables else None
    statuses = ds["usa_status"].values[k] if "usa_status" in ds.variables else None
    return resample_track_six_hourly(
        ds["time"].values[k],
        ds["lat"].values[k],
        ds["lon"].values[k],
        natures=natures,
        statuses=statuses,
        grid_hours=config.grid_hours,
        maximum_source_gap_hours=config.maximum_source_gap_hours,
    )


def build_track_cache(
    ds: xr.Dataset,
    storm_indices: Iterable[int],
    config: DetectorConfig = BASELINE_DETECTOR,
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    cache: dict[int, pd.DataFrame] = {}
    audit_rows = []
    for k in storm_indices:
        track = _storm_track_from_dataset(ds, int(k), config)
        cache[int(k)] = track
        source_valid = (
            np.isfinite(ds["lat"].values[k])
            & np.isfinite(ds["lon"].values[k])
            & ~pd.isna(ds["time"].values[k])
        )
        source_times = _round_times_to_minute(ds["time"].values[k][source_valid])
        source_dt = (
            pd.Series(source_times)
            .diff()
            .dt.total_seconds()
            .div(3600.0)
            .dropna()
        )
        eligible = has_detection_opportunity(track, config)
        audit_rows.append(
            {
                "dataset_index": int(k),
                "source_points": int(source_valid.sum()),
                "resampled_points": int(len(track)),
                "source_min_dt_h": float(source_dt.min()) if len(source_dt) else np.nan,
                "source_median_dt_h": float(source_dt.median()) if len(source_dt) else np.nan,
                "source_max_dt_h": float(source_dt.max()) if len(source_dt) else np.nan,
                "eligible": bool(eligible),
            }
        )
    return cache, pd.DataFrame(audit_rows)


def classify_recurving_storms(
    ds: xr.Dataset,
    storm_indices: Iterable[int],
    track_cache: dict[int, pd.DataFrame],
    island_projected,
    detector: str = "velocity",
    detector_config: DetectorConfig = BASELINE_DETECTOR,
    region_config: RegionConfig = BASELINE_REGION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    eligibility = []
    season_values = ds["season"].values.astype(int)
    for k in storm_indices:
        k = int(k)
        track = track_cache[k]
        eligible = has_detection_opportunity(track, detector_config)
        eligibility.append(
            {
                "dataset_index": k,
                "sid": decode_text(ds["sid"].values[k]),
                "season": int(season_values[k]),
                "name": decode_text(ds["name"].values[k]),
                "eligible": bool(eligible),
            }
        )
        if not eligible:
            continue

        if detector == "velocity":
            result = detect_recurvature_velocity(track, detector_config)
        elif detector == "heading":
            result = detect_recurvature_heading(track, detector_config)
        else:
            raise ValueError("detector must be 'velocity' or 'heading'")
        if result is None:
            continue

        distance_diagnostics = post_track_distance_diagnostics(
            track,
            result["recurv_index"],
            island_projected,
            projection_epsg=region_config.projection_epsg,
        )
        distance_km = distance_diagnostics[
            "minimum_distance_to_newfoundland_km"
        ]
        row = {
            "dataset_index": k,
            "sid": decode_text(ds["sid"].values[k]),
            "season": int(season_values[k]),
            "name": decode_text(ds["name"].values[k]),
            "detector": detector,
            "newfoundland_relevant": bool(
                distance_km <= region_config.proximity_threshold_km
            ),
            **distance_diagnostics,
            **result,
        }
        rows.append(row)

    recurvers = pd.DataFrame(rows)
    if len(recurvers):
        recurvers = recurvers.sort_values(["season", "sid"]).reset_index(drop=True)
    eligible = pd.DataFrame(eligibility).sort_values(["season", "sid"]).reset_index(drop=True)
    return recurvers, eligible


def annual_counts(
    events: pd.DataFrame,
    year_min: int = YEAR_MIN,
    year_max: int = YEAR_MAX,
) -> pd.DataFrame:
    years = pd.Index(range(year_min, year_max + 1), name="season")
    values = events["season"].value_counts() if len(events) else pd.Series(dtype=int)
    return values.reindex(years, fill_value=0).rename("count").reset_index()


def poisson_trend(
    counts: pd.DataFrame,
    count_col: str = "count",
    offset: np.ndarray | None = None,
    hac_lags: int = 2,
) -> tuple[dict, sm.GLM]:
    year = counts["season"].to_numpy(dtype=float)
    response = counts[count_col].to_numpy(dtype=float)
    centered = year - year.mean()
    design = sm.add_constant(centered)
    model = sm.GLM(
        response,
        design,
        family=sm.families.Poisson(),
        offset=offset,
    )
    result = model.fit()
    robust = model.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    beta = float(result.params[1])
    se = float(result.bse[1])
    robust_se = float(robust.bse[1])
    pearson_dispersion = float(result.pearson_chi2 / result.df_resid)
    prediction = result.get_prediction(design, offset=offset).summary_frame()
    residuals = np.asarray(result.resid_pearson, dtype=float)
    lag1 = float(pd.Series(residuals).autocorr(lag=1))
    lb = acorr_ljungbox(residuals, lags=[min(5, max(1, len(residuals) // 8))])

    summary = {
        "n_years": int(len(counts)),
        "total_events": int(response.sum()),
        "irr_per_decade": float(np.exp(beta * 10.0)),
        "ci_low": float(np.exp((beta - 1.96 * se) * 10.0)),
        "ci_high": float(np.exp((beta + 1.96 * se) * 10.0)),
        "p_value": float(result.pvalues[1]),
        "hac_ci_low": float(np.exp((beta - 1.96 * robust_se) * 10.0)),
        "hac_ci_high": float(np.exp((beta + 1.96 * robust_se) * 10.0)),
        "hac_p_value": float(robust.pvalues[1]),
        "pearson_dispersion": pearson_dispersion,
        "lag1_residual_autocorrelation": lag1,
        "ljung_box_p_value": float(lb["lb_pvalue"].iloc[0]),
    }
    fitted = counts[["season", count_col]].copy()
    fitted["fitted"] = prediction["mean"].to_numpy()
    fitted["ci_low"] = prediction["mean_ci_lower"].to_numpy()
    fitted["ci_high"] = prediction["mean_ci_upper"].to_numpy()
    return summary, fitted


def binomial_pathway_trend(
    annual_frame: pd.DataFrame,
    successes: str,
    trials: str,
) -> dict:
    valid = annual_frame[trials] > 0
    frame = annual_frame.loc[valid].copy()
    year = frame["season"].to_numpy(dtype=float)
    centered = year - year.mean()
    design = sm.add_constant(centered)
    proportion = frame[successes].to_numpy(dtype=float) / frame[trials].to_numpy(dtype=float)
    model = sm.GLM(
        proportion,
        design,
        family=sm.families.Binomial(),
        freq_weights=frame[trials].to_numpy(dtype=float),
    )
    result = model.fit()
    beta = float(result.params[1])
    se = float(result.bse[1])
    return {
        "n_years": int(len(frame)),
        "total_successes": int(frame[successes].sum()),
        "total_trials": int(frame[trials].sum()),
        "odds_ratio_per_decade": float(np.exp(beta * 10.0)),
        "ci_low": float(np.exp((beta - 1.96 * se) * 10.0)),
        "ci_high": float(np.exp((beta + 1.96 * se) * 10.0)),
        "p_value": float(result.pvalues[1]),
    }


def robust_linear_trend(
    frame: pd.DataFrame,
    response: str,
    scale_per_decade: float = 10.0,
) -> dict:
    clean = frame[["season", response]].dropna()
    year = clean["season"].to_numpy(dtype=float)
    centered = year - year.mean()
    design = sm.add_constant(centered)
    result = sm.OLS(clean[response].to_numpy(dtype=float), design).fit(cov_type="HC3")
    slope = float(result.params[1]) * scale_per_decade
    se = float(result.bse[1]) * scale_per_decade
    return {
        "n": int(len(clean)),
        "slope_per_decade": slope,
        "ci_low": slope - 1.96 * se,
        "ci_high": slope + 1.96 * se,
        "p_value": float(result.pvalues[1]),
    }


def quantile_distance_trend(frame: pd.DataFrame, response: str) -> dict:
    clean = frame[["season", response]].dropna()
    year = clean["season"].to_numpy(dtype=float)
    centered = year - year.mean()
    design = sm.add_constant(centered)
    result = sm.QuantReg(clean[response].to_numpy(dtype=float), design).fit(q=0.5)
    slope = float(result.params[1]) * 10.0
    se = float(result.bse[1]) * 10.0
    return {
        "n": int(len(clean)),
        "median_slope_km_per_decade": slope,
        "ci_low": slope - 1.96 * se,
        "ci_high": slope + 1.96 * se,
        "p_value": float(result.pvalues[1]),
    }


def energy_distance_test(
    first: np.ndarray,
    second: np.ndarray,
    permutations: int = 4999,
    seed: int = 42,
) -> dict:
    """Two-sample multivariate energy-distance permutation test."""

    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    n, m = len(first), len(second)
    pooled = np.vstack([first, second])

    def statistic(a, b):
        cross = 2.0 * cdist(a, b).mean()
        within_a = 0.0 if len(a) < 2 else 2.0 * pdist(a).sum() / (len(a) ** 2)
        within_b = 0.0 if len(b) < 2 else 2.0 * pdist(b).sum() / (len(b) ** 2)
        return cross - within_a - within_b

    observed = float(statistic(first, second))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        permutation = rng.permutation(n + m)
        value = statistic(pooled[permutation[:n]], pooled[permutation[n:]])
        exceed += value >= observed
    return {
        "n_first": int(n),
        "n_second": int(m),
        "energy_statistic": observed,
        "permutation_p_value": float((exceed + 1) / (permutations + 1)),
        "permutations": int(permutations),
    }


def lonlat_to_cartesian_km(
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
) -> np.ndarray:
    """Map lon/lat to three-dimensional Earth-centred coordinates in kilometres."""

    longitude = np.radians(np.asarray(longitude_deg, dtype=float))
    latitude = np.radians(np.asarray(latitude_deg, dtype=float))
    cos_latitude = np.cos(latitude)
    return R_EARTH_KM * np.column_stack(
        [
            cos_latitude * np.cos(longitude),
            cos_latitude * np.sin(longitude),
            np.sin(latitude),
        ]
    )


def seasonal_counts(events: pd.DataFrame) -> pd.DataFrame:
    month = pd.to_datetime(events["recurv_time"]).dt.month
    counts = month.value_counts().reindex(range(1, 13), fill_value=0).sort_index()
    frame = pd.DataFrame(
        {
            "month": counts.index,
            "month_name": [
                pd.Timestamp(2000, value, 1).strftime("%B") for value in counts.index
            ],
            "events": counts.to_numpy(dtype=int),
        }
    )
    frame["percent"] = 100.0 * frame["events"] / frame["events"].sum()
    return frame


def status_summary(events: pd.DataFrame) -> pd.DataFrame:
    values = (
        events["nature_at_recurvature"]
        .replace("", "unknown")
        .fillna("unknown")
        .value_counts()
    )
    return values.rename_axis("nature_code").rename("events").reset_index()


def annual_analysis_frame(
    relevant: pd.DataFrame,
    recurvers: pd.DataFrame,
    eligible: pd.DataFrame,
    year_min: int = YEAR_MIN,
    year_max: int = YEAR_MAX,
) -> pd.DataFrame:
    frame = pd.DataFrame({"season": np.arange(year_min, year_max + 1)})
    for label, source in (
        ("relevant_events", relevant),
        ("recurving_storms", recurvers),
        ("eligible_storms", eligible[eligible["eligible"]]),
    ):
        values = source["season"].value_counts()
        frame[label] = frame["season"].map(values).fillna(0).astype(int)
    frame["relevant_fraction_of_eligible"] = np.where(
        frame["eligible_storms"] > 0,
        frame["relevant_events"] / frame["eligible_storms"],
        np.nan,
    )
    frame["relevant_fraction_of_recurvers"] = np.where(
        frame["recurving_storms"] > 0,
        frame["relevant_events"] / frame["recurving_storms"],
        np.nan,
    )
    return frame


def classification_overlap(first: pd.DataFrame, second: pd.DataFrame) -> dict:
    a = set(first["sid"])
    b = set(second["sid"])
    union = a | b
    return {
        "first_n": len(a),
        "second_n": len(b),
        "overlap": len(a & b),
        "only_first": len(a - b),
        "only_second": len(b - a),
        "jaccard": float(len(a & b) / len(union)) if union else np.nan,
    }


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str))


def config_dict(
    detector: DetectorConfig = BASELINE_DETECTOR,
    region: RegionConfig = BASELINE_REGION,
) -> dict:
    return {"detector": asdict(detector), "region": asdict(region)}


def suppress_expected_warnings() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
