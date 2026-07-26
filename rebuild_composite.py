"""Rebuild the illustrative satellite-era 500-hPa composite."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import urlopen

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import RegularGridInterpolator
import xarray as xr

from analysis_core import (
    SATELLITE_START,
    YEAR_MAX,
    load_newfoundland_island_polygon,
    write_json,
)


ROOT = Path(os.environ.get("NL_RECURV_WORKDIR", Path.cwd())).resolve()
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = ROOT / "figures"
CACHE_DIR = Path(
    os.environ.get("NCEP_EVENT_CACHE", ROOT / "data" / "ncep_event_fields")
).resolve()
CARTOPY_DIR = Path(
    os.environ.get("CARTOPY_DATA_DIR", ROOT / "data" / "cartopy")
).resolve()
CLIMATOLOGY_PATH = Path(
    os.environ.get(
        "NCEP_HGT_CLIMATOLOGY",
        ROOT / "data" / "hgt.mon.ltm.1991-2020.nc",
    )
).resolve()

EVENTS_PATH = OUTPUT_DIR / "corrected_newfoundland_relevant_events.csv"
CLIMATOLOGY_URL = (
    "https://downloads.psl.noaa.gov/Datasets/"
    "ncep.reanalysis.derived/pressure/hgt.mon.ltm.1991-2020.nc"
)
NCSS_TEMPLATE = (
    "https://psl.noaa.gov/thredds/ncss/grid/Datasets/"
    "ncep.reanalysis/pressure/hgt.{year}.nc"
)

LAT_NORTH = 80.0
LAT_SOUTH = 5.0
LON_WEST = 220.0
LON_EAST = 360.0
LEVEL_HPA = 500.0

BLUE = "#1769AA"
RED = "#B42318"


def download_file(url: str, destination: Path, attempts: int = 3) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 1000:
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(url, timeout=90) as response, temporary.open("wb") as stream:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
            temporary.replace(destination)
            return destination
        except Exception:
            if temporary.exists():
                temporary.unlink()
            if attempt == attempts:
                raise
            time.sleep(attempt * 2)
    raise RuntimeError("unreachable")


def event_url(row: pd.Series) -> str:
    event_time = pd.Timestamp(row["recurv_time"])
    query = urlencode(
        {
            "var": "hgt",
            "north": LAT_NORTH,
            "west": LON_WEST,
            "east": LON_EAST,
            "south": LAT_SOUTH,
            "horizStride": 1,
            "time": event_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "vertCoord": LEVEL_HPA,
            "accept": "netcdf4",
        }
    )
    return f"{NCSS_TEMPLATE.format(year=event_time.year)}?{query}"


def fetch_event(row: pd.Series) -> Path:
    destination = CACHE_DIR / f"{row['sid']}.nc"
    return download_file(event_url(row), destination)


def false_discovery_rate_mask(p_values: np.ndarray, q: float = 0.05) -> np.ndarray:
    flat = np.asarray(p_values, dtype=float).ravel()
    valid = np.isfinite(flat)
    ordered = np.argsort(flat[valid])
    sorted_p = flat[valid][ordered]
    thresholds = q * np.arange(1, len(sorted_p) + 1) / len(sorted_p)
    passing = np.flatnonzero(sorted_p <= thresholds)
    mask = np.zeros_like(flat, dtype=bool)
    if len(passing):
        cutoff = sorted_p[passing[-1]]
        mask = np.isfinite(flat) & (flat <= cutoff)
    return mask.reshape(p_values.shape)


def add_map_ticks(ax, extent):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#F1EFE8")
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#EAF2F8")
    ax.coastlines("50m", linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.3)
    ax.set_xticks(np.arange(-120, 1, 20), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(20, 81, 10), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.4)


if not EVENTS_PATH.exists():
    raise FileNotFoundError(
        "Run run_analysis.py before rebuilding the composite"
    )

CACHE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CARTOPY_DIR.mkdir(parents=True, exist_ok=True)
cartopy.config["data_dir"] = str(CARTOPY_DIR)

events = pd.read_csv(EVENTS_PATH, parse_dates=["recurv_time"])
events = events[events["season"] >= SATELLITE_START].copy().reset_index(drop=True)
print(f"Baseline satellite-era events: {len(events)}")

if not CLIMATOLOGY_PATH.exists():
    print("Downloading 1991-2020 monthly climatology")
    download_file(CLIMATOLOGY_URL, CLIMATOLOGY_PATH)

print("Downloading/caching event-time fields")
errors = []
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {
        executor.submit(fetch_event, row): row["sid"]
        for _, row in events.iterrows()
    }
    for completed, future in enumerate(as_completed(futures), start=1):
        sid = futures[future]
        try:
            future.result()
        except Exception as exc:
            errors.append({"sid": sid, "error": str(exc)})
        if completed % 10 == 0 or completed == len(futures):
            print(f"  {completed}/{len(futures)} fields resolved")

if errors:
    raise RuntimeError(f"Failed event downloads: {errors}")

climatology = xr.open_dataset(CLIMATOLOGY_PATH, decode_times=False)["hgt"].sel(
    level=LEVEL_HPA,
    lat=slice(LAT_NORTH, LAT_SOUTH),
    lon=slice(LON_WEST, 357.5),
)

anomalies = []
absolute_fields = []
event_metadata = []
for _, row in events.iterrows():
    event_path = CACHE_DIR / f"{row['sid']}.nc"
    event_ds = xr.open_dataset(event_path)
    field = (
        event_ds["hgt"]
        .sel(level=LEVEL_HPA)
        .squeeze(drop=True)
        .sel(lon=slice(LON_WEST, 357.5))
        .load()
    )
    month_index = int(pd.Timestamp(row["recurv_time"]).month - 1)
    monthly_normal = climatology.isel(time=month_index).load()
    monthly_normal = monthly_normal.sel(lat=field["lat"], lon=field["lon"])
    anomaly = field - monthly_normal
    anomalies.append(np.asarray(anomaly.values, dtype=float))
    absolute_fields.append(np.asarray(field.values, dtype=float))
    event_metadata.append(
        {
            "sid": row["sid"],
            "event_time": str(pd.Timestamp(row["recurv_time"])),
            "recurv_lat": float(row["recurv_lat"]),
            "recurv_lon": float(row["recurv_lon"]),
            "source_file": str(event_path.name),
        }
    )
    event_ds.close()

anomaly_stack = np.stack(anomalies)
absolute_stack = np.stack(absolute_fields)
latitude = np.asarray(field["lat"].values, dtype=float)
longitude_360 = np.asarray(field["lon"].values, dtype=float)
longitude = np.where(longitude_360 > 180.0, longitude_360 - 360.0, longitude_360)

geographic_mean = np.nanmean(anomaly_stack, axis=0)
absolute_mean = np.nanmean(absolute_stack, axis=0)
test = stats.ttest_1samp(anomaly_stack, popmean=0.0, axis=0, nan_policy="omit")
geographic_fdr = false_discovery_rate_mask(test.pvalue, q=0.05)


relative_lon = np.arange(-30.0, 30.1, 2.5)
relative_lat = np.arange(-20.0, 20.1, 2.5)
relative_mesh_lon, relative_mesh_lat = np.meshgrid(relative_lon, relative_lat)
relative_anomalies = []
relative_absolute = []

ascending_lat = latitude[::-1]
for event_index, row in events.iterrows():
    event_lon = float(row["recurv_lon"]) % 360.0
    event_lat = float(row["recurv_lat"])
    query = np.column_stack(
        [
            (event_lat + relative_mesh_lat).ravel(),
            (event_lon + relative_mesh_lon).ravel(),
        ]
    )
    anomaly_interpolator = RegularGridInterpolator(
        (ascending_lat, longitude_360),
        anomaly_stack[event_index][::-1, :],
        bounds_error=False,
        fill_value=np.nan,
    )
    absolute_interpolator = RegularGridInterpolator(
        (ascending_lat, longitude_360),
        absolute_stack[event_index][::-1, :],
        bounds_error=False,
        fill_value=np.nan,
    )
    relative_anomalies.append(
        anomaly_interpolator(query).reshape(relative_mesh_lat.shape)
    )
    relative_absolute.append(
        absolute_interpolator(query).reshape(relative_mesh_lat.shape)
    )

relative_anomaly_stack = np.stack(relative_anomalies)
relative_absolute_stack = np.stack(relative_absolute)
relative_mean = np.nanmean(relative_anomaly_stack, axis=0)
relative_absolute_mean = np.nanmean(relative_absolute_stack, axis=0)
relative_test = stats.ttest_1samp(
    relative_anomaly_stack, popmean=0.0, axis=0, nan_policy="omit"
)
relative_fdr = false_discovery_rate_mask(relative_test.pvalue, q=0.05)


levels = np.arange(-60, 61, 10)
absolute_levels = np.arange(5400, 5941, 60)
fig = plt.figure(figsize=(11.2, 5.6))

ax1 = fig.add_subplot(121, projection=ccrs.PlateCarree())
add_map_ticks(ax1, (-110, 0, 15, 75))
filled = ax1.contourf(
    longitude,
    latitude,
    geographic_mean,
    levels=levels,
    cmap="RdBu_r",
    extend="both",
    transform=ccrs.PlateCarree(),
)
height_contours = ax1.contour(
    longitude,
    latitude,
    absolute_mean,
    levels=absolute_levels,
    colors="#333333",
    linewidths=0.55,
    transform=ccrs.PlateCarree(),
)
ax1.clabel(height_contours, fmt="%d", fontsize=6, inline=True)
stipple_y, stipple_x = np.where(geographic_fdr)
keep = (stipple_y % 2 == 0) & (stipple_x % 2 == 0)
ax1.scatter(
    longitude[stipple_x[keep]],
    latitude[stipple_y[keep]],
    s=2.5,
    color="#222222",
    alpha=0.65,
    transform=ccrs.PlateCarree(),
)
island_lonlat, _ = load_newfoundland_island_polygon(CARTOPY_DIR)
ax1.add_geometries(
    [island_lonlat],
    crs=ccrs.PlateCarree(),
    facecolor="none",
    edgecolor=RED,
    linewidth=1.2,
)
ax1.set_title("(a) Geographic composite")

ax2 = fig.add_subplot(122)
ax2.contourf(
    relative_lon,
    relative_lat,
    relative_mean,
    levels=levels,
    cmap="RdBu_r",
    extend="both",
)
relative_contours = ax2.contour(
    relative_lon,
    relative_lat,
    relative_absolute_mean,
    levels=absolute_levels,
    colors="#333333",
    linewidths=0.55,
)
ax2.clabel(relative_contours, fmt="%d", fontsize=6, inline=True)
stipple_y, stipple_x = np.where(relative_fdr)
keep = (stipple_y % 2 == 0) & (stipple_x % 2 == 0)
ax2.scatter(
    relative_lon[stipple_x[keep]],
    relative_lat[stipple_y[keep]],
    s=3,
    color="#222222",
    alpha=0.65,
)
ax2.scatter(0, 0, marker="*", s=95, color="#111111", edgecolor="white", linewidth=0.5)
ax2.axhline(0, color="#777777", linewidth=0.4)
ax2.axvline(0, color="#777777", linewidth=0.4)
ax2.set_xlabel("Longitude offset from recurvature point (degrees)")
ax2.set_ylabel("Latitude offset from recurvature point (degrees)")
ax2.set_title("(b) Storm-relative composite")
ax2.grid(alpha=0.2, linestyle="--", linewidth=0.4)

colorbar = fig.colorbar(
    filled,
    ax=[ax1, ax2],
    orientation="horizontal",
    fraction=0.045,
    pad=0.18,
    aspect=40,
)
colorbar.set_label("500-hPa geopotential-height anomaly (m)")
fig.suptitle(
    f"500-hPa environment at baseline recurvature times "
    f"({SATELLITE_START}-{YEAR_MAX}; N={len(events)})",
    y=0.99,
)
fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.27, wspace=0.19)
figure_path = FIGURE_DIR / "fig9_z500_composite.png"
fig.savefig(figure_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved:", figure_path)


field_dataset = xr.Dataset(
    {
        "geographic_anomaly": (("lat", "lon"), geographic_mean),
        "geographic_absolute_height": (("lat", "lon"), absolute_mean),
        "geographic_fdr_significant": (
            ("lat", "lon"),
            geographic_fdr.astype(np.int8),
        ),
        "storm_relative_anomaly": (
            ("relative_lat", "relative_lon"),
            relative_mean,
        ),
        "storm_relative_absolute_height": (
            ("relative_lat", "relative_lon"),
            relative_absolute_mean,
        ),
        "storm_relative_fdr_significant": (
            ("relative_lat", "relative_lon"),
            relative_fdr.astype(np.int8),
        ),
    },
    coords={
        "lat": latitude,
        "lon": longitude,
        "relative_lat": relative_lat,
        "relative_lon": relative_lon,
    },
    attrs={
        "event_count": int(len(events)),
        "event_period": f"{SATELLITE_START}-{YEAR_MAX}",
        "pressure_level_hpa": LEVEL_HPA,
        "native_grid_spacing_degrees": 2.5,
        "event_time_resolution": "6-hourly",
        "storm_relative_interpolation": "bilinear on the native latitude-longitude grid",
        "climatology": "1991-2020 monthly NCEP/NCAR Reanalysis 1",
        "significance": "two-sided one-sample t test with Benjamini-Hochberg FDR q=0.05",
    },
)
field_dataset.to_netcdf(OUTPUT_DIR / "z500_composite_fields.nc")

metadata = {
    "event_count": int(len(events)),
    "period": f"{SATELLITE_START}-{YEAR_MAX}",
    "pressure_level_hpa": LEVEL_HPA,
    "native_grid_spacing_degrees": 2.5,
    "event_time_resolution": "6-hourly",
    "storm_relative_interpolation": "bilinear on the native latitude-longitude grid",
    "source": "NCEP/NCAR Reanalysis 1",
    "climatology": "1991-2020 monthly long-term mean",
    "geographic_domain": {
        "latitude": [LAT_SOUTH, LAT_NORTH],
        "longitude_east": [LON_WEST, LON_EAST],
    },
    "storm_relative_domain_degrees": {
        "latitude_offset": [float(relative_lat.min()), float(relative_lat.max())],
        "longitude_offset": [float(relative_lon.min()), float(relative_lon.max())],
    },
    "significance": (
        "Two-sided one-sample t test of event anomalies against zero; "
        "Benjamini-Hochberg false-discovery-rate control at q=0.05."
    ),
    "events": event_metadata,
}
write_json(OUTPUT_DIR / "z500_composite_metadata.json", metadata)
print("Composite rebuild complete")
