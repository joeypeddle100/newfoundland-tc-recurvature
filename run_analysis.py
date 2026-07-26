"""Run the track-climatology analysis and generate manuscript outputs."""

from __future__ import annotations

import math
import os
from pathlib import Path

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely.ops import transform as shapely_transform
import statsmodels.api as sm
import xarray as xr

from analysis_core import (
    BASELINE_DETECTOR,
    BASELINE_REGION,
    DetectorConfig,
    YEAR_MAX,
    YEAR_MIN,
    SATELLITE_START,
    annual_analysis_frame,
    annual_counts,
    binomial_pathway_trend,
    build_track_cache,
    classification_overlap,
    classify_legacy_method,
    classify_recurving_storms,
    config_dict,
    decode_text,
    download_if_needed,
    energy_distance_test,
    load_ibtracs,
    load_newfoundland_island_polygon,
    lonlat_to_cartesian_km,
    north_atlantic_indices,
    poisson_trend,
    post_track_distance_diagnostics_with_transformer,
    quantile_distance_trend,
    robust_linear_trend,
    seasonal_counts,
    status_summary,
    tropical_origin_indices,
    write_json,
    IBTRACS_URL,
)


ROOT = Path(os.environ.get("NL_RECURV_WORKDIR", Path.cwd())).resolve()
DATA_PATH = Path(
    os.environ.get("IBTRACS_PATH", ROOT / "data" / "IBTrACS.ALL.v04r00.nc")
).resolve()
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = ROOT / "figures"
CARTOPY_DIR = Path(
    os.environ.get("CARTOPY_DATA_DIR", ROOT / "data" / "cartopy")
).resolve()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
CARTOPY_DIR.mkdir(parents=True, exist_ok=True)
cartopy.config["data_dir"] = str(CARTOPY_DIR)

BLUE = "#1769AA"
ORANGE = "#D97706"
RED = "#B42318"
GREEN = "#2E7D32"
GRAY = "#555B66"
LIGHT_BLUE = "#D8EAF7"

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save_figure(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved figure:", filename)


def format_interval(low: float, high: float, digits: int = 3) -> str:
    return f"{low:.{digits}f}--{high:.{digits}f}"


def projection_axes(fig, position=111, extent=(-100, -40, 15, 60)):
    ax = fig.add_subplot(position, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#F1EFE8", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#EAF2F8", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.55, zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.35, zorder=2)
    longitude_ticks = np.arange(
        math.ceil(extent[0] / 10.0) * 10.0,
        math.floor(extent[1] / 10.0) * 10.0 + 0.1,
        10.0,
    )
    latitude_ticks = np.arange(
        math.ceil(extent[2] / 10.0) * 10.0,
        math.floor(extent[3] / 10.0) * 10.0 + 0.1,
        10.0,
    )
    ax.set_xticks(longitude_ticks, crs=ccrs.PlateCarree())
    ax.set_yticks(latitude_ticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.grid(linewidth=0.35, alpha=0.45, linestyle="--")
    return ax


def dataframe_to_latex(
    frame: pd.DataFrame,
    path: Path,
    caption: str,
    label: str,
    column_format: str | None = None,
    resize_to_textwidth: bool = False,
) -> None:
    latex = frame.to_latex(
        index=False,
        escape=False,
        column_format=column_format,
        caption=caption,
        label=label,
        position="htbp",
    )
    if resize_to_textwidth:
        latex = latex.replace(
            "\\begin{tabular}",
            "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}",
            1,
        ).replace(
            "\\end{tabular}",
            "\\end{tabular}%\n}",
            1,
        )
    path.write_text(latex)


print("1/8 Loading IBTrACS")
download_if_needed(IBTRACS_URL, DATA_PATH)
variables = [
    "time",
    "lat",
    "lon",
    "season",
    "sid",
    "basin",
    "name",
    "nature",
    "usa_status",
]
ds = load_ibtracs(DATA_PATH)[variables].load()
north_atlantic_source_indices = north_atlantic_indices(ds)
storm_indices = tropical_origin_indices(ds, north_atlantic_source_indices)
print(
    f"North Atlantic source storms in {YEAR_MIN}-{YEAR_MAX}: "
    f"{len(north_atlantic_source_indices)}"
)
print(
    "Storms coded tropical at least once: "
    f"{len(storm_indices)}"
)

tropical_origin_set = set(int(value) for value in storm_indices)
source_scope_audit = pd.DataFrame(
    {
        "dataset_index": north_atlantic_source_indices.astype(int),
        "sid": [
            decode_text(ds["sid"].values[int(k)])
            for k in north_atlantic_source_indices
        ],
        "season": ds["season"].values[
            north_atlantic_source_indices
        ].astype(int),
        "ever_coded_tropical": [
            int(k) in tropical_origin_set
            for k in north_atlantic_source_indices
        ],
    }
)


print("2/8 Running the native-cadence workflow comparator")
legacy_events = classify_legacy_method(ds, storm_indices)
legacy_summary, _ = poisson_trend(annual_counts(legacy_events))
if len(legacy_events) != 165:
    raise RuntimeError(
        "Native-cadence comparator validation failed: "
        f"expected 165 after the tropical-origin filter, got {len(legacy_events)}"
    )


print("3/8 Building regular 6-hour track archive")
track_cache, time_audit = build_track_cache(ds, storm_indices, BASELINE_DETECTOR)
island_lonlat, island_projected = load_newfoundland_island_polygon(CARTOPY_DIR)
velocity_recurvers, eligible_storms = classify_recurving_storms(
    ds,
    storm_indices,
    track_cache,
    island_projected,
    detector="velocity",
)
velocity_events = velocity_recurvers[
    velocity_recurvers["newfoundland_relevant"]
].copy()

heading_recurvers, _ = classify_recurving_storms(
    ds,
    storm_indices,
    track_cache,
    island_projected,
    detector="heading",
)
heading_events = heading_recurvers[
    heading_recurvers["newfoundland_relevant"]
].copy()

method_overlap = classification_overlap(velocity_events, heading_events)
legacy_overlap = classification_overlap(legacy_events, velocity_events)

print(
    "Baseline events:",
    len(velocity_events),
    "| all baseline recurvers:",
    len(velocity_recurvers),
    "| eligible storms:",
    int(eligible_storms["eligible"].sum()),
)


print("4/8 Calculating frequency and pathway trends")
annual = annual_analysis_frame(
    velocity_events,
    velocity_recurvers,
    eligible_storms,
)
full_count_summary, full_count_fitted = poisson_trend(
    annual[["season", "relevant_events"]].rename(
        columns={"relevant_events": "count"}
    )
)
satellite_annual = annual[annual["season"] >= SATELLITE_START].copy()
satellite_count_summary, satellite_count_fitted = poisson_trend(
    satellite_annual[["season", "relevant_events"]].rename(
        columns={"relevant_events": "count"}
    )
)

pathway_eligible_full = binomial_pathway_trend(
    annual, "relevant_events", "eligible_storms"
)
pathway_eligible_satellite = binomial_pathway_trend(
    satellite_annual, "relevant_events", "eligible_storms"
)
pathway_recurver_full = binomial_pathway_trend(
    annual, "relevant_events", "recurving_storms"
)
pathway_recurver_satellite = binomial_pathway_trend(
    satellite_annual, "relevant_events", "recurving_storms"
)


print("5/8 Calculating location, proximity, and seasonal diagnostics")
latitude_trend = robust_linear_trend(velocity_events, "recurv_lat")
longitude_trend = robust_linear_trend(velocity_events, "recurv_lon")
early_events = velocity_events[velocity_events["season"] < SATELLITE_START]
late_events = velocity_events[velocity_events["season"] >= SATELLITE_START]
early_location_xyz = lonlat_to_cartesian_km(
    early_events["recurv_lon"].to_numpy(),
    early_events["recurv_lat"].to_numpy(),
)
late_location_xyz = lonlat_to_cartesian_km(
    late_events["recurv_lon"].to_numpy(),
    late_events["recurv_lat"].to_numpy(),
)
location_energy = energy_distance_test(
    early_location_xyz,
    late_location_xyz,
)

proximity_all_mean = robust_linear_trend(
    velocity_recurvers, "minimum_distance_to_newfoundland_km"
)
proximity_all_median = quantile_distance_trend(
    velocity_recurvers, "minimum_distance_to_newfoundland_km"
)
proximity_relevant_mean = robust_linear_trend(
    velocity_events, "minimum_distance_to_newfoundland_km"
)
proximity_relevant_median = quantile_distance_trend(
    velocity_events, "minimum_distance_to_newfoundland_km"
)

non_endpoint_recurvers = velocity_recurvers[
    ~velocity_recurvers["closest_at_track_endpoint"]
].copy()
proximity_non_endpoint_mean = robust_linear_trend(
    non_endpoint_recurvers, "minimum_distance_to_newfoundland_km"
)
proximity_non_endpoint_median = quantile_distance_trend(
    non_endpoint_recurvers, "minimum_distance_to_newfoundland_km"
)
post_duration_trend = robust_linear_trend(
    velocity_recurvers, "post_recurvature_duration_hours"
)
early_recurvers = velocity_recurvers[
    velocity_recurvers["season"] < SATELLITE_START
]
late_recurvers = velocity_recurvers[
    velocity_recurvers["season"] >= SATELLITE_START
]
endpoint_summary = {
    "endpoint_count": int(
        velocity_recurvers["closest_at_track_endpoint"].sum()
    ),
    "endpoint_fraction": float(
        velocity_recurvers["closest_at_track_endpoint"].mean()
    ),
    "relevant_endpoint_count": int(
        velocity_events["closest_at_track_endpoint"].sum()
    ),
    "relevant_endpoint_fraction": float(
        velocity_events["closest_at_track_endpoint"].mean()
    ),
    "early_endpoint_fraction": float(
        early_recurvers["closest_at_track_endpoint"].mean()
    ),
    "late_endpoint_fraction": float(
        late_recurvers["closest_at_track_endpoint"].mean()
    ),
    "non_endpoint_mean_trend": proximity_non_endpoint_mean,
    "non_endpoint_median_trend": proximity_non_endpoint_median,
    "post_recurvature_duration_trend": post_duration_trend,
}

seasonality = seasonal_counts(velocity_events)
nature_counts = status_summary(velocity_events)


print("6/8 Running sensitivity analyses")
threshold_rows = []
for threshold in (300, 500, 600, 800, 1000):
    events = velocity_recurvers[
        velocity_recurvers["minimum_distance_to_newfoundland_km"] <= threshold
    ]
    trend, _ = poisson_trend(annual_counts(events))
    satellite_events = events[events["season"] >= SATELLITE_START]
    satellite_trend, _ = poisson_trend(
        annual_counts(
            satellite_events,
            year_min=SATELLITE_START,
            year_max=YEAR_MAX,
        )
    )
    threshold_annual = annual_analysis_frame(
        events,
        velocity_recurvers,
        eligible_storms,
    )
    threshold_satellite_annual = threshold_annual[
        threshold_annual["season"] >= SATELLITE_START
    ]
    satellite_eligible_pathway = binomial_pathway_trend(
        threshold_satellite_annual,
        "relevant_events",
        "eligible_storms",
    )
    satellite_recurver_pathway = binomial_pathway_trend(
        threshold_satellite_annual,
        "relevant_events",
        "recurving_storms",
    )
    threshold_rows.append(
        {
            "threshold_km": threshold,
            "events": len(events),
            "irr_per_decade": trend["irr_per_decade"],
            "ci_low": trend["ci_low"],
            "ci_high": trend["ci_high"],
            "p_value": trend["p_value"],
            "dispersion": trend["pearson_dispersion"],
            "satellite_events": len(satellite_events),
            "satellite_irr_per_decade": satellite_trend["irr_per_decade"],
            "satellite_ci_low": satellite_trend["ci_low"],
            "satellite_ci_high": satellite_trend["ci_high"],
            "satellite_p_value": satellite_trend["p_value"],
            "satellite_hac_ci_low": satellite_trend["hac_ci_low"],
            "satellite_hac_ci_high": satellite_trend["hac_ci_high"],
            "satellite_hac_p_value": satellite_trend["hac_p_value"],
            "satellite_eligible_or": satellite_eligible_pathway[
                "odds_ratio_per_decade"
            ],
            "satellite_eligible_ci_low": satellite_eligible_pathway["ci_low"],
            "satellite_eligible_ci_high": satellite_eligible_pathway["ci_high"],
            "satellite_eligible_p_value": satellite_eligible_pathway["p_value"],
            "satellite_recurver_or": satellite_recurver_pathway[
                "odds_ratio_per_decade"
            ],
            "satellite_recurver_ci_low": satellite_recurver_pathway["ci_low"],
            "satellite_recurver_ci_high": satellite_recurver_pathway["ci_high"],
            "satellite_recurver_p_value": satellite_recurver_pathway["p_value"],
        }
    )
threshold_sensitivity = pd.DataFrame(threshold_rows)

detector_rows = []
baseline_set = set(velocity_events["sid"])
for latitude_gate in (25.0, 30.0):
    for window_hours in (18, 24, 30):
        for post_east in (2.5, 5.0, 7.5, 10.0):
            detector_config = DetectorConfig(
                grid_hours=6,
                window_hours=window_hours,
                latitude_gate_deg_n=latitude_gate,
                pre_east_max_kmh=0.0,
                post_east_min_kmh=post_east,
                post_north_min_kmh=0.0,
                min_eastward_acceleration_kmh=max(5.0, post_east),
                maximum_source_gap_hours=12.0,
            )
            recurvers, candidate_eligible = classify_recurving_storms(
                ds,
                storm_indices,
                track_cache,
                island_projected,
                detector="velocity",
                detector_config=detector_config,
            )
            events = recurvers[recurvers["newfoundland_relevant"]]
            trend, _ = poisson_trend(annual_counts(events))
            satellite_events = events[events["season"] >= SATELLITE_START]
            satellite_trend, _ = poisson_trend(
                annual_counts(
                    satellite_events,
                    year_min=SATELLITE_START,
                    year_max=YEAR_MAX,
                )
            )
            candidate_annual = annual_analysis_frame(
                events,
                recurvers,
                candidate_eligible,
            )
            candidate_satellite_annual = candidate_annual[
                candidate_annual["season"] >= SATELLITE_START
            ]
            satellite_eligible_pathway = binomial_pathway_trend(
                candidate_satellite_annual,
                "relevant_events",
                "eligible_storms",
            )
            satellite_recurver_pathway = binomial_pathway_trend(
                candidate_satellite_annual,
                "relevant_events",
                "recurving_storms",
            )
            candidate_set = set(events["sid"])
            union = baseline_set | candidate_set
            detector_rows.append(
                {
                    "latitude_gate_deg_n": latitude_gate,
                    "window_hours": window_hours,
                    "post_east_min_kmh": post_east,
                    "events": len(events),
                    "irr_per_decade": trend["irr_per_decade"],
                    "ci_low": trend["ci_low"],
                    "ci_high": trend["ci_high"],
                    "p_value": trend["p_value"],
                    "dispersion": trend["pearson_dispersion"],
                    "satellite_events": len(satellite_events),
                    "satellite_irr_per_decade": satellite_trend[
                        "irr_per_decade"
                    ],
                    "satellite_ci_low": satellite_trend["ci_low"],
                    "satellite_ci_high": satellite_trend["ci_high"],
                    "satellite_p_value": satellite_trend["p_value"],
                    "satellite_hac_ci_low": satellite_trend["hac_ci_low"],
                    "satellite_hac_ci_high": satellite_trend["hac_ci_high"],
                    "satellite_hac_p_value": satellite_trend["hac_p_value"],
                    "satellite_eligible_or": satellite_eligible_pathway[
                        "odds_ratio_per_decade"
                    ],
                    "satellite_eligible_ci_low": satellite_eligible_pathway[
                        "ci_low"
                    ],
                    "satellite_eligible_ci_high": satellite_eligible_pathway[
                        "ci_high"
                    ],
                    "satellite_eligible_p_value": satellite_eligible_pathway[
                        "p_value"
                    ],
                    "satellite_recurver_or": satellite_recurver_pathway[
                        "odds_ratio_per_decade"
                    ],
                    "satellite_recurver_ci_low": satellite_recurver_pathway[
                        "ci_low"
                    ],
                    "satellite_recurver_ci_high": satellite_recurver_pathway[
                        "ci_high"
                    ],
                    "satellite_recurver_p_value": satellite_recurver_pathway[
                        "p_value"
                    ],
                    "jaccard_with_baseline": (
                        len(baseline_set & candidate_set) / len(union)
                        if union
                        else np.nan
                    ),
                }
            )
detector_sensitivity = pd.DataFrame(detector_rows)

alternative_crs = CRS.from_proj4(
    "+proj=aeqd +lat_0=48.5 +lon_0=-56 "
    "+datum=WGS84 +units=m +no_defs"
)
alternative_transformer = Transformer.from_crs(
    "EPSG:4326",
    alternative_crs,
    always_xy=True,
)
alternative_island_projected = shapely_transform(
    alternative_transformer.transform,
    island_lonlat,
)
projection_rows = []
for event in velocity_recurvers.itertuples(index=False):
    alternative_diagnostics = (
        post_track_distance_diagnostics_with_transformer(
            track_cache[int(event.dataset_index)],
            int(event.recurv_index),
            alternative_island_projected,
            alternative_transformer,
        )
    )
    alternative_distance = alternative_diagnostics[
        "minimum_distance_to_newfoundland_km"
    ]
    projection_rows.append(
        {
            "dataset_index": int(event.dataset_index),
            "sid": event.sid,
            "season": int(event.season),
            "baseline_distance_km": float(
                event.minimum_distance_to_newfoundland_km
            ),
            "alternative_distance_km": alternative_distance,
            "absolute_difference_km": abs(
                alternative_distance
                - float(event.minimum_distance_to_newfoundland_km)
            ),
            "baseline_relevant": bool(event.newfoundland_relevant),
            "alternative_relevant": bool(
                alternative_distance
                <= BASELINE_REGION.proximity_threshold_km
            ),
        }
    )
projection_distance_sensitivity = pd.DataFrame(projection_rows)
alternative_projection_events = projection_distance_sensitivity[
    projection_distance_sensitivity["alternative_relevant"]
]
alternative_projection_count_trend, _ = poisson_trend(
    annual_counts(alternative_projection_events)
)
alternative_projection_satellite_trend, _ = poisson_trend(
    annual_counts(
        alternative_projection_events[
            alternative_projection_events["season"] >= SATELLITE_START
        ],
        year_min=SATELLITE_START,
        year_max=YEAR_MAX,
    )
)
alternative_projection_mean_trend = robust_linear_trend(
    projection_distance_sensitivity,
    "alternative_distance_km",
)
alternative_projection_median_trend = quantile_distance_trend(
    projection_distance_sensitivity,
    "alternative_distance_km",
)
projection_summary = {
    "baseline_projection": "EPSG:3347",
    "alternative_projection": (
        "Newfoundland-centred azimuthal equidistant "
        "(48.5 N, 56 W; WGS84)"
    ),
    "baseline_relevant_events": int(len(velocity_events)),
    "alternative_relevant_events": int(len(alternative_projection_events)),
    "classification_changes": int(
        (
            projection_distance_sensitivity["baseline_relevant"]
            != projection_distance_sensitivity["alternative_relevant"]
        ).sum()
    ),
    "distance_correlation": float(
        projection_distance_sensitivity[
            ["baseline_distance_km", "alternative_distance_km"]
        ].corr().iloc[0, 1]
    ),
    "median_absolute_difference_km": float(
        projection_distance_sensitivity["absolute_difference_km"].median()
    ),
    "p95_absolute_difference_km": float(
        projection_distance_sensitivity["absolute_difference_km"].quantile(
            0.95
        )
    ),
    "maximum_absolute_difference_km": float(
        projection_distance_sensitivity["absolute_difference_km"].max()
    ),
    "full_count_trend": alternative_projection_count_trend,
    "satellite_count_trend": alternative_projection_satellite_trend,
    "mean_distance_trend": alternative_projection_mean_trend,
    "median_distance_trend": alternative_projection_median_trend,
}

endpoint_distance_sensitivity = pd.DataFrame(
    [
        {
            "sample": "All detected recurvers",
            "n": len(velocity_recurvers),
            "mean_slope_km_per_decade": proximity_all_mean[
                "slope_per_decade"
            ],
            "mean_ci_low": proximity_all_mean["ci_low"],
            "mean_ci_high": proximity_all_mean["ci_high"],
            "mean_p_value": proximity_all_mean["p_value"],
            "median_slope_km_per_decade": proximity_all_median[
                "median_slope_km_per_decade"
            ],
            "median_ci_low": proximity_all_median["ci_low"],
            "median_ci_high": proximity_all_median["ci_high"],
            "median_p_value": proximity_all_median["p_value"],
        },
        {
            "sample": "Excluding endpoint minima",
            "n": len(non_endpoint_recurvers),
            "mean_slope_km_per_decade": proximity_non_endpoint_mean[
                "slope_per_decade"
            ],
            "mean_ci_low": proximity_non_endpoint_mean["ci_low"],
            "mean_ci_high": proximity_non_endpoint_mean["ci_high"],
            "mean_p_value": proximity_non_endpoint_mean["p_value"],
            "median_slope_km_per_decade": proximity_non_endpoint_median[
                "median_slope_km_per_decade"
            ],
            "median_ci_low": proximity_non_endpoint_median["ci_low"],
            "median_ci_high": proximity_non_endpoint_median["ci_high"],
            "median_p_value": proximity_non_endpoint_median["p_value"],
        },
    ]
)


print("7/8 Writing data products")
source_scope_audit.to_csv(OUTPUT_DIR / "source_scope_audit.csv", index=False)
time_audit.to_csv(OUTPUT_DIR / "time_step_and_eligibility_audit.csv", index=False)
legacy_events.to_csv(
    OUTPUT_DIR / "native_cadence_comparator_events.csv",
    index=False,
)
velocity_recurvers.to_csv(
    OUTPUT_DIR / "all_corrected_recurving_storms.csv", index=False
)
velocity_events.to_csv(
    OUTPUT_DIR / "corrected_newfoundland_relevant_events.csv", index=False
)
heading_events.to_csv(
    OUTPUT_DIR / "alternative_heading_events.csv", index=False
)
eligible_storms.to_csv(OUTPUT_DIR / "eligible_storms.csv", index=False)
annual.to_csv(OUTPUT_DIR / "annual_counts_and_denominators.csv", index=False)
seasonality.to_csv(OUTPUT_DIR / "seasonality.csv", index=False)
nature_counts.to_csv(OUTPUT_DIR / "nature_at_recurvature.csv", index=False)
threshold_sensitivity.to_csv(
    OUTPUT_DIR / "threshold_sensitivity.csv", index=False
)
detector_sensitivity.to_csv(
    OUTPUT_DIR / "detector_sensitivity.csv", index=False
)
endpoint_distance_sensitivity.to_csv(
    OUTPUT_DIR / "endpoint_distance_sensitivity.csv",
    index=False,
)
projection_distance_sensitivity.to_csv(
    OUTPUT_DIR / "projection_distance_sensitivity.csv",
    index=False,
)
write_json(
    OUTPUT_DIR / "projection_sensitivity_summary.json",
    projection_summary,
)

only_velocity = sorted(set(velocity_events["sid"]) - set(heading_events["sid"]))
only_heading = sorted(set(heading_events["sid"]) - set(velocity_events["sid"]))
discordant = pd.DataFrame(
    [
        {"sid": sid, "classification": "velocity_only"} for sid in only_velocity
    ]
    + [{"sid": sid, "classification": "heading_only"} for sid in only_heading]
)
discordant.to_csv(OUTPUT_DIR / "discordant_detector_events.csv", index=False)


summary = {
    "configuration": config_dict(),
    "sample": {
        "north_atlantic_source_storms": int(
            len(north_atlantic_source_indices)
        ),
        "tropical_origin_storms": int(len(storm_indices)),
        "excluded_without_tropical_code": int(
            len(north_atlantic_source_indices) - len(storm_indices)
        ),
        "eligible_storms": int(eligible_storms["eligible"].sum()),
        "all_corrected_recurvers": int(len(velocity_recurvers)),
        "corrected_relevant_events": int(len(velocity_events)),
        "native_cadence_comparator_events": int(len(legacy_events)),
        "heading_relevant_events": int(len(heading_events)),
    },
    "frequency": {
        "full_period": full_count_summary,
        "satellite_era": satellite_count_summary,
    },
    "pathway_rates": {
        "relevant_among_eligible_full": pathway_eligible_full,
        "relevant_among_eligible_satellite": pathway_eligible_satellite,
        "relevant_among_recurvers_full": pathway_recurver_full,
        "relevant_among_recurvers_satellite": pathway_recurver_satellite,
    },
    "location": {
        "latitude": latitude_trend,
        "longitude": longitude_trend,
        "era_energy_test": location_energy,
        "early_n": int(len(early_events)),
        "late_n": int(len(late_events)),
        "early_median_lat": float(early_events["recurv_lat"].median()),
        "late_median_lat": float(late_events["recurv_lat"].median()),
        "early_median_lon": float(early_events["recurv_lon"].median()),
        "late_median_lon": float(late_events["recurv_lon"].median()),
    },
    "proximity": {
        "all_recurvers_mean_trend": proximity_all_mean,
        "all_recurvers_median_trend": proximity_all_median,
        "relevant_only_mean_trend": proximity_relevant_mean,
        "relevant_only_median_trend": proximity_relevant_median,
        "all_recurvers_early_median_km": float(
            velocity_recurvers.loc[
                velocity_recurvers["season"] < SATELLITE_START,
                "minimum_distance_to_newfoundland_km",
            ].median()
        ),
        "all_recurvers_late_median_km": float(
            velocity_recurvers.loc[
                velocity_recurvers["season"] >= SATELLITE_START,
                "minimum_distance_to_newfoundland_km",
            ].median()
        ),
        "endpoint_diagnostic": endpoint_summary,
        "alternative_projection": projection_summary,
    },
    "sensitivity": {
        "threshold": {
            "full_period_irr_range": [
                float(threshold_sensitivity["irr_per_decade"].min()),
                float(threshold_sensitivity["irr_per_decade"].max()),
            ],
            "satellite_irr_range": [
                float(
                    threshold_sensitivity[
                        "satellite_irr_per_decade"
                    ].min()
                ),
                float(
                    threshold_sensitivity[
                        "satellite_irr_per_decade"
                    ].max()
                ),
            ],
            "satellite_model_p_below_0_05": int(
                (threshold_sensitivity["satellite_p_value"] < 0.05).sum()
            ),
            "satellite_hac_p_below_0_05": int(
                (
                    threshold_sensitivity["satellite_hac_p_value"]
                    < 0.05
                ).sum()
            ),
        },
        "detector": {
            "full_period_event_range": [
                int(detector_sensitivity["events"].min()),
                int(detector_sensitivity["events"].max()),
            ],
            "full_period_irr_range": [
                float(detector_sensitivity["irr_per_decade"].min()),
                float(detector_sensitivity["irr_per_decade"].max()),
            ],
            "satellite_event_range": [
                int(detector_sensitivity["satellite_events"].min()),
                int(detector_sensitivity["satellite_events"].max()),
            ],
            "satellite_irr_range": [
                float(
                    detector_sensitivity[
                        "satellite_irr_per_decade"
                    ].min()
                ),
                float(
                    detector_sensitivity[
                        "satellite_irr_per_decade"
                    ].max()
                ),
            ],
            "satellite_model_p_below_0_05": int(
                (detector_sensitivity["satellite_p_value"] < 0.05).sum()
            ),
            "satellite_hac_p_below_0_05": int(
                (
                    detector_sensitivity["satellite_hac_p_value"]
                    < 0.05
                ).sum()
            ),
            "satellite_eligible_pathway_p_below_0_05": int(
                (
                    detector_sensitivity[
                        "satellite_eligible_p_value"
                    ]
                    < 0.05
                ).sum()
            ),
            "satellite_recurver_pathway_p_below_0_05": int(
                (
                    detector_sensitivity[
                        "satellite_recurver_p_value"
                    ]
                    < 0.05
                ).sum()
            ),
        },
    },
    "method_comparison": {
        "velocity_vs_heading": method_overlap,
        "legacy_vs_corrected": legacy_overlap,
    },
    "legacy_reproduction": legacy_summary,
}
write_json(OUTPUT_DIR / "analysis_summary.json", summary)


print("8/8 Generating figures and LaTeX tables")

# Figure 1: objective geographic definition
inverse_project = Transformer.from_crs(
    "EPSG:3347", "EPSG:4326", always_xy=True
).transform
buffer_600_lonlat = shapely_transform(
    inverse_project, island_projected.buffer(600_000.0)
)
fig = plt.figure(figsize=(7.4, 6.2))
ax = projection_axes(fig, extent=(-70, -44, 40, 57))
ax.add_geometries(
    [buffer_600_lonlat],
    crs=ccrs.PlateCarree(),
    facecolor=LIGHT_BLUE,
    edgecolor=BLUE,
    linewidth=1.0,
    alpha=0.55,
    zorder=1,
)
ax.add_geometries(
    [island_lonlat],
    crs=ccrs.PlateCarree(),
    facecolor="#F7F2D0",
    edgecolor="#222222",
    linewidth=1.0,
    zorder=3,
)
ax.set_title("Newfoundland island and the 600-km track-proximity region")
ax.legend(
    handles=[
        Line2D([0], [0], color=BLUE, lw=6, alpha=0.35, label="600-km buffer"),
        Line2D([0], [0], color="#222222", lw=2, label="Newfoundland island"),
    ],
    loc="lower left",
)
save_figure(fig, "fig1_newfoundland_geometry.png")

# Figure 2: seasonality
fig, ax = plt.subplots(figsize=(7.4, 4.2))
ax.bar(
    seasonality["month_name"].str[:3],
    seasonality["events"],
    color=BLUE,
    width=0.78,
)
ax.set_ylabel("Number of events")
ax.set_xlabel("Month of diagnosed recurvature")
ax.set_title(
    f"Seasonality of baseline Newfoundland-relevant recurvature "
    f"({YEAR_MIN}-{YEAR_MAX})"
)
ax.grid(axis="y", alpha=0.2)
save_figure(fig, "fig2_seasonality.png")

# Figure 3: annual counts with Poisson fit
fig, ax = plt.subplots(figsize=(8.2, 4.5))
ax.plot(
    annual["season"],
    annual["relevant_events"],
    color=GRAY,
    lw=0.9,
    marker="o",
    ms=2.8,
    label="Annual count",
)
ax.fill_between(
    full_count_fitted["season"],
    full_count_fitted["ci_low"],
    full_count_fitted["ci_high"],
    color=BLUE,
    alpha=0.16,
    linewidth=0,
    label="95% confidence interval",
)
ax.plot(
    full_count_fitted["season"],
    full_count_fitted["fitted"],
    color=BLUE,
    lw=2.0,
    label="Poisson fitted mean",
)
ax.axvline(
    SATELLITE_START,
    color=ORANGE,
    linestyle="--",
    lw=1,
    label="Satellite-era sensitivity start",
)
ax.set_ylabel("Number of events")
ax.set_xlabel("Season")
ax.set_ylim(bottom=0)
ax.set_title("Annual Newfoundland-relevant recurvature counts")
ax.legend(ncol=2, frameon=False)
ax.grid(axis="y", alpha=0.2)
save_figure(fig, "fig3_annual_counts_poisson.png")

# Figure 4: rolling context, with partial-window ends suppressed
rolling = (
    annual.set_index("season")["relevant_events"]
    .rolling(window=10, center=True, min_periods=10)
    .mean()
)
fig, ax = plt.subplots(figsize=(8.2, 4.4))
ax.plot(
    annual["season"],
    annual["relevant_events"],
    color="#8A8F98",
    lw=0.8,
    label="Annual count",
)
ax.plot(
    rolling.index,
    rolling.values,
    color=ORANGE,
    lw=2.1,
    label="Centered 10-year mean",
)
ax.set_ylabel("Number of events")
ax.set_xlabel("Season")
ax.set_title("Interannual variability and centered 10-year mean")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.2)
save_figure(fig, "fig4_rolling_mean.png")

# Figure 5: recurvature locations in equal-area visual panels
fig = plt.figure(figsize=(10.2, 4.8))
for panel, events, title in (
    (121, early_events, f"{YEAR_MIN}-{SATELLITE_START - 1} (N={len(early_events)})"),
    (122, late_events, f"{SATELLITE_START}-{YEAR_MAX} (N={len(late_events)})"),
):
    ax = projection_axes(fig, panel, extent=(-100, -40, 20, 55))
    ax.scatter(
        events["recurv_lon"],
        events["recurv_lat"],
        s=15,
        alpha=0.65,
        color=BLUE if panel == 121 else ORANGE,
        transform=ccrs.PlateCarree(),
        zorder=4,
    )
    ax.scatter(
        events["recurv_lon"].median(),
        events["recurv_lat"].median(),
        marker="*",
        s=95,
        color=RED,
        edgecolor="white",
        linewidth=0.5,
        transform=ccrs.PlateCarree(),
        zorder=5,
        label="Median location",
    )
    ax.set_title(title)
    ax.legend(loc="lower left", frameon=False)
fig.suptitle("Baseline recurvature-point locations by era", y=1.01)
save_figure(fig, "fig5_recurvature_locations.png")

# Figure 6: proximity of every detected recurver, avoiding threshold truncation
distance_response = velocity_recurvers[
    "minimum_distance_to_newfoundland_km"
].to_numpy(dtype=float)
year = velocity_recurvers["season"].to_numpy(dtype=float)
design = sm.add_constant(year - year.mean())
distance_model = sm.OLS(distance_response, design).fit(cov_type="HC3")
grid_year = np.linspace(YEAR_MIN, YEAR_MAX, 150)
grid_design = sm.add_constant(grid_year - year.mean())
grid_prediction = distance_model.get_prediction(grid_design).summary_frame()

fig, ax = plt.subplots(figsize=(8.2, 4.8))
inside = velocity_recurvers["newfoundland_relevant"]
ax.scatter(
    velocity_recurvers.loc[~inside, "season"],
    velocity_recurvers.loc[
        ~inside, "minimum_distance_to_newfoundland_km"
    ],
    s=10,
    alpha=0.35,
    color="#8A8F98",
    label="Other recurvers",
)
ax.scatter(
    velocity_recurvers.loc[inside, "season"],
    velocity_recurvers.loc[
        inside, "minimum_distance_to_newfoundland_km"
    ],
    s=13,
    alpha=0.65,
    color=BLUE,
    label="Within 600 km",
)
ax.plot(grid_year, grid_prediction["mean"], color=ORANGE, lw=2, label="Mean trend")
ax.fill_between(
    grid_year,
    grid_prediction["mean_ci_lower"],
    grid_prediction["mean_ci_upper"],
    color=ORANGE,
    alpha=0.15,
    linewidth=0,
)
ax.axhline(
    BASELINE_REGION.proximity_threshold_km,
    color=RED,
    linestyle="--",
    lw=1,
    label="600-km classification threshold",
)
ax.set_ylabel("Minimum post-recurvature distance (km)")
ax.set_xlabel("Season")
ax.set_title("Proximity of all detected recurving storms to Newfoundland")
ax.legend(ncol=2, frameon=False)
ax.grid(axis="y", alpha=0.2)
save_figure(fig, "fig6_proximity_all_recurvers.png")

# Figure 7: normalized distance distributions by era
fig, ax = plt.subplots(figsize=(7.6, 4.6))
for events, color, label in (
    (
        velocity_recurvers[velocity_recurvers["season"] < SATELLITE_START],
        BLUE,
        f"{YEAR_MIN}-{SATELLITE_START - 1}",
    ),
    (
        velocity_recurvers[velocity_recurvers["season"] >= SATELLITE_START],
        ORANGE,
        f"{SATELLITE_START}-{YEAR_MAX}",
    ),
):
    values = np.sort(
        events["minimum_distance_to_newfoundland_km"].to_numpy(dtype=float)
    )
    probability = np.arange(1, len(values) + 1) / len(values)
    ax.step(values, probability, where="post", color=color, lw=2, label=label)
ax.axvline(
    BASELINE_REGION.proximity_threshold_km,
    color=RED,
    linestyle="--",
    lw=1,
    label="600-km threshold",
)
ax.set_xlabel("Minimum post-recurvature distance (km)")
ax.set_ylabel("Cumulative proportion")
ax.set_title("Normalized proximity distributions for all recurvers")
ax.legend(frameon=False)
ax.grid(alpha=0.2)
save_figure(fig, "fig7_proximity_ecdf.png")

# Figure 8: threshold sensitivity distinguishes event magnitude from trend
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 4.2))
ax1.plot(
    threshold_sensitivity["threshold_km"],
    threshold_sensitivity["events"],
    marker="o",
    color=BLUE,
    lw=2,
)
ax1.axvline(600, color=RED, linestyle="--", lw=1)
ax1.set_xlabel("Proximity threshold (km)")
ax1.set_ylabel("Classified events")
ax1.set_title("Event-list sensitivity")
ax1.grid(alpha=0.2)

ax2.errorbar(
    threshold_sensitivity["threshold_km"],
    threshold_sensitivity["irr_per_decade"],
    yerr=[
        threshold_sensitivity["irr_per_decade"]
        - threshold_sensitivity["ci_low"],
        threshold_sensitivity["ci_high"]
        - threshold_sensitivity["irr_per_decade"],
    ],
    fmt="o-",
    color=ORANGE,
    capsize=3,
    lw=1.8,
)
ax2.axhline(1.0, color=GRAY, linestyle="--", lw=1)
ax2.axvline(600, color=RED, linestyle="--", lw=1)
ax2.set_xlabel("Proximity threshold (km)")
ax2.set_ylabel("IRR per decade (95% CI)")
ax2.set_title("Frequency-trend sensitivity")
ax2.grid(alpha=0.2)
save_figure(fig, "fig8_threshold_sensitivity.png")

# Appendix figure: all detector-discordant tracks
velocity_lookup = {
    row["sid"]: row for _, row in velocity_events.iterrows()
}
heading_lookup = {row["sid"]: row for _, row in heading_events.iterrows()}
discordant_ids = sorted(set(velocity_lookup) ^ set(heading_lookup))
ncols = 4
nrows = int(np.ceil(len(discordant_ids) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 2.45 * nrows))
axes = np.atleast_1d(axes).ravel()
island_parts = (
    list(island_lonlat.geoms)
    if hasattr(island_lonlat, "geoms")
    else [island_lonlat]
)
for ax, sid in zip(axes, discordant_ids):
    row = velocity_lookup.get(sid, heading_lookup.get(sid))
    track = track_cache[int(row["dataset_index"])]
    ax.plot(track["lon"], track["lat"], color="#9A9A9A", lw=0.9)
    for polygon in island_parts:
        x_coord, y_coord = polygon.exterior.xy
        ax.fill(
            x_coord,
            y_coord,
            facecolor="#F1EFE8",
            edgecolor="#333333",
            linewidth=0.45,
        )
    if sid in velocity_lookup:
        event = velocity_lookup[sid]
        ax.scatter(
            event["recurv_lon"],
            event["recurv_lat"],
            color=RED,
            s=20,
            label="Velocity detector",
        )
    if sid in heading_lookup:
        event = heading_lookup[sid]
        ax.scatter(
            event["recurv_lon"],
            event["recurv_lat"],
            color=BLUE,
            marker="x",
            s=22,
            label="Heading detector",
        )
    detector_label = "V" if sid in velocity_lookup else "H"
    ax.set_title(
        f"{row['name']} ({int(row['season'])}; {detector_label} only)",
        fontsize=7.5,
    )
    ax.grid(alpha=0.18)
for ax in axes[len(discordant_ids) :]:
    ax.axis("off")
fig.suptitle("Tracks classified by only one detector", y=1.002)
save_figure(fig, "figA1_detector_discordant_tracks.pdf")
stale_discordant_png = FIGURE_DIR / "figA1_detector_discordant_tracks.png"
if stale_discordant_png.exists():
    stale_discordant_png.unlink()


# Main trend table
frequency_table = pd.DataFrame(
    [
        {
            "Period": f"{YEAR_MIN}--{YEAR_MAX}",
            "$N$": full_count_summary["total_events"],
            "IRR decade$^{-1}$": f"{full_count_summary['irr_per_decade']:.3f}",
            "95\\% CI": format_interval(
                full_count_summary["ci_low"], full_count_summary["ci_high"]
            ),
            "$p$": f"{full_count_summary['p_value']:.3f}",
            "HAC 95\\% CI": format_interval(
                full_count_summary["hac_ci_low"],
                full_count_summary["hac_ci_high"],
            ),
            "Dispersion": f"{full_count_summary['pearson_dispersion']:.2f}",
        },
        {
            "Period": f"{SATELLITE_START}--{YEAR_MAX}",
            "$N$": satellite_count_summary["total_events"],
            "IRR decade$^{-1}$": f"{satellite_count_summary['irr_per_decade']:.3f}",
            "95\\% CI": format_interval(
                satellite_count_summary["ci_low"],
                satellite_count_summary["ci_high"],
            ),
            "$p$": f"{satellite_count_summary['p_value']:.3f}",
            "HAC 95\\% CI": format_interval(
                satellite_count_summary["hac_ci_low"],
                satellite_count_summary["hac_ci_high"],
            ),
            "Dispersion": f"{satellite_count_summary['pearson_dispersion']:.2f}",
        },
    ]
)
dataframe_to_latex(
    frequency_table,
    OUTPUT_DIR / "table_frequency_trends.tex",
    (
        "Poisson trend estimates for annual Newfoundland-relevant "
        "recurvature counts. HAC intervals allow for short-lag residual "
        "dependence."
    ),
    "tab:frequency_trends",
)

pathway_table = pd.DataFrame(
    [
        {
            "Denominator": "Eligible storms",
            "Period": f"{YEAR_MIN}--{YEAR_MAX}",
            "Events/trials": (
                f"{pathway_eligible_full['total_successes']}/"
                f"{pathway_eligible_full['total_trials']}"
            ),
            "OR decade$^{-1}$": f"{pathway_eligible_full['odds_ratio_per_decade']:.3f}",
            "95\\% CI": format_interval(
                pathway_eligible_full["ci_low"],
                pathway_eligible_full["ci_high"],
            ),
            "$p$": f"{pathway_eligible_full['p_value']:.3f}",
        },
        {
            "Denominator": "Eligible storms",
            "Period": f"{SATELLITE_START}--{YEAR_MAX}",
            "Events/trials": (
                f"{pathway_eligible_satellite['total_successes']}/"
                f"{pathway_eligible_satellite['total_trials']}"
            ),
            "OR decade$^{-1}$": (
                f"{pathway_eligible_satellite['odds_ratio_per_decade']:.3f}"
            ),
            "95\\% CI": format_interval(
                pathway_eligible_satellite["ci_low"],
                pathway_eligible_satellite["ci_high"],
            ),
            "$p$": f"{pathway_eligible_satellite['p_value']:.3f}",
        },
        {
            "Denominator": "Detected recurvers",
            "Period": f"{YEAR_MIN}--{YEAR_MAX}",
            "Events/trials": (
                f"{pathway_recurver_full['total_successes']}/"
                f"{pathway_recurver_full['total_trials']}"
            ),
            "OR decade$^{-1}$": f"{pathway_recurver_full['odds_ratio_per_decade']:.3f}",
            "95\\% CI": format_interval(
                pathway_recurver_full["ci_low"],
                pathway_recurver_full["ci_high"],
            ),
            "$p$": f"{pathway_recurver_full['p_value']:.3f}",
        },
        {
            "Denominator": "Detected recurvers",
            "Period": f"{SATELLITE_START}--{YEAR_MAX}",
            "Events/trials": (
                f"{pathway_recurver_satellite['total_successes']}/"
                f"{pathway_recurver_satellite['total_trials']}"
            ),
            "OR decade$^{-1}$": (
                f"{pathway_recurver_satellite['odds_ratio_per_decade']:.3f}"
            ),
            "95\\% CI": format_interval(
                pathway_recurver_satellite["ci_low"],
                pathway_recurver_satellite["ci_high"],
            ),
            "$p$": f"{pathway_recurver_satellite['p_value']:.3f}",
        },
    ]
)
dataframe_to_latex(
    pathway_table,
    OUTPUT_DIR / "table_pathway_trends.tex",
    (
        "Binomial trends in the probability of a Newfoundland-relevant "
        "pathway among eligible storms and among detected recurvers."
    ),
    "tab:pathway_trends",
)

location_proximity_table = pd.DataFrame(
    [
        {
            "Outcome": "Recurvature latitude",
            "Sample": "Relevant events",
            "Effect decade$^{-1}$": f"{latitude_trend['slope_per_decade']:.2f}$^\\circ$",
            "95\\% CI": (
                f"{latitude_trend['ci_low']:.2f}--"
                f"{latitude_trend['ci_high']:.2f}$^\\circ$"
            ),
            "$p$": f"{latitude_trend['p_value']:.3f}",
        },
        {
            "Outcome": "Recurvature longitude",
            "Sample": "Relevant events",
            "Effect decade$^{-1}$": f"{longitude_trend['slope_per_decade']:.2f}$^\\circ$",
            "95\\% CI": (
                f"{longitude_trend['ci_low']:.2f}--"
                f"{longitude_trend['ci_high']:.2f}$^\\circ$"
            ),
            "$p$": f"{longitude_trend['p_value']:.3f}",
        },
        {
            "Outcome": "Mean minimum distance",
            "Sample": "All recurvers",
            "Effect decade$^{-1}$": (
                f"{proximity_all_mean['slope_per_decade']:.1f} km"
            ),
            "95\\% CI": (
                f"{proximity_all_mean['ci_low']:.1f}--"
                f"{proximity_all_mean['ci_high']:.1f} km"
            ),
            "$p$": f"{proximity_all_mean['p_value']:.3f}",
        },
        {
            "Outcome": "Median minimum distance",
            "Sample": "All recurvers",
            "Effect decade$^{-1}$": (
                f"{proximity_all_median['median_slope_km_per_decade']:.1f} km"
            ),
            "95\\% CI": (
                f"{proximity_all_median['ci_low']:.1f}--"
                f"{proximity_all_median['ci_high']:.1f} km"
            ),
            "$p$": f"{proximity_all_median['p_value']:.3f}",
        },
    ]
)
dataframe_to_latex(
    location_proximity_table,
    OUTPUT_DIR / "table_location_proximity.tex",
    (
        "Trend estimates for recurvature-point location and minimum "
        "post-recurvature distance. Linear estimates use HC3 standard "
        "errors; the median-distance estimate uses quantile regression."
    ),
    "tab:location_proximity",
)

threshold_table = threshold_sensitivity[
    [
        "threshold_km",
        "events",
        "irr_per_decade",
        "ci_low",
        "ci_high",
        "p_value",
        "dispersion",
    ]
].copy()
threshold_table.columns = [
    "Threshold (km)",
    "$N$",
    "IRR decade$^{-1}$",
    "CI low",
    "CI high",
    "$p$",
    "Dispersion",
]
for column in ("IRR decade$^{-1}$", "CI low", "CI high", "$p$", "Dispersion"):
    threshold_table[column] = threshold_table[column].map(lambda value: f"{value:.3f}")
dataframe_to_latex(
    threshold_table,
    OUTPUT_DIR / "table_threshold_sensitivity.tex",
    (
        "Sensitivity of event classification and Poisson trend estimates "
        "to the Newfoundland-island proximity threshold."
    ),
    "tab:threshold_sensitivity",
)

satellite_threshold_table = pd.DataFrame(
    {
        "Threshold (km)": threshold_sensitivity["threshold_km"].astype(int),
        "$N$": threshold_sensitivity["satellite_events"].astype(int),
        "Count IRR (95\\% CI)": [
            (
                f"{row.satellite_irr_per_decade:.3f} "
                f"({row.satellite_ci_low:.3f}--"
                f"{row.satellite_ci_high:.3f})"
            )
            for row in threshold_sensitivity.itertuples(index=False)
        ],
        "$p$": threshold_sensitivity["satellite_p_value"].map(
            lambda value: f"{value:.3f}"
        ),
        "Eligible-storm OR (95\\% CI)": [
            (
                f"{row.satellite_eligible_or:.3f} "
                f"({row.satellite_eligible_ci_low:.3f}--"
                f"{row.satellite_eligible_ci_high:.3f})"
            )
            for row in threshold_sensitivity.itertuples(index=False)
        ],
        "Recurver OR (95\\% CI)": [
            (
                f"{row.satellite_recurver_or:.3f} "
                f"({row.satellite_recurver_ci_low:.3f}--"
                f"{row.satellite_recurver_ci_high:.3f})"
            )
            for row in threshold_sensitivity.itertuples(index=False)
        ],
    }
)
dataframe_to_latex(
    satellite_threshold_table,
    OUTPUT_DIR / "table_satellite_threshold_sensitivity.tex",
    (
        "Satellite-era sensitivity to the Newfoundland-island proximity "
        "threshold. Conditional odds ratios use the eligible-storm and "
        "detected-recurver denominators."
    ),
    "tab:satellite_threshold_sensitivity",
    resize_to_textwidth=True,
)

endpoint_table = pd.DataFrame(
    {
        "Sample": endpoint_distance_sensitivity["sample"],
        "$N$": endpoint_distance_sensitivity["n"].astype(int),
        "Mean slope (95\\% CI)": [
            (
                f"{row.mean_slope_km_per_decade:.1f} "
                f"({row.mean_ci_low:.1f}--{row.mean_ci_high:.1f})"
            )
            for row in endpoint_distance_sensitivity.itertuples(index=False)
        ],
        "Mean $p$": endpoint_distance_sensitivity["mean_p_value"].map(
            lambda value: f"{value:.3f}"
        ),
        "Median slope (95\\% CI)": [
            (
                f"{row.median_slope_km_per_decade:.1f} "
                f"({row.median_ci_low:.1f}--{row.median_ci_high:.1f})"
            )
            for row in endpoint_distance_sensitivity.itertuples(index=False)
        ],
        "Median $p$": endpoint_distance_sensitivity[
            "median_p_value"
        ].map(lambda value: f"{value:.3f}"),
    }
)
dataframe_to_latex(
    endpoint_table,
    OUTPUT_DIR / "table_endpoint_sensitivity.tex",
    (
        "Sensitivity of minimum-distance trends to post-recurvature tracks "
        "whose closest observed location is the final recorded point. "
        "Slopes are in km decade$^{-1}$."
    ),
    "tab:endpoint_sensitivity",
    resize_to_textwidth=True,
)

method_table = pd.DataFrame(
    [
        {
            "Comparison": "Velocity vs. heading",
            "$N_1$": method_overlap["first_n"],
            "$N_2$": method_overlap["second_n"],
            "Both": method_overlap["overlap"],
            "Only 1": method_overlap["only_first"],
            "Only 2": method_overlap["only_second"],
            "$J$": f"{method_overlap['jaccard']:.3f}",
        },
    ]
)
dataframe_to_latex(
    method_table,
    OUTPUT_DIR / "table_method_comparison.tex",
    "Event-list overlap between the baseline velocity and alternative heading detectors.",
    "tab:method_comparison",
)


peak = seasonality.loc[seasonality["events"].idxmax()]
aug_oct = int(
    seasonality.loc[seasonality["month"].isin([8, 9, 10]), "events"].sum()
)
aug_oct_pct = 100.0 * aug_oct / len(velocity_events)
ts_events = int(
    nature_counts.loc[nature_counts["nature_code"] == "TS", "events"].sum()
)
threshold_300 = threshold_sensitivity.loc[
    threshold_sensitivity["threshold_km"] == 300
].iloc[0]
threshold_1000 = threshold_sensitivity.loc[
    threshold_sensitivity["threshold_km"] == 1000
].iloc[0]
most_detectable_satellite_detector = detector_sensitivity.sort_values(
    "satellite_p_value"
).iloc[0]
full_record_decades = (YEAR_MAX - YEAR_MIN) / 10.0
satellite_record_decades = (YEAR_MAX - SATELLITE_START) / 10.0

macro_values = {
    "NorthAtlanticSourceN": len(north_atlantic_source_indices),
    "TropicalOriginSourceN": len(storm_indices),
    "NonTropicalExcludedN": (
        len(north_atlantic_source_indices) - len(storm_indices)
    ),
    "LegacyN": len(legacy_events),
    "BaselineN": len(velocity_events),
    "AllRecurverN": len(velocity_recurvers),
    "EligibleN": int(eligible_storms["eligible"].sum()),
    "HeadingN": len(heading_events),
    "DetectorOverlap": method_overlap["overlap"],
    "DetectorOnlyVelocity": method_overlap["only_first"],
    "DetectorOnlyHeading": method_overlap["only_second"],
    "DetectorDiscordantN": (
        method_overlap["only_first"] + method_overlap["only_second"]
    ),
    "DetectorJaccard": f"{method_overlap['jaccard']:.3f}",
    "LegacyOverlap": legacy_overlap["overlap"],
    "LegacyOnly": legacy_overlap["only_first"],
    "BaselineOnlyLegacyComparison": legacy_overlap["only_second"],
    "LegacyJaccard": f"{legacy_overlap['jaccard']:.3f}",
    "FullIRR": f"{full_count_summary['irr_per_decade']:.3f}",
    "FullCILow": f"{full_count_summary['ci_low']:.3f}",
    "FullCIHigh": f"{full_count_summary['ci_high']:.3f}",
    "FullP": f"{full_count_summary['p_value']:.3f}",
    "FullDispersion": f"{full_count_summary['pearson_dispersion']:.2f}",
    "FullHACCILow": f"{full_count_summary['hac_ci_low']:.3f}",
    "FullHACCIHigh": f"{full_count_summary['hac_ci_high']:.3f}",
    "FullHACP": f"{full_count_summary['hac_p_value']:.3f}",
    "FullLagOne": (
        f"{full_count_summary['lag1_residual_autocorrelation']:.3f}"
    ),
    "FullLjungBoxP": f"{full_count_summary['ljung_box_p_value']:.3f}",
    "FullRecordFactorLow": (
        f"{full_count_summary['ci_low'] ** full_record_decades:.2f}"
    ),
    "FullRecordFactorHigh": (
        f"{full_count_summary['ci_high'] ** full_record_decades:.2f}"
    ),
    "SatelliteN": satellite_count_summary["total_events"],
    "SatelliteIRR": f"{satellite_count_summary['irr_per_decade']:.3f}",
    "SatelliteCILow": f"{satellite_count_summary['ci_low']:.3f}",
    "SatelliteCIHigh": f"{satellite_count_summary['ci_high']:.3f}",
    "SatelliteP": f"{satellite_count_summary['p_value']:.3f}",
    "SatelliteDispersion": f"{satellite_count_summary['pearson_dispersion']:.2f}",
    "SatelliteHACCILow": f"{satellite_count_summary['hac_ci_low']:.3f}",
    "SatelliteHACCIHigh": f"{satellite_count_summary['hac_ci_high']:.3f}",
    "SatelliteHACP": f"{satellite_count_summary['hac_p_value']:.3f}",
    "SatelliteRecordFactorLow": (
        f"{satellite_count_summary['ci_low'] ** satellite_record_decades:.2f}"
    ),
    "SatelliteRecordFactorHigh": (
        f"{satellite_count_summary['ci_high'] ** satellite_record_decades:.2f}"
    ),
    "EligibleOR": f"{pathway_eligible_full['odds_ratio_per_decade']:.3f}",
    "EligibleORLow": f"{pathway_eligible_full['ci_low']:.3f}",
    "EligibleORHigh": f"{pathway_eligible_full['ci_high']:.3f}",
    "EligibleORP": f"{pathway_eligible_full['p_value']:.3f}",
    "RecurverOR": f"{pathway_recurver_full['odds_ratio_per_decade']:.3f}",
    "RecurverORLow": f"{pathway_recurver_full['ci_low']:.3f}",
    "RecurverORHigh": f"{pathway_recurver_full['ci_high']:.3f}",
    "RecurverORP": f"{pathway_recurver_full['p_value']:.3f}",
    "PeakMonth": str(peak["month_name"]),
    "PeakMonthN": int(peak["events"]),
    "PeakMonthPct": f"{peak['percent']:.1f}",
    "AugOctN": aug_oct,
    "AugOctPct": f"{aug_oct_pct:.1f}",
    "EarlyN": len(early_events),
    "LateN": len(late_events),
    "EarlyMedianLatitude": f"{early_events['recurv_lat'].median():.1f}",
    "LateMedianLatitude": f"{late_events['recurv_lat'].median():.1f}",
    "EarlyMedianLongitudeWest": (
        f"{abs(early_events['recurv_lon'].median()):.1f}"
    ),
    "LateMedianLongitudeWest": (
        f"{abs(late_events['recurv_lon'].median()):.1f}"
    ),
    "LatitudeSlope": f"{latitude_trend['slope_per_decade']:.2f}",
    "LatitudeLow": f"{latitude_trend['ci_low']:.2f}",
    "LatitudeHigh": f"{latitude_trend['ci_high']:.2f}",
    "LatitudeP": f"{latitude_trend['p_value']:.3f}",
    "LongitudeSlope": f"{longitude_trend['slope_per_decade']:.2f}",
    "LongitudeLow": f"{longitude_trend['ci_low']:.2f}",
    "LongitudeHigh": f"{longitude_trend['ci_high']:.2f}",
    "LongitudeP": f"{longitude_trend['p_value']:.3f}",
    "EnergyP": f"{location_energy['permutation_p_value']:.3f}",
    "DistanceMeanSlope": f"{proximity_all_mean['slope_per_decade']:.1f}",
    "DistanceMeanLow": f"{proximity_all_mean['ci_low']:.1f}",
    "DistanceMeanHigh": f"{proximity_all_mean['ci_high']:.1f}",
    "DistanceMeanP": f"{proximity_all_mean['p_value']:.3f}",
    "DistanceMedianSlope": (
        f"{proximity_all_median['median_slope_km_per_decade']:.1f}"
    ),
    "DistanceMedianLow": f"{proximity_all_median['ci_low']:.1f}",
    "DistanceMedianHigh": f"{proximity_all_median['ci_high']:.1f}",
    "DistanceMedianP": f"{proximity_all_median['p_value']:.3f}",
    "RelevantDistanceMeanSlope": (
        f"{proximity_relevant_mean['slope_per_decade']:.1f}"
    ),
    "RelevantDistanceMeanLow": f"{proximity_relevant_mean['ci_low']:.1f}",
    "RelevantDistanceMeanHigh": f"{proximity_relevant_mean['ci_high']:.1f}",
    "RelevantDistanceMeanP": f"{proximity_relevant_mean['p_value']:.3f}",
    "EarlyDistanceMedian": (
        f"{early_recurvers['minimum_distance_to_newfoundland_km'].median():.0f}"
    ),
    "LateDistanceMedian": (
        f"{late_recurvers['minimum_distance_to_newfoundland_km'].median():.0f}"
    ),
    "EndpointMinimumN": endpoint_summary["endpoint_count"],
    "EndpointMinimumPct": (
        f"{100.0 * endpoint_summary['endpoint_fraction']:.1f}"
    ),
    "EarlyEndpointPct": (
        f"{100.0 * endpoint_summary['early_endpoint_fraction']:.1f}"
    ),
    "LateEndpointPct": (
        f"{100.0 * endpoint_summary['late_endpoint_fraction']:.1f}"
    ),
    "NonEndpointN": len(non_endpoint_recurvers),
    "NonEndpointMeanSlope": (
        f"{proximity_non_endpoint_mean['slope_per_decade']:.1f}"
    ),
    "NonEndpointMeanLow": f"{proximity_non_endpoint_mean['ci_low']:.1f}",
    "NonEndpointMeanHigh": f"{proximity_non_endpoint_mean['ci_high']:.1f}",
    "NonEndpointMeanP": f"{proximity_non_endpoint_mean['p_value']:.3f}",
    "NonEndpointMedianSlope": (
        f"{proximity_non_endpoint_median['median_slope_km_per_decade']:.1f}"
    ),
    "NonEndpointMedianLow": (
        f"{proximity_non_endpoint_median['ci_low']:.1f}"
    ),
    "NonEndpointMedianHigh": (
        f"{proximity_non_endpoint_median['ci_high']:.1f}"
    ),
    "NonEndpointMedianP": (
        f"{proximity_non_endpoint_median['p_value']:.3f}"
    ),
    "ProjectionAlternativeN": projection_summary[
        "alternative_relevant_events"
    ],
    "ProjectionClassificationChanges": projection_summary[
        "classification_changes"
    ],
    "ProjectionDistanceCorrelation": (
        f"{projection_summary['distance_correlation']:.4f}"
    ),
    "ProjectionMedianAbsoluteDifference": (
        f"{projection_summary['median_absolute_difference_km']:.1f}"
    ),
    "ProjectionAlternativeMeanSlope": (
        f"{alternative_projection_mean_trend['slope_per_decade']:.1f}"
    ),
    "ProjectionAlternativeMeanLow": (
        f"{alternative_projection_mean_trend['ci_low']:.1f}"
    ),
    "ProjectionAlternativeMeanHigh": (
        f"{alternative_projection_mean_trend['ci_high']:.1f}"
    ),
    "ProjectionAlternativeMeanP": (
        f"{alternative_projection_mean_trend['p_value']:.3f}"
    ),
    "ThresholdLowN": int(threshold_300["events"]),
    "ThresholdHighN": int(threshold_1000["events"]),
    "ThresholdFullIRRMin": (
        f"{threshold_sensitivity['irr_per_decade'].min():.3f}"
    ),
    "ThresholdFullIRRMax": (
        f"{threshold_sensitivity['irr_per_decade'].max():.3f}"
    ),
    "ThresholdSatelliteIRRMin": (
        f"{threshold_sensitivity['satellite_irr_per_decade'].min():.3f}"
    ),
    "ThresholdSatelliteIRRMax": (
        f"{threshold_sensitivity['satellite_irr_per_decade'].max():.3f}"
    ),
    "ThresholdHighSatelliteN": int(threshold_1000["satellite_events"]),
    "ThresholdHighSatelliteIRR": (
        f"{threshold_1000['satellite_irr_per_decade']:.3f}"
    ),
    "ThresholdHighSatelliteLow": f"{threshold_1000['satellite_ci_low']:.3f}",
    "ThresholdHighSatelliteHigh": (
        f"{threshold_1000['satellite_ci_high']:.3f}"
    ),
    "ThresholdHighSatelliteP": (
        f"{threshold_1000['satellite_p_value']:.3f}"
    ),
    "DetectorEventMin": int(detector_sensitivity["events"].min()),
    "DetectorEventMax": int(detector_sensitivity["events"].max()),
    "DetectorFullIRRMin": (
        f"{detector_sensitivity['irr_per_decade'].min():.3f}"
    ),
    "DetectorFullIRRMax": (
        f"{detector_sensitivity['irr_per_decade'].max():.3f}"
    ),
    "DetectorSatelliteIRRMin": (
        f"{detector_sensitivity['satellite_irr_per_decade'].min():.3f}"
    ),
    "DetectorSatelliteIRRMax": (
        f"{detector_sensitivity['satellite_irr_per_decade'].max():.3f}"
    ),
    "DetectorSatelliteModelSignificantN": int(
        (detector_sensitivity["satellite_p_value"] < 0.05).sum()
    ),
    "DetectorSatelliteHACSignificantN": int(
        (detector_sensitivity["satellite_hac_p_value"] < 0.05).sum()
    ),
    "DetectorSensitiveLatitude": (
        f"{most_detectable_satellite_detector['latitude_gate_deg_n']:.0f}"
    ),
    "DetectorSensitiveWindow": int(
        most_detectable_satellite_detector["window_hours"]
    ),
    "DetectorSensitiveThreshold": (
        f"{most_detectable_satellite_detector['post_east_min_kmh']:.1f}"
    ),
    "DetectorSensitiveSatelliteN": int(
        most_detectable_satellite_detector["satellite_events"]
    ),
    "DetectorSensitiveSatelliteIRR": (
        f"{most_detectable_satellite_detector['satellite_irr_per_decade']:.3f}"
    ),
    "DetectorSensitiveSatelliteLow": (
        f"{most_detectable_satellite_detector['satellite_ci_low']:.3f}"
    ),
    "DetectorSensitiveSatelliteHigh": (
        f"{most_detectable_satellite_detector['satellite_ci_high']:.3f}"
    ),
    "DetectorSensitiveSatelliteP": (
        f"{most_detectable_satellite_detector['satellite_p_value']:.3f}"
    ),
    "TropicalAtTurnN": ts_events,
}
macro_lines = [
    f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macro_values.items()
]
(OUTPUT_DIR / "results_macros.tex").write_text("\n".join(macro_lines) + "\n")

print("Analysis complete.")
print(
    f"Baseline N={len(velocity_events)}; "
    f"IRR={full_count_summary['irr_per_decade']:.3f} "
    f"({full_count_summary['ci_low']:.3f}-"
    f"{full_count_summary['ci_high']:.3f})"
)
