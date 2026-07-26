"""Build a self-contained Google Colab notebook for the analysis."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "notebooks" / "01_track_climatology.ipynb"


def embedded_writer_cell(filename: str) -> nbf.NotebookNode:
    source = (HERE / filename).read_text()
    code = (
        "from pathlib import Path\n\n"
        f"source = {source!r}\n"
        f"path = PROJECT_ROOT / {filename!r}\n"
        "path.write_text(source)\n"
        "print(f\"Wrote {path.name} ({len(source):,} characters)\")"
    )
    return nbf.v4.new_code_cell(code)


notebook = nbf.v4.new_notebook()
notebook.metadata = {
    "colab": {
        "name": OUTPUT.name,
        "provenance": [],
    },
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3",
    },
}

notebook.cells = [
    nbf.v4.new_markdown_cell(
        """# Newfoundland recurvature climatology

This notebook reproduces the 1950–2023 track-climatology analysis and,
optionally, the satellite-era 500-hPa composite.

Principal workflow features:

- an explicit tropical-origin sample requiring an IBTrACS `TS` nature code at
  least once;
- regular 6-hour track interpolation without bridging source gaps longer than
  12 hours;
- a fixed-duration, physically constrained recurvature detector using 24-hour
  pre- and post-turn velocity windows;
- distance from the continuous post-turn track to the Newfoundland island
  polygon;
- separate models for event counts, conditional pathway probabilities,
  recurvature location, and the untruncated proximity distribution;
- full-period and satellite-era detector, temporal-window,
  velocity-threshold, and regional-distance sensitivity analyses;
- diagnostics for potential end-of-track distance censoring and alternative
  map projection;
- a 500-hPa anomaly composite with both fixed geographic and
  storm-relative coordinates.

The notebook downloads public IBTrACS and NCEP/NCAR Reanalysis data. The track
analysis is substantially faster than the optional composite, which must cache
one field for each satellite-era event."""
    ),
    nbf.v4.new_markdown_cell("## 1. Install and configure"),
    nbf.v4.new_code_cell(
        """import importlib.util
import subprocess
import sys

required = {
    "cartopy": "cartopy",
    "netCDF4": "netCDF4",
    "pyproj": "pyproj",
    "shapely": "shapely",
    "statsmodels": "statsmodels",
    "xarray": "xarray",
}
missing = [package for module, package in required.items()
           if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *missing]
    )
print("Dependencies ready")"""
    ),
    nbf.v4.new_code_cell(
        """import os
from pathlib import Path

if "COLAB_RELEASE_TAG" in os.environ:
    PROJECT_ROOT = Path("/content/newfoundland_tc_recurvature")
else:
    PROJECT_ROOT = Path(
        os.environ.get("NL_RECURV_WORKDIR", Path.cwd() / "nl_recurvature")
    ).resolve()

PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["NL_RECURV_WORKDIR"] = str(PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
print("Project root:", PROJECT_ROOT)"""
    ),
    nbf.v4.new_markdown_cell(
        """## 2. Materialize the audited analysis source

The three source modules are embedded in this notebook so that the notebook is
portable and does not depend on a separate repository checkout."""
    ),
    embedded_writer_cell("analysis_core.py"),
    embedded_writer_cell("run_analysis.py"),
    embedded_writer_cell("rebuild_composite.py"),
    nbf.v4.new_markdown_cell("## 3. Run the track analysis"),
    nbf.v4.new_code_cell(
        """import runpy

runpy.run_path(str(PROJECT_ROOT / "run_analysis.py"), run_name="__main__")"""
    ),
    nbf.v4.new_markdown_cell(
        """## 4. Rebuild the 500-hPa composite

Set `RUN_COMPOSITE = False` when only the track climatology is needed. With the
default `True`, this step downloads and caches the event-time NCEP fields and
recreates the composite."""
    ),
    nbf.v4.new_code_cell(
        """RUN_COMPOSITE = True

if RUN_COMPOSITE:
    runpy.run_path(
        str(PROJECT_ROOT / "rebuild_composite.py"),
        run_name="__main__",
    )
else:
    print("Composite skipped")"""
    ),
    nbf.v4.new_markdown_cell("## 5. Inspect the principal estimates"),
    nbf.v4.new_code_cell(
        """import json
import pandas as pd
from IPython.display import display

summary = json.loads(
    (PROJECT_ROOT / "outputs" / "analysis_summary.json").read_text()
)

display(pd.DataFrame(
    [
        {
            "period": "1950–2023",
            **summary["frequency"]["full_period"],
        },
        {
            "period": "1979–2023",
            **summary["frequency"]["satellite_era"],
        },
    ]
)[
    [
        "period",
        "total_events",
        "irr_per_decade",
        "ci_low",
        "ci_high",
        "p_value",
    ]
])

print(
    "Baseline relevant events:",
    summary["sample"]["corrected_relevant_events"],
)
print(
    "Tropical-origin source storms:",
    summary["sample"]["tropical_origin_storms"],
)
print(
    "Eligible storms:",
    summary["sample"]["eligible_storms"],
)
print(
    "All baseline recurvers:",
    summary["sample"]["all_corrected_recurvers"],
)"""
    ),
    nbf.v4.new_code_cell(
        """from IPython.display import Image, display

for filename in [
    "fig3_annual_counts_poisson.png",
    "fig5_recurvature_locations.png",
    "fig6_proximity_all_recurvers.png",
    "fig8_threshold_sensitivity.png",
    "fig9_z500_composite.png",
]:
    path = PROJECT_ROOT / "figures" / filename
    if path.exists():
        display(Image(filename=str(path), width=900))"""
    ),
    nbf.v4.new_markdown_cell("## 6. Package generated outputs"),
    nbf.v4.new_code_cell(
        """import shutil

archive = shutil.make_archive(
    str(PROJECT_ROOT / "newfoundland_recurvature_outputs"),
    "zip",
    root_dir=PROJECT_ROOT,
    base_dir="outputs",
)
print("Created:", archive)
print("Figures remain in:", PROJECT_ROOT / "figures")"""
    ),
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(f"Wrote {OUTPUT}")
