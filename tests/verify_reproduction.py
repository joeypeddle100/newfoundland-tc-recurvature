"""Verify the publication repository and its committed headline outputs."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


summary = json.loads((OUTPUTS / "analysis_summary.json").read_text())
sample = summary["sample"]

expected_counts = {
    "north_atlantic_source_storms": 1195,
    "tropical_origin_storms": 1161,
    "excluded_without_tropical_code": 34,
    "eligible_storms": 778,
    "all_corrected_recurvers": 417,
    "corrected_relevant_events": 144,
    "native_cadence_comparator_events": 165,
    "heading_relevant_events": 131,
}
for key, expected in expected_counts.items():
    require(sample[key] == expected, f"{key}: {sample[key]} != {expected}")

expected_frequency = {
    ("full_period", "total_events"): 144,
    ("full_period", "irr_per_decade"): 0.998935,
    ("full_period", "ci_low"): 0.925397,
    ("full_period", "ci_high"): 1.078317,
    ("satellite_era", "total_events"): 86,
    ("satellite_era", "irr_per_decade"): 1.109363,
    ("satellite_era", "ci_low"): 0.941915,
    ("satellite_era", "ci_high"): 1.306579,
}
for (period, key), expected in expected_frequency.items():
    actual = summary["frequency"][period][key]
    if isinstance(expected, int):
        require(actual == expected, f"{period}.{key}: {actual} != {expected}")
    else:
        require(
            math.isclose(actual, expected, abs_tol=5e-7),
            f"{period}.{key}: {actual} != {expected}",
        )

event_rows = csv_rows(OUTPUTS / "corrected_newfoundland_relevant_events.csv")
recurver_rows = csv_rows(OUTPUTS / "all_corrected_recurving_storms.csv")
source_rows = csv_rows(OUTPUTS / "source_scope_audit.csv")
require(len(event_rows) == 144, "Baseline event CSV does not contain 144 rows")
require(len(recurver_rows) == 417, "Recurver CSV does not contain 417 rows")
require(len(source_rows) == 1195, "Source-scope audit does not contain 1195 rows")
require(
    sum(row["ever_coded_tropical"].lower() == "true" for row in source_rows)
    == 1161,
    "Source-scope audit does not contain 1161 tropical-origin storms",
)
require(
    len({row["sid"] for row in event_rows}) == 144,
    "Baseline event identifiers are not unique",
)

required_figures = {
    "fig1_newfoundland_geometry.png",
    "fig2_seasonality.png",
    "fig3_annual_counts_poisson.png",
    "fig4_rolling_mean.png",
    "fig5_recurvature_locations.png",
    "fig6_proximity_all_recurvers.png",
    "fig7_proximity_ecdf.png",
    "fig8_threshold_sensitivity.png",
    "fig9_z500_composite.png",
    "figA1_detector_discordant_tracks.pdf",
}
for filename in required_figures:
    path = ROOT / "figures" / filename
    require(path.exists() and path.stat().st_size > 0, f"Missing {path}")

manuscript = ROOT / "manuscript.pdf"
require(manuscript.read_bytes()[:5] == b"%PDF-", "manuscript.pdf is not a PDF")

tex = (ROOT / "main.tex").read_text()
for match in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
    require((ROOT / match.group(1)).exists(), f"Missing TeX figure {match.group(1)}")

notebook_path = ROOT / "notebooks" / "01_track_climatology.ipynb"
notebook = json.loads(notebook_path.read_text())
require(notebook["nbformat"] == 4, "Unexpected notebook format")
require(len(notebook["cells"]) == 17, "Unexpected notebook cell count")

embedded = {}
for cell in notebook["cells"]:
    if cell["cell_type"] != "code":
        continue
    source_text = "".join(cell["source"])
    if "source =" not in source_text or "path = PROJECT_ROOT" not in source_text:
        continue
    tree = ast.parse(source_text)
    source_value = None
    path_value = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "source"
               for target in node.targets):
            source_value = ast.literal_eval(node.value)
        if any(isinstance(target, ast.Name) and target.id == "path"
               for target in node.targets):
            if isinstance(node.value, ast.BinOp):
                path_value = ast.literal_eval(node.value.right)
    if source_value is not None and path_value is not None:
        embedded[path_value] = source_value

for filename in ("analysis_core.py", "run_analysis.py", "rebuild_composite.py"):
    require(filename in embedded, f"Notebook does not embed {filename}")
    notebook_hash = hashlib.sha256(embedded[filename].encode()).hexdigest()
    source_hash = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
    require(notebook_hash == source_hash, f"Notebook copy of {filename} has drifted")

print("Repository verification passed")
print("1195 candidate / 1161 tropical-origin / 778 eligible")
print("417 recurvers / 144 regional events / 86 satellite-era events")
