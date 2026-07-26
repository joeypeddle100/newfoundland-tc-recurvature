# Newfoundland-relevant tropical-cyclone recurvature

[![Repository validation](https://github.com/joeypeddle100/newfoundland-tc-recurvature/actions/workflows/validate.yml/badge.svg)](https://github.com/joeypeddle100/newfoundland-tc-recurvature/actions/workflows/validate.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joeypeddle100/newfoundland-tc-recurvature/blob/main/notebooks/01_track_climatology.ipynb)

This repository contains the manuscript, executable analysis, derived event
lists, sensitivity results, and figures for:

> **Climatology of North Atlantic Tropical-Cyclone Recurvature Relevant to
> Newfoundland: Frequency, Location, and Proximity, 1950–2023**

The study uses IBTrACS v04r00 tracks on a regular 6-hour grid. Recurvature is
defined as a sustained transition from non-eastward translation to eastward
and poleward translation using fixed 24-hour windows. Regional relevance is
based on the minimum distance between the continuous post-recurvature track
and the Newfoundland island polygon.

The 600-km classification is a broad track-proximity screen. It is not a
landfall, impact, or hazard record.

## Principal results

The publication workflow should reproduce:

| Quantity | Expected value |
|---|---:|
| North Atlantic candidate source storms | 1,195 |
| Storms coded tropical at least once | 1,161 |
| Eligible storms | 778 |
| Baseline detected recurvers | 417 |
| Newfoundland-relevant events at 600 km | 144 |
| Full-period count IRR per decade | 0.999 |
| Full-period 95% CI | 0.925–1.078 |
| Satellite-era count IRR per decade | 1.109 |
| Satellite-era 95% CI | 0.942–1.307 |

The full-period result does not resolve a monotonic frequency trend. The
satellite-era raw-count result is positive but sensitive to reasonable
detector and proximity definitions. Conditional pathway models remain
unresolved across the tested definitions.

## Repository contents

```text
analysis_core.py              Track processing, detectors, geometry, statistics
run_analysis.py               Complete track analysis and figure/table build
rebuild_composite.py          Optional satellite-era 500-hPa composite
make_notebook.py              Rebuilds the self-contained Colab notebook
notebooks/                    Portable executable notebook
outputs/                      Derived event lists, diagnostics, and tables
figures/                      Manuscript figures
tests/verify_reproduction.py  Repository and headline-result checks
main.tex                      Manuscript source
manuscript.pdf                Current compiled manuscript
data/README.md                Input data versions, URLs, and checksums
```

## Run in Google Colab

Open
[`notebooks/01_track_climatology.ipynb`](notebooks/01_track_climatology.ipynb)
or use the Colab badge above, then run all cells.

The notebook is self-contained: it materializes the audited source modules and
downloads the public inputs. Set `RUN_COMPOSITE = False` to reproduce only the
track climatology without downloading the 86 event-time reanalysis fields.

## Run locally

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python run_analysis.py
python rebuild_composite.py
python tests/verify_reproduction.py

pdflatex main.tex
pdflatex main.tex
```

The track analysis generally completes within a few minutes. The optional
composite takes longer because it retrieves and caches one NCEP/NCAR
Reanalysis field for each of 86 satellite-era events.

Input and cache locations can be overridden with:

- `IBTRACS_PATH`
- `CARTOPY_DATA_DIR`
- `NCEP_EVENT_CACHE`
- `NCEP_HGT_CLIMATOLOGY`
- `NL_RECURV_WORKDIR`

## Derived products

The committed `outputs/` directory allows the reported results to be audited
without rerunning external downloads. It includes:

- the 144-event baseline list;
- all 417 baseline recurvers;
- source-scope and time-step audits;
- annual counts and pathway denominators;
- alternative-detector and native-cadence comparator event lists;
- detector, threshold, projection, and endpoint sensitivities;
- generated LaTeX tables and numerical macros; and
- the composite fields and event metadata.

Run the integrity check at any time:

```bash
python tests/verify_reproduction.py
```

## Reproducibility boundary

The repository does not redistribute the raw IBTrACS archive, Cartopy/Natural
Earth cache, monthly NCEP climatology, or per-event NCEP cache. The analysis
downloads these public products as needed. Exact source URLs and checksums for
the principal input files are recorded in [`data/README.md`](data/README.md).

The detector comparison is algorithmic rather than an independent
expert-labelled validation. The 500-hPa composite is illustrative and does
not contain a matched non-Newfoundland control population. These limitations
are stated in the manuscript and should be preserved when reusing the
results.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). A versioned
archival DOI can be added after the submission release is deposited.

## License

The analysis software is released under the [MIT License](LICENSE).
Manuscript text and figures remain copyright © 2026 Joey Peddle pending the
terms of the eventual journal publication.
