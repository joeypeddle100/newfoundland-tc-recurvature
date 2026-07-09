# newfoundland-tc-recurvature
Track-based climatology of Newfoundland-relevant North Atlantic tropical cyclone recurvature, 1950–2023.

# Newfoundland-Relevant Tropical Cyclone Recurvature

This repository contains the analysis code and derived outputs for a track-based climatology of North Atlantic tropical cyclone recurvature relevant to Newfoundland, 1950–2023.

## Project overview

The study uses IBTrACS v04r00 tropical cyclone track data to identify recurvature points using an objective zonal-translation criterion expressed in physical distance per analyzed track increment. Newfoundland relevance is defined using post-recurvature intersection with a Newfoundland envelope and/or proximity within 600 km of coastal proxy points.

The analysis includes:

- baseline Newfoundland-relevant recurvature classification,
- annual count and Poisson trend analysis,
- sensitivity tests for proximity thresholds and recurvature-detection parameters,
- seasonal timing of diagnosed recurvature,
- a limited satellite-era 500-hPa physical-context composite using NCEP/NCAR Reanalysis.

## Repository structure

```text
notebooks/
  01_track_climatology_main.ipynb
  02_physical_context_composite.ipynb

data/
  baseline_zonal_events_600km.csv
  seasonality_monthly_counts.csv
  physical_context_composite_event_metadata_ncep.csv

figures/



requirements.txt
