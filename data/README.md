# Input data

Raw public datasets and download caches are intentionally not committed to the
repository. The analysis scripts create this directory and download the files
below when they are absent.

## IBTrACS

- Product: International Best Track Archive for Climate Stewardship
- Version: v04r00
- File: `IBTrACS.ALL.v04r00.nc`
- URL:
  `https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/netcdf/IBTrACS.ALL.v04r00.nc`
- SHA-256 of the file used for the publication analysis:
  `6fa86054a0723017f6a74fc8ca225cd9bea1090baf864b7634cb1f11dcb3aee7`

## NCEP/NCAR Reanalysis 1

Monthly 1991–2020 500-hPa geopotential-height climatology:

- File: `hgt.mon.ltm.1991-2020.nc`
- URL:
  `https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.derived/pressure/hgt.mon.ltm.1991-2020.nc`
- SHA-256 of the file used for the publication analysis:
  `cc49de28e07c786878df4c1c9e9c7ac5f9ec4a7cd99abed517e3f86d54551b81`

Event-time 500-hPa fields are requested from NOAA PSL's NetCDF Subset
Service using the year and diagnosed recurvature time. They are cached under
`data/ncep_event_fields/`. The exact 86 event identifiers and times are
recorded in `outputs/z500_composite_metadata.json`.

## Natural Earth geometry

Cartopy downloads the Natural Earth 1:10-million administrative geometry used
to construct the Newfoundland island polygon. Its cache is stored under
`data/cartopy/`.

## Derived data

All small derived event lists, sensitivity tables, audit files, and composite
fields needed to inspect the reported results are committed under `outputs/`.
