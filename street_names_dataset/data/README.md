# Street Names Data

This directory contains raw street name datasets and configuration files used for extracting, classifying, and analyzing U.S. street names.

## Directory Structure

```
data/
├── NYC/                      # NYC LION geodatabase (included)
│   └── original_data/
│       ├── lion.gdb/         # ESRI File Geodatabase
│       ├── *.lyr             # ArcGIS layer files
│       └── ReadMe.txt        # NYC Dept of City Planning documentation
├── language_clusters.json    # Language family groupings for classification
├── suffix_clusters.json      # USPS street suffix abbreviation mappings
└── usps_street_suffixes.csv  # USPS Publication 28 street suffix data
```

## City Data (Not Included)

The original street name datasets for each city are not included in this repository. To reproduce the analysis, download each city's data into `<City>/original_data/`.

All dataset sources were accessed in December 2025.

| City | File | Format | Source |
|------|------|--------|--------|
| Chicago | `Chicago_Street_Names_20251209.csv` | CSV | [Chicago Data Portal](https://data.cityofchicago.org/Transportation/Chicago-Street-Names/i6bp-fvbx/about_data) |
| Dallas | `SAN__STREET_LAYER.csv` | CSV | [Dallas GIS Hub](https://egisdata-dallasgis.hub.arcgis.com/datasets/e0cf1e436b014353a10c02eaebb20dfa_0/explore?location=32.816244%2C-96.771840%2C11.17) |
| Houston | `COH_RoadCenterline_*.csv` | CSV | [Houston Open Data](https://houston-mycity.opendata.arcgis.com/datasets/MyCity::houston-road-centerline/about) |
| Jacksonville | `jacksonville street names - Sheet1.csv` | Raw text | |
| Los Angeles | `Street_Names_20251209.csv` | CSV | [LA Open Data](https://data.lacity.org/City-Infrastructure-Service-Requests/Street-Names/hntu-mwxc/about_data) |
| NYC | `lion.gdb` (included) | ESRI File Geodatabase | [NYC Dept of City Planning](https://www.nyc.gov/content/planning/pages/resources/datasets/lion) |
| Philadelphia | `Street_Centerline.csv` | CSV | [OpenDataPhilly](https://opendataphilly.org/datasets/street-centerlines/) |
| Phoenix | `Street_Name_Labels.csv` | CSV | [Phoenix Open Data](https://www.phoenixopendata.com/dataset/street-name-labels) |
| San Antonio | `Streets_*.csv` | CSV | [San Antonio Open Data](https://data.sanantonio.gov/dataset/streets) |
| San Diego | `roads_datasd.csv` | CSV | [San Diego Open Data](https://data.sandiego.gov/datasets/roads-lines/) |
| San Jose | `Streets.csv` | CSV | [San Jose GIS](https://gisdata-csj.opendata.arcgis.com/datasets/CSJ::streets/explore) |
| San Francisco | `Street_Names_20251209.csv` | CSV | [DataSF](https://data.sfgov.org/Geographic-Locations-and-Boundaries/Street-Names/6d9h-4u5v/about_data) |


## Configuration Files

### `language_clusters.json`

Maps language family names to lists of specific languages. Used to group GPT-classified street name origins into broader categories (e.g., "Germanic" includes German, Dutch, Swedish, Norwegian, etc.).

### `suffix_clusters.json`

Maps canonical USPS street suffixes to all accepted abbreviations (e.g., `"ST"` maps to `["ST", "STR", "STRT", "STREET", "STREETS"]`). Used for normalizing street type suffixes during extraction.

### `usps_street_suffixes.csv`

USPS street suffix abbreviations from [Publication 28](https://pe.usps.com/text/pub28/28apc_002.htm). Contains columns for primary name, common abbreviations, and USPS standard abbreviation.
