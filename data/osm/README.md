# OSM Download Instructions

This folder is for year-wise OSM building data used outside Git.

## Why this folder exists

The raw building shapefiles are too large to store in the repository. GitHub rejects several of these files because the `.shp` and `.dbf` parts exceed the normal file-size limit.

Use this folder to keep the downloaded files locally in year-specific subfolders.

## Source

Download the required files from Geofabrik:

- https://download.geofabrik.de/europe/germany/bayern/mittelfranken.html#

## Expected local structure

```text
data/osm/
├── 2019/
├── 2020/
├── 2021/
└── README.md
```

Place each downloaded year's files inside the matching folder, for example:

```text
data/osm/2019/
data/osm/2020/
data/osm/2021/
```

## Recommended workflow

1. Open the Geofabrik Mittelfranken download page.
2. Download the OSM building dataset you want to use.
3. Extract the archive locally.
4. Copy the extracted shapefile set into the matching year folder under `data/osm/`.

A shapefile normally includes multiple companion files with the same basename, for example:

- `.shp`
- `.dbf`
- `.shx`
- `.prj`
- `.cpg`

Keep those files together in the same year folder.

## Example layout

```text
data/osm/2020/
├── gis_osm_buildings_a_free_1.shp
├── gis_osm_buildings_a_free_1.dbf
├── gis_osm_buildings_a_free_1.shx
├── gis_osm_buildings_a_free_1.prj
└── gis_osm_buildings_a_free_1.cpg
```

## Important note

These files are intentionally not meant to be committed to GitHub because they can exceed GitHub's file-size limits.

If needed, keep only the local copies in `data/osm/` and avoid `git add` on the large shapefile components.
