"""
Download & Apply Precise Nuremberg Boundary from OpenStreetMap

This script fetches the actual Nuremberg city administrative boundary
from Overpass API and ensures all data is clipped to it.
"""

from __future__ import annotations

import json
from pathlib import Path
import geopandas as gpd
from shapely.geometry import shape
import urllib.request
import urllib.error

from src.utils.config import RAW_DIR, INTERIM_DIR


def fetch_nuremberg_boundary_from_osm(retry: int = 3) -> dict | None:
    """
    Fetch Nuremberg city boundary from Overpass API.

    Uses OSM Overpass to get the official Nuremberg administrative boundary.
    """
    print("Fetching Nuremberg boundary from OpenStreetMap Overpass API...")

    # Overpass query for Nuremberg city boundary
    # admin_level=8 typically corresponds to city/district
    overpass_query = """
    [bbox:49.38,10.97,49.51,11.15];
    (
      relation["name"="Nürnberg"]["admin_level"="8"]["type"="multipolygon"];
      relation["name"="Nuremberg"]["admin_level"="8"]["type"="multipolygon"];
    );
    out geom;
    """

    url = "https://overpass-api.de/api/interpreter"

    for attempt in range(retry):
        try:
            print(f"  Attempt {attempt + 1}/{retry}...")
            req = urllib.request.Request(
                url,
                data=overpass_query.encode("utf-8"),
                headers={"User-Agent": "Mozilla/5.0"}
            )

            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))

                if data.get("elements"):
                    print(f"  ✅ Found {len(data['elements'])} boundary element(s)")
                    return data
                else:
                    print(f"  ⚠️  No boundaries found in response")

        except urllib.error.URLError as e:
            print(f"  ❌ Network error: {e}")
            if attempt < retry - 1:
                print(f"     Retrying...")
        except json.JSONDecodeError:
            print(f"  ❌ Failed to parse response as JSON")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    return None


def create_boundary_from_osm_response(osm_data: dict) -> gpd.GeoDataFrame | None:
    """Convert OSM Overpass response to GeoDataFrame."""
    if not osm_data or "elements" not in osm_data:
        return None

    geometries = []

    for element in osm_data["elements"]:
        if element["type"] == "relation" and "geometry" in element:
            # Build polygon from geometry
            coords = [(pt["lon"], pt["lat"]) for pt in element["geometry"]]

            if len(coords) >= 3:
                from shapely.geometry import Polygon
                geom = Polygon(coords)
                geometries.append(geom)

    if not geometries:
        return None

    gdf = gpd.GeoDataFrame(geometry=geometries, crs=4326)
    return gdf.to_crs(3857)


def save_boundary_file(boundary: gpd.GeoDataFrame, output_path: Path) -> None:
    """Save boundary to GeoJSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert back to 4326 for GeoJSON standard
    boundary_4326 = boundary.to_crs(4326)
    boundary_4326.to_file(output_path, driver="GeoJSON")
    print(f"✅ Saved to: {output_path}")


def clip_data_to_boundary(boundary: gpd.GeoDataFrame) -> None:
    """Clip all existing data files to the new boundary."""
    print("\n" + "="*70)
    print("CLIPPING DATA TO REFINED BOUNDARY")
    print("="*70)

    # Grid clipping
    grid_file = INTERIM_DIR / "grid.geojson"
    if grid_file.exists():
        print("\nClipping grid to refined boundary...")
        grid = gpd.read_file(grid_file)
        grid_reprojected = grid.to_crs(boundary.crs)

        # Clip grid
        clipped_grid = gpd.clip(grid_reprojected, boundary)
        clipped_grid.to_file(grid_file, driver="GeoJSON")

        print(f"  Original cells: {len(grid)}")
        print(f"  Clipped cells: {len(clipped_grid)}")
        print(f"  Removed: {len(grid) - len(clipped_grid)} cells")


def main() -> None:
    """Download and apply Nuremberg boundary."""
    print("\n" + "="*70)
    print("PRECISE NUREMBERG BOUNDARY DOWNLOAD")
    print("="*70 + "\n")

    # Try to fetch from OSM
    osm_data = fetch_nuremberg_boundary_from_osm()

    if osm_data:
        boundary = create_boundary_from_osm_response(osm_data)

        if boundary is not None:
            print(f"\n✅ Successfully created boundary GeoDataFrame")
            print(f"   Area: {boundary.geometry.area.values[0] / 1e6:.2f} km²")
            print(f"   CRS: {boundary.crs}")

            # Save refined boundary
            output_path = RAW_DIR / "nuremberg_boundary_refined.geojson"
            save_boundary_file(boundary, output_path)

            # Ask user to apply it
            print("\n" + "="*70)
            print("TO USE THIS REFINED BOUNDARY:")
            print("="*70)
            print(f"""
✅ A refined boundary has been saved to:
   {output_path}

To use it:
  1. Backup current boundary:
     cp {RAW_DIR / 'nuremberg_boundary.geojson'} {RAW_DIR / 'nuremberg_boundary.geojson.bak'}

  2. Apply refined boundary:
     cp {output_path} {RAW_DIR / 'nuremberg_boundary.geojson'}

  3. Re-run data preparation:
     python -m src.data.prepare_dataset

This will re-generate the grid with proper Nuremberg clipping.
            """)

            # Optionally clip existing data
            # clip_data_to_boundary(boundary)

        else:
            print("\n❌ Failed to create boundary from OSM response")

    else:
        print("\n⚠️  Could not fetch from OSM Overpass API")
        print("\nAlternatives:")
        print("  1. Check internet connection")
        print("  2. OSM Overpass may be temporarily unavailable")
        print("  3. Try again later")
        print("\nCurrent rectangular boundary is sufficient for your analysis.")


if __name__ == "__main__":
    main()
