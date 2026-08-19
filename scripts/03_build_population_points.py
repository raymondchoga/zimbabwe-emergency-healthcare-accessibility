"""
03_build_population_points.py

Converts the calibrated population raster into a population-weighted point
grid at a coarser, more tractable resolution (5km by default) -- these
points are the "communities" used throughout the rest of this analysis.

Chosen over the native ~1km WorldPop resolution (~480,000 points -- too
large to route against comfortably) and over ward/district centroids (too
coarse to show the flood-vs-baseline contrast that is the point of this
project). Since results are always reported relative to the named facility
a point is nearest to ("X hospital is hard to reach from this community"),
the grid points themselves don't need place names -- only a district label,
kept for grouping/filtering.

Input:
  data/processed/population_2022_calibrated_district.tif

Output:
  data/processed/population_points_5km_2022.gpkg
    columns: population, district, n_source_cells, geometry
"""
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import Point


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raster", default="data/processed/population_2022_calibrated_district.tif")
    ap.add_argument("--districts", default="data_raw/admin_boundaries/geoBoundaries-ZWE-ADM2.geojson")
    ap.add_argument("--district-name-field", default="shapeName")
    ap.add_argument("--grid-size-m", type=float, default=5000.0)
    ap.add_argument("--out", default="data/processed/population_points_5km_2022.gpkg")
    args = ap.parse_args()

    with rasterio.open(args.raster) as src:
        band = src.read(1)
        transform = src.transform
        raster_crs = src.crs

    rows, cols = np.where(band > 0)
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    pts = gpd.GeoDataFrame(
        {"pop": band[rows, cols]},
        geometry=[Point(x, y) for x, y in zip(xs, ys)],
        crs=raster_crs,
    ).to_crs("EPSG:32736")

    # snap each source cell to a 5km grid cell, sum population per grid cell
    gx = np.floor(pts.geometry.x / args.grid_size_m).astype(int)
    gy = np.floor(pts.geometry.y / args.grid_size_m).astype(int)
    pts["grid_key"] = list(zip(gx, gy))

    grouped = pts.groupby("grid_key").agg(
        population=("pop", "sum"),
        n_source_cells=("pop", "size"),
        x=("geometry", lambda g: g.x.mean()),
        y=("geometry", lambda g: g.y.mean()),
    ).reset_index(drop=True)

    grid_pts = gpd.GeoDataFrame(
        grouped[["population", "n_source_cells"]],
        geometry=[Point(x, y) for x, y in zip(grouped["x"], grouped["y"])],
        crs="EPSG:32736",
    )

    districts = gpd.read_file(args.districts).to_crs("EPSG:32736")
    grid_pts = gpd.sjoin(grid_pts, districts[[args.district_name_field, "geometry"]],
                          how="left", predicate="within")
    grid_pts = grid_pts.rename(columns={args.district_name_field: "district"}).drop(columns=["index_right"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    grid_pts[["population", "district", "n_source_cells", "geometry"]].to_file(args.out, driver="GPKG")

    print(f"{len(grid_pts):,} population points, total population "
          f"{grid_pts['population'].sum():,.0f}")
    n_no_district = grid_pts["district"].isna().sum()
    if n_no_district:
        print(f"NOTE: {n_no_district} points fell outside every district polygon "
              f"(likely just off the coastline/border of the source boundaries)")


if __name__ == "__main__":
    main()
