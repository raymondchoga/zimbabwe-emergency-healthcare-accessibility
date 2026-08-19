"""
01_prepare_population.py

Calibrates WorldPop's 2020 gridded population density to Zimbabwe's 2022
census, at district level, so downstream per-point population figures sum
exactly to the census total rather than WorldPop's own 2020 model estimate.

Method: for each of the 91 geoBoundaries ADM2 districts, sum the WorldPop
raster cells falling inside it, compute a calibration factor
(census_district_total / worldpop_district_total), and multiply every cell
in that district by its own factor. The raster is kept in its native
EPSG:4326 grid throughout -- reprojecting a population raster to a metres
CRS and back would lose population mass to resampling (an earlier,
now-superseded province-level pass in this project measured ~19% mass loss
doing exactly that).

Known caveat carried through from the original analysis: Mvurwi's
calibration factor comes out around 7.85x, an outlier far outside the
0.66-1.43x range seen elsewhere. Its geoBoundaries polygon is only 2.8 km2,
covered by just 2 native WorldPop cells -- too coarse a base to trust the
within-district *pattern* for Mvurwi specifically, even though the
district *total* is still exactly correct against the census. If you rerun
this against updated source data, check for other very small districts with
similarly few covering cells; the calibration factor itself is a red flag
worth eyeballing (anything far outside the ~0.5-1.5x range deserves a look).

Inputs:
  data_raw/population/zwe_pd_2020_1km_ASCII_XYZ.csv   WorldPop 2020, 1km, XYZ
  data_raw/admin_boundaries/geoBoundaries-ZWE-ADM2.geojson   91 districts
  data/processed/population_census_2022_by_district.csv
    (district, census_population -- parsed from ZimStat's official report;
    parsing that PDF is a separate, one-off step not scripted here since it
    needed manual table-layout handling; see README)

Output:
  data/processed/population_2022_calibrated_district.tif
  data/processed/population_by_district_summary.csv  (modelled vs. census
    totals and the calibration factor applied, per district)
"""
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
from rasterio.transform import from_origin


# Known geoBoundaries-name -> census-row-name mismatches from the original
# analysis. geoBoundaries names a district "Centenary" where the census
# reports it jointly as "Muzarabani District"; "Kadoma Urban" needs to match
# the census's "Kadoma District" (which is not urban/rural-split there); and
# geoBoundaries splits Harare into two overlapping polygons ("Harare" +
# a small "Harare Rural" sliver) that the census counts as one "Harare Urban
# District" figure -- both should share that single total via a joint
# calibration rather than each pulling the whole figure (which would double-
# count). This script applies the joint-Harare rule; other "Urban"-suffixed
# geoBoundaries districts that don't cleanly match a census row (see the
# outlier warnings at the end of a run) reflect the same class of census-vs-
# geoBoundaries structural mismatch and need similar manual crosswalk entries
# added here if you want them calibrated precisely rather than left as-is.
MANUAL_CROSSWALK = {
    "Centenary/ Muzarabani": "Muzarabani District",
    "Kadoma Urban": "Kadoma District",
}
JOINT_CALIBRATION_GROUPS = [["Harare", "Harare Rural"]]  # share one census total, one factor


def load_worldpop_xyz(path):
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"z": "pop"})[["x", "y", "pop"]]
    df = df[df["pop"] > 0]
    return df


def xyz_to_raster(df, cell_size_deg):
    xs = np.sort(df["x"].unique())
    ys = np.sort(df["y"].unique())[::-1]
    x_idx = {v: i for i, v in enumerate(xs)}
    y_idx = {v: i for i, v in enumerate(ys)}
    grid = np.zeros((len(ys), len(xs)), dtype="float32")
    for x, y, pop in df[["x", "y", "pop"]].itertuples(index=False):
        grid[y_idx[y], x_idx[x]] = pop
    transform = from_origin(xs.min() - cell_size_deg / 2, ys.max() + cell_size_deg / 2,
                             cell_size_deg, cell_size_deg)
    return grid, transform


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worldpop-xyz", default="data_raw/population/zwe_pd_2020_1km_ASCII_XYZ.csv")
    ap.add_argument("--districts", default="data_raw/admin_boundaries/geoBoundaries-ZWE-ADM2.geojson")
    ap.add_argument("--census", default="data/processed/population_census_2022_by_district.csv")
    ap.add_argument("--census-district-field", default="district")
    ap.add_argument("--census-population-field", default="total")
    ap.add_argument("--district-name-field", default="shapeName")
    ap.add_argument("--cell-size-deg", type=float, default=1 / 120)  # ~1km at the equator
    ap.add_argument("--out-raster", default="data/processed/population_2022_calibrated_district.tif")
    ap.add_argument("--out-summary", default="data/processed/population_by_district_summary.csv")
    args = ap.parse_args()

    wp = load_worldpop_xyz(args.worldpop_xyz)
    grid, transform = xyz_to_raster(wp, args.cell_size_deg)

    districts = gpd.read_file(args.districts).to_crs("EPSG:4326")
    census = pd.read_csv(args.census)

    calibrated = grid.copy()
    summary_rows = []

    # cell centroid coordinates for a fast point-in-polygon pass
    height, width = grid.shape
    xs = transform.c + (np.arange(width) + 0.5) * transform.a
    ys = transform.f + (np.arange(height) + 0.5) * transform.e
    xx, yy = np.meshgrid(xs, ys)

    cdf, cpf = args.census_district_field, args.census_population_field
    joint_members = {m for group in JOINT_CALIBRATION_GROUPS for m in group}

    # pre-compute one shared factor for each joint-calibration group (e.g.
    # Harare + Harare Rural split geoBoundaries polygons, one census total)
    joint_factor = {}
    for group in JOINT_CALIBRATION_GROUPS:
        group_modeled = 0.0
        for _, row in districts[districts[args.district_name_field].isin(group)].iterrows():
            mask = rasterio.features.geometry_mask(
                [row.geometry], out_shape=grid.shape, transform=transform, invert=True)
            group_modeled += grid[mask].sum()
        lookup_name = MANUAL_CROSSWALK.get(group[0], group[0])
        crow = census[census[cdf].str.contains(lookup_name, case=False, na=False)]
        if not crow.empty and group_modeled > 0:
            f = float(crow.iloc[0][cpf]) / group_modeled
            for m in group:
                joint_factor[m] = (f, float(crow.iloc[0][cpf]))

    for _, row in districts.iterrows():
        name = row[args.district_name_field]
        mask = rasterio.features.geometry_mask(
            [row.geometry], out_shape=grid.shape, transform=transform, invert=True)
        modeled_pop = grid[mask].sum()

        if name in joint_members:
            factor, census_pop = joint_factor.get(name, (None, None))
            if factor is None:
                print(f"WARNING: joint-calibration group member '{name}' had no census match")
                continue
        else:
            # crosswalk note: geoBoundaries district names don't always match
            # the census table exactly -- try the manual crosswalk first,
            # then a short-name substring match; anything still unmatched
            # needs a manual crosswalk entry added above (logged as a
            # warning, not silently skipped -- see the outlier list printed
            # at the end of a run for other likely candidates).
            lookup_name = MANUAL_CROSSWALK.get(name, name)
            short_name = lookup_name.replace(" District", "").replace(" Urban", "").replace(" Rural", "").strip()
            census_row = census[census[cdf].str.contains(short_name, case=False, na=False)]
            if census_row.empty:
                print(f"WARNING: no census match for district '{name}' -- left uncalibrated "
                      f"(add a manual crosswalk entry)")
                continue
            census_pop = float(census_row.iloc[0][cpf])
            factor = census_pop / modeled_pop if modeled_pop > 0 else 0.0

        calibrated[mask] = grid[mask] * factor
        summary_rows.append({
            "district": name, "worldpop_2020_modeled": modeled_pop,
            "census_2022": census_pop, "calibration_factor": factor,
            "n_worldpop_cells": int(mask.sum()),
        })

    summary = pd.DataFrame(summary_rows)
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_summary, index=False)

    with rasterio.open(
        args.out_raster, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=0,
    ) as dst:
        dst.write(calibrated, 1)

    print(f"Calibrated {len(summary)} districts; total population "
          f"{calibrated.sum():,.0f} (target: {summary['census_2022'].sum():,.0f})")
    outliers = summary[(summary["calibration_factor"] > 2) | (summary["calibration_factor"] < 0.5)]
    if not outliers.empty:
        print("\nCalibration-factor outliers worth a manual look:")
        print(outliers[["district", "calibration_factor", "n_worldpop_cells"]].to_string(index=False))


if __name__ == "__main__":
    main()
