"""
08_combined_status.py

Combines each population point's baseline and flood-scenario travel-time
class (for one travel mode) into a single four-category status, so the
before/after picture can be read off one map instead of compared across two
separate ones. This is what the two report maps (driving, walking) are
symbolised by.

Categories:
  Well-served (unaffected by flood)      well-served at baseline AND flood
  Persistently underserved/critical      underserved/critical at baseline,
                                          and no worse (or already at the
                                          floor) during the flood
  Degraded by flood (not cut off)        still reachable during the flood,
                                          but the travel-time class worsens
  Newly cut off by flood                 reachable at baseline, completely
                                          unreachable during the flood
  Always cut off                         unreachable in both baseline and
                                          flood (present in the schema; in
                                          practice ~0 population)

Input:
  data/processed/flood_accessibility_5km.gpkg  (layer "flood_accessibility",
  produced by 07_flood_scenario.py -- already carries both baseline and
  flood-scenario results for both modes, plus the newly_cutoff_drive /
  newly_cutoff_walk flags used directly below)

Output:
  data/processed/combined_status_drive.gpkg  (layer "combined_status")
  data/processed/combined_status_walk.gpkg   (layer "combined_status")
"""
import argparse
from pathlib import Path

import numpy as np
import geopandas as gpd

RANK = {"well-served": 0, "underserved": 1, "critical": 2}


def classify(gdf, mode):
    base_class = gdf[f"{mode}_base_class"].values
    flood_class = gdf[f"{mode}_flood_class"].values
    base_reachable = np.isfinite(gdf[f"{mode}_base_min"].values)
    flood_reachable = np.isfinite(gdf[f"{mode}_flood_time_min"].values)
    newly_cutoff = gdf[f"newly_cutoff_{mode}"].values.astype(bool)

    status = np.empty(len(gdf), dtype=object)
    for i in range(len(gdf)):
        if newly_cutoff[i]:
            status[i] = "Newly cut off by flood"
        elif not base_reachable[i] and not flood_reachable[i]:
            status[i] = "Always cut off"
        elif base_class[i] == "well-served" and flood_class[i] == "well-served":
            status[i] = "Well-served (unaffected by flood)"
        elif RANK[flood_class[i]] > RANK[base_class[i]]:
            status[i] = "Degraded by flood (not cut off)"
        else:
            status[i] = "Persistently underserved/critical"
    return status


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flood-accessibility", default="data/processed/flood_accessibility_5km.gpkg")
    ap.add_argument("--out-prefix", default="data/processed/combined_status")
    args = ap.parse_args()

    gdf = gpd.read_file(args.flood_accessibility, layer="flood_accessibility")

    for mode in ["drive", "walk"]:
        status = classify(gdf, mode)

        out = gdf[["geometry"]].copy()
        if "population" in gdf.columns:
            out["population"] = gdf["population"]
        if "district" in gdf.columns:
            out["district"] = gdf["district"]
        out[f"{mode}_combined_status"] = status
        out["baseline_min"] = gdf[f"{mode}_base_min"]
        out["flood_min"] = gdf[f"{mode}_flood_time_min"]

        out_path = f"{args.out_prefix}_{mode}.gpkg"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_file(out_path, layer="combined_status", driver="GPKG")

        print(f"\n{mode} combined status -> {out_path}")
        counts = out[f"{mode}_combined_status"].value_counts()
        print(counts.to_string())
        if "population" in out.columns:
            pop_counts = out.groupby(f"{mode}_combined_status")["population"].sum()
            print(pop_counts.round(0).to_string())


if __name__ == "__main__":
    main()
