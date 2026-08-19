"""
06_baseline_accessibility.py

Baseline (no flooding) accessibility: for every population point, find the
travel time and identity of the nearest health facility, for both driving
and walking. Uses a single multi-source Dijkstra search per mode (seeded
from all facilities simultaneously) rather than one search per facility.

Inputs:
  data/processed/health_facilities.gpkg          points, 1,486 facilities
  data/processed/population_points_5km_2022.gpkg points, 15,894 grid points
  .cache/graph_drive.npz, graph_walk.npz, node_coords.npy   (from step 05)

Output:
  data/processed/baseline_accessibility_5km_drive_and_walk.gpkg
    layer "baseline_accessibility", one row per population point:
      drive_time_min, drive_class, drive_nearest_facility
      walk_time_min,  walk_class,  walk_nearest_facility
      snap_dist_m     (population point's distance to the nearest road)
      access_gap      (walk class rank minus drive class rank; 0 = no gap,
                        higher = more dependent on vehicle access)

Also caches, for step 09 (case studies): the predecessor arrays needed to
reconstruct an actual shortest-path route from any point back to its
nearest facility (.cache/baseline_drive_pred.npy / _walk_pred.npy).
"""
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import sparse

from utils_routing import snap_points_to_graph, multi_source_travel_time, classify_time

CLASS_RANK = {"well-served": 0, "underserved": 1, "critical": 2}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--facilities", default="data/processed/health_facilities.gpkg")
    ap.add_argument("--points", default="data/processed/population_points_5km_2022.gpkg")
    ap.add_argument("--cache-dir", default=".cache")
    ap.add_argument("--out", default="data/processed/baseline_accessibility_5km_drive_and_walk.gpkg")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    graph_drive = sparse.load_npz(cache / "graph_drive.npz")
    graph_walk = sparse.load_npz(cache / "graph_walk.npz")
    node_coords = np.load(cache / "node_coords.npy")

    facilities = gpd.read_file(args.facilities).to_crs("EPSG:32736")
    points = gpd.read_file(args.points).to_crs("EPSG:32736")
    print(f"{len(facilities):,} facilities, {len(points):,} population points")

    fac_xy = np.column_stack([facilities.geometry.x, facilities.geometry.y])
    pts_xy = np.column_stack([points.geometry.x, points.geometry.y])

    fac_node, fac_snap_dist, _ = snap_points_to_graph(fac_xy, node_coords, graph_drive)
    pts_node, pts_snap_dist, pts_snap_time = snap_points_to_graph(pts_xy, node_coords, graph_drive)

    facility_name_col = "NAMEOFFACI" if "NAMEOFFACI" in facilities.columns else facilities.columns[0]
    fac_names = facilities[facility_name_col].astype(str).values
    node_to_facrow = {int(n): i for i, n in enumerate(fac_node)}

    out = points[["geometry"]].copy()
    out["snap_dist_m"] = pts_snap_dist

    for mode, graph, pred_name in [("drive", graph_drive, "baseline_drive_pred.npy"),
                                     ("walk", graph_walk, "baseline_walk_pred.npy")]:
        dist, nearest_source_node, predecessors = multi_source_travel_time(graph, fac_node)
        point_time = dist[pts_node] + pts_snap_time  # add off-road "last mile" at walking speed
        point_nearest_node = nearest_source_node[pts_node]
        point_nearest_fac = [fac_names[node_to_facrow.get(int(n), -1)] if int(n) in node_to_facrow else None
                              for n in point_nearest_node]

        out[f"{mode}_time_min"] = point_time
        out[f"{mode}_class"] = [classify_time(t) for t in point_time]
        out[f"{mode}_nearest_facility"] = point_nearest_fac

        np.save(cache / pred_name, predecessors)
        np.save(cache / f"baseline_{mode}_dist.npy", dist)

    out["access_gap"] = (out["walk_class"].map(CLASS_RANK) - out["drive_class"].map(CLASS_RANK))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_file(args.out, layer="baseline_accessibility", driver="GPKG")

    pop_col = "population" if "population" in points.columns else None
    if pop_col:
        out[pop_col] = points[pop_col].values
        total_pop = out[pop_col].sum()

    total = len(out)
    for mode in ["drive", "walk"]:
        counts = out[f"{mode}_class"].value_counts()
        print(f"\n{mode} baseline (by point count):")
        for cls in ["well-served", "underserved", "critical"]:
            n = counts.get(cls, 0)
            print(f"  {cls:14s} {n:6,} points  ({n/total:.1%})")
        if pop_col:
            pop_counts = out.groupby(f"{mode}_class")[pop_col].sum()
            print(f"{mode} baseline (by population, total {total_pop:,.0f}):")
            for cls in ["well-served", "underserved", "critical"]:
                p = pop_counts.get(cls, 0)
                print(f"  {cls:14s} {p:11,.0f} people  ({p/total_pop:.1%})")


if __name__ == "__main__":
    main()
