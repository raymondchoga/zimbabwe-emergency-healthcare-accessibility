"""
09_case_studies.py

Builds concrete, human-scale case studies: for a handful of representative
population points, reconstructs the actual baseline shortest-path route to
their nearest facility (not just the travel time), and checks which
specific edges along that route the flood makes impassable. This shows
exactly where a route gets severed, and -- because the underlying search is
exhaustive with no distance cap -- confirms whether any detour survives at
all.

Selection (edit DISTRICTS_OF_INTEREST / N_SURPRISE below to change): the
largest-population newly-cut-off point in each of a set of chronically
flood-prone districts, plus a "surprise" case -- the largest-population
point that was well-served at baseline but ends up newly cut off, outside
those named districts.

Inputs:
  data/processed/combined_status_drive.gpkg / combined_status_walk.gpkg
  data/processed/flood_accessibility_5km.gpkg
  data/processed/population_points_5km_2022.gpkg
  data/processed/health_facilities.gpkg
  .cache/{baseline,flood}_{drive,walk}_pred.npy   (route reconstruction)
  .cache/edges.parquet, edge_flood_depth.npy, node_coords.npy

Output:
  data/processed/case_study_routes.gpkg
    layer "case_study_routes"       6 baseline driving route lines
    layer "case_study_flood_cuts"   points marking every flooded edge on
                                     each of the 6 driving routes
    layer "case_study_routes_walk"       6 baseline walking route lines
    layer "case_study_flood_cuts_walk"   flooded edges on the walking routes
"""
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from shapely.geometry import LineString, Point

from utils_routing import reconstruct_route

DISTRICTS_OF_INTEREST = ["Centenary/ Muzarabani", "Kariba", "Chipinge", "Binga", "Tsholotsho"]


def build_route_and_cuts(pred, node_coords, edges, flood_depth, start_node,
                          edge_lookup, case_label, district, population, nearest_fac):
    path = reconstruct_route(pred, start_node)
    coords = [tuple(node_coords[n]) for n in path]
    line = LineString(coords) if len(coords) > 1 else None

    length_m = sum(
        ((coords[i][0] - coords[i + 1][0]) ** 2 + (coords[i][1] - coords[i + 1][1]) ** 2) ** 0.5
        for i in range(len(coords) - 1)
    )

    cuts = []
    n_flooded = 0
    for i in range(len(path) - 1):
        key = (path[i], path[i + 1])
        row = edge_lookup.get(key) or edge_lookup.get((path[i + 1], path[i]))
        if row is None:
            continue
        depth = flood_depth[row]
        if np.isfinite(depth):
            n_flooded += 1
            mx = (edges.at[row, "ax"] + edges.at[row, "bx"]) / 2
            my = (edges.at[row, "ay"] + edges.at[row, "by"]) / 2
            cuts.append((case_label, depth, Point(mx, my)))

    route_row = {
        "case_label": case_label, "district": district,
        "population_at_point": population, "nearest_facility": nearest_fac,
        "baseline_time_min": None, "baseline_dist_km": length_m / 1000.0,
        "n_flooded_segments_on_route": n_flooded, "geometry": line,
    }
    return route_row, cuts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flood-accessibility", default="data/processed/flood_accessibility_5km.gpkg")
    ap.add_argument("--points", default="data/processed/population_points_5km_2022.gpkg")
    ap.add_argument("--cache-dir", default=".cache")
    ap.add_argument("--out", default="data/processed/case_study_routes.gpkg")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    node_coords = np.load(cache / "node_coords.npy")
    edges = pd.read_parquet(cache / "edges.parquet").reset_index(drop=True)
    flood_depth = np.load(cache / "edge_flood_depth.npy")
    edge_lookup = {(int(r.a), int(r.b)): i for i, r in enumerate(edges.itertuples())}

    gdf = gpd.read_file(args.flood_accessibility, layer="flood_accessibility")
    points = gpd.read_file(args.points).to_crs("EPSG:32736")
    gdf["population"] = points["population"].values if "population" in points.columns else np.nan
    gdf["district"] = points["district"].values if "district" in points.columns else None

    # which graph node each population point snapped to -- recomputed the same
    # way 06/07 did, so route reconstruction starts from the right place
    from utils_routing import snap_points_to_graph
    from scipy.sparse import load_npz as _load
    graph_drive = _load(cache / "graph_drive.npz")
    pts_xy = np.column_stack([points.geometry.x, points.geometry.y])
    pts_node, _, _ = snap_points_to_graph(pts_xy, node_coords, graph_drive)
    gdf["_node"] = pts_node

    picks = []
    for district in DISTRICTS_OF_INTEREST:
        cand = gdf[(gdf["district"] == district) & (gdf["newly_cutoff_drive"])]
        if cand.empty:
            print(f"WARNING: no newly-cut-off driving point found in {district}; skipping")
            continue
        row = cand.loc[cand["population"].idxmax()]
        picks.append(("case", district, row))

    surprise_pool = gdf[
        (gdf["newly_cutoff_drive"]) &
        (gdf["drive_base_class"] == "well-served") &
        (~gdf["district"].isin(DISTRICTS_OF_INTEREST))
    ]
    if not surprise_pool.empty:
        row = surprise_pool.loc[surprise_pool["population"].idxmax()]
        picks.append(("surprise", row["district"], row))

    pred_drive = np.load(cache / "baseline_drive_pred.npy")
    pred_walk = np.load(cache / "baseline_walk_pred.npy")

    routes, cuts_all = [], []
    routes_w, cuts_all_w = [], []
    for kind, district, row in picks:
        label = f"{district}{' (surprise)' if kind == 'surprise' else ''}"
        r, c = build_route_and_cuts(pred_drive, node_coords, edges, flood_depth, int(row["_node"]),
                                     edge_lookup, label, district, row["population"], row["drive_base_fac"])
        r["baseline_time_min"] = row["drive_base_min"]
        routes.append(r); cuts_all.extend(c)

        rw, cw = build_route_and_cuts(pred_walk, node_coords, edges, flood_depth, int(row["_node"]),
                                       edge_lookup, label, district, row["population"], row["walk_base_fac"])
        rw["baseline_time_min"] = row["walk_base_min"]
        routes_w.append(rw); cuts_all_w.extend(cw)

        print(f"{label}: pop {row['population']:.0f}, drive {row['drive_base_min']:.1f} min "
              f"({r['baseline_dist_km']:.1f} km), {r['n_flooded_segments_on_route']} flooded segments "
              f"| walk {row['walk_base_min']:.1f} min, {rw['n_flooded_segments_on_route']} flooded segments "
              f"| drive_flood={row['drive_flood_time_min']}, walk_flood={row['walk_flood_time_min']}")

    routes_gdf = gpd.GeoDataFrame(routes, crs="EPSG:32736")
    cuts_gdf = gpd.GeoDataFrame(
        [{"case_label": c[0], "depth_m": c[1], "geometry": c[2]} for c in cuts_all], crs="EPSG:32736")
    routes_w_gdf = gpd.GeoDataFrame(routes_w, crs="EPSG:32736")
    cuts_w_gdf = gpd.GeoDataFrame(
        [{"case_label": c[0], "depth_m": c[1], "geometry": c[2]} for c in cuts_all_w], crs="EPSG:32736")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    routes_gdf.to_file(args.out, layer="case_study_routes", driver="GPKG")
    cuts_gdf.to_file(args.out, layer="case_study_flood_cuts", driver="GPKG")
    routes_w_gdf.to_file(args.out, layer="case_study_routes_walk", driver="GPKG")
    cuts_w_gdf.to_file(args.out, layer="case_study_flood_cuts_walk", driver="GPKG")
    print(f"\nWrote {len(routes_gdf)} driving + {len(routes_w_gdf)} walking case-study routes to {args.out}")


if __name__ == "__main__":
    main()
