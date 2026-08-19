"""
07_flood_scenario.py

Models a 1-in-50-year river flood (RP50) scenario: samples flood depth at
the midpoint of every routing-graph edge, applies a depth-based passability
rule (Table 2 in the report), rebuilds the driving and walking graphs with
flood-affected edges penalized or removed, and re-runs the same
multi-source Dijkstra search from 06_baseline_accessibility.py against the
flood-penalized graphs.

The flood raster is left in its native CRS (EPSG:4326) and edge midpoints
are reprojected to match it for sampling, rather than reprojecting the
raster itself -- this avoids the resampling artifacts a raster reprojection
would introduce.

Flood-depth passability rule (a modelling assumption, stated explicitly
rather than treated as fact):
  < 0.3m            passable, heavy penalty  (drive capped 10 km/h, walk 1.5 km/h)
  0.3m - 1.0m       impassable by vehicle; walkable, capped at 1.0 km/h
  >= 1.0m           impassable for both modes; edge removed from both graphs

Inputs:
  data/processed/flood_hazard_RP50_depth.tif
  data/processed/health_facilities.gpkg, population_points_5km_2022.gpkg
  data/processed/baseline_accessibility_5km_drive_and_walk.gpkg (for the
    baseline figures the flood scenario is compared against)
  .cache/graph_drive.npz, graph_walk.npz, node_coords.npy, edges.parquet

Outputs:
  data/processed/flood_accessibility_5km.gpkg   layer "flood_accessibility"
  data/processed/flooded_roads_RP50.gpkg        layer "flooded_roads"
  .cache/graph_drive_flood.npz, graph_walk_flood.npz, edge_flood_depth.npy
"""
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy import sparse
from shapely.geometry import LineString

from utils_routing import snap_points_to_graph, multi_source_travel_time, classify_time

DEPTH_LIGHT = 0.3   # m -- below this: heavy penalty, still passable
DEPTH_SEVERE = 1.0  # m -- at/above this: impassable for both modes

DRIVE_PENALTY_KMH = 10.0
WALK_PENALTY_LIGHT_KMH = 1.5
WALK_PENALTY_MODERATE_KMH = 1.0


def sample_flood_depth(raster_path, xs_32736, ys_32736):
    transformer = Transformer.from_crs("EPSG:32736", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(xs_32736, ys_32736)
    with rasterio.open(raster_path) as src:
        inv = ~src.transform
        depths = np.full(len(xs_32736), np.nan)
        band = src.read(1)
        nodata = src.nodata
        for i, (x, y) in enumerate(zip(lon, lat)):
            col, row = inv * (x, y)
            row, col = int(row), int(col)
            if 0 <= row < band.shape[0] and 0 <= col < band.shape[1]:
                v = band[row, col]
                if nodata is None or v != nodata:
                    depths[i] = v
    return depths


def severity_label(depth):
    if depth < DEPTH_LIGHT:
        return "light"
    if depth < DEPTH_SEVERE:
        return "moderate"
    return "severe"


def penalize_edges(edges, depth, mode):
    """Return (minutes, keep_mask) for the given mode under the flood rule."""
    minutes = edges[f"{'minutes' if mode == 'drive' else 'walk_minutes'}"].values.copy()
    keep = np.ones(len(edges), dtype=bool)
    length_km = edges["length_m"].values / 1000.0

    light = depth < DEPTH_LIGHT
    moderate = (depth >= DEPTH_LIGHT) & (depth < DEPTH_SEVERE)
    severe = depth >= DEPTH_SEVERE

    flooded = light | moderate | severe
    # only edges that actually intersect the flood extent (finite sampled depth) are touched
    flooded &= np.isfinite(depth)
    light &= flooded; moderate &= flooded; severe &= flooded

    if mode == "drive":
        minutes[light] = length_km[light] / DRIVE_PENALTY_KMH * 60.0
        keep[moderate] = False
        keep[severe] = False
    else:
        minutes[light] = length_km[light] / WALK_PENALTY_LIGHT_KMH * 60.0
        minutes[moderate] = length_km[moderate] / WALK_PENALTY_MODERATE_KMH * 60.0
        keep[severe] = False

    return minutes, keep


def build_penalized_graph(edges, minutes, keep_mask, n_nodes):
    a = edges["a"].values[keep_mask]
    b = edges["b"].values[keep_mask]
    w = minutes[keep_mask]
    g = sparse.csr_matrix((w, (a, b)), shape=(n_nodes, n_nodes))
    return g.maximum(g.T)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flood-raster", default="data/processed/flood_hazard_RP50_depth.tif")
    ap.add_argument("--facilities", default="data/processed/health_facilities.gpkg")
    ap.add_argument("--points", default="data/processed/population_points_5km_2022.gpkg")
    ap.add_argument("--baseline", default="data/processed/baseline_accessibility_5km_drive_and_walk.gpkg")
    ap.add_argument("--cache-dir", default=".cache")
    ap.add_argument("--out", default="data/processed/flood_accessibility_5km.gpkg")
    ap.add_argument("--out-roads", default="data/processed/flooded_roads_RP50.gpkg")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    node_coords = np.load(cache / "node_coords.npy")
    edges = pd.read_parquet(cache / "edges.parquet")
    n_nodes = node_coords.shape[0]

    mid_x = (edges["ax"] + edges["bx"]) / 2.0
    mid_y = (edges["ay"] + edges["by"]) / 2.0
    depth = sample_flood_depth(args.flood_raster, mid_x.values, mid_y.values)
    edges["flood_depth_m"] = depth
    np.save(cache / "edge_flood_depth.npy", depth)

    flooded = np.isfinite(depth)
    print(f"{flooded.sum():,} of {len(edges):,} edges ({flooded.sum()/len(edges):.1%}) "
          f"intersect the RP50 flood extent; mean depth among them "
          f"{np.nanmean(depth):.1f}m")

    flooded_roads = edges[flooded].copy()
    flooded_roads["severity"] = flooded_roads["flood_depth_m"].apply(severity_label)
    flooded_roads_gdf = gpd.GeoDataFrame(
        flooded_roads[["source", "flood_depth_m", "severity"]].rename(columns={"flood_depth_m": "depth_m"}),
        geometry=[LineString([(r.ax, r.ay), (r.bx, r.by)]) for r in flooded_roads.itertuples()],
        crs="EPSG:32736",
    )
    Path(args.out_roads).parent.mkdir(parents=True, exist_ok=True)
    flooded_roads_gdf.to_file(args.out_roads, layer="flooded_roads", driver="GPKG")
    print(f"Severity breakdown:\n{flooded_roads_gdf['severity'].value_counts().to_string()}")

    facilities = gpd.read_file(args.facilities).to_crs("EPSG:32736")
    points = gpd.read_file(args.points).to_crs("EPSG:32736")
    fac_xy = np.column_stack([facilities.geometry.x, facilities.geometry.y])
    pts_xy = np.column_stack([points.geometry.x, points.geometry.y])

    graph_drive = sparse.load_npz(cache / "graph_drive.npz")
    fac_node, _, _ = snap_points_to_graph(fac_xy, node_coords, graph_drive)
    pts_node, pts_snap_dist, pts_snap_time = snap_points_to_graph(pts_xy, node_coords, graph_drive)

    facility_name_col = "NAMEOFFACI" if "NAMEOFFACI" in facilities.columns else facilities.columns[0]
    fac_names = facilities[facility_name_col].astype(str).values
    node_to_facrow = {int(n): i for i, n in enumerate(fac_node)}

    baseline = gpd.read_file(args.baseline, layer="baseline_accessibility")

    out = points[["geometry"]].copy()
    if "population" in points.columns:
        out["population"] = points["population"].values

    for mode in ["drive", "walk"]:
        minutes, keep = penalize_edges(edges, edges["flood_depth_m"].values, mode)
        n_removed = (~keep).sum()
        print(f"{mode}: {n_removed:,} edges impassable under the flood rule")

        graph_flood = build_penalized_graph(edges, minutes, keep, n_nodes)
        sparse.save_npz(cache / f"graph_{mode}_flood.npz", graph_flood)

        dist, nearest_source_node, predecessors = multi_source_travel_time(graph_flood, fac_node)
        point_time = dist[pts_node] + pts_snap_time
        point_nearest_node = nearest_source_node[pts_node]
        point_nearest_fac = [fac_names[node_to_facrow.get(int(n), -1)] if int(n) in node_to_facrow else None
                              for n in point_nearest_node]

        out[f"{mode}_base_min"] = baseline[f"{mode}_time_min"].values
        out[f"{mode}_base_class"] = baseline[f"{mode}_class"].values
        out[f"{mode}_base_fac"] = baseline[f"{mode}_nearest_facility"].values
        out[f"{mode}_flood_time_min"] = point_time
        out[f"{mode}_flood_class"] = [classify_time(t) for t in point_time]
        out[f"{mode}_flood_nearest_fac"] = point_nearest_fac

        base_rank = baseline[f"{mode}_class"].map({"well-served": 0, "underserved": 1, "critical": 2})
        flood_rank = out[f"{mode}_flood_class"].map({"well-served": 0, "underserved": 1, "critical": 2})
        out[f"{mode}_flood_delta"] = flood_rank - base_rank
        base_reachable = np.isfinite(baseline[f"{mode}_time_min"].values)
        flood_unreachable = ~np.isfinite(point_time)
        out[f"newly_cutoff_{mode}"] = base_reachable & flood_unreachable

        np.save(cache / f"flood_{mode}_pred.npy", predecessors)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_file(args.out, layer="flood_accessibility", driver="GPKG")

    pop_col = "population" if "population" in out.columns else None
    for mode in ["drive", "walk"]:
        n_cutoff = int(out[f"newly_cutoff_{mode}"].sum())
        print(f"{mode}: {n_cutoff:,} points newly cut off (of {len(out):,})")
        if pop_col:
            pop_cutoff = out.loc[out[f"newly_cutoff_{mode}"], pop_col].sum()
            print(f"  representing {pop_cutoff:,.0f} people")


if __name__ == "__main__":
    main()
