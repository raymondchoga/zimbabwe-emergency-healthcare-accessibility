"""
05_build_routing_graph.py

Builds the routing graph used by every later step (baseline accessibility,
flood scenario, case studies) from the merged road network. See
utils_routing.build_graph_and_edges for the method.

Input:
  data/processed/roads_network.gpkg (or the roads_network_part*.gpkg split
  used in this repo purely to keep individual files under a friendly size for
  git — concatenate them first if you're working from the split files)

Output (written to --out-dir, default .cache/ -- gitignored; these are
large, fully reproducible from the inputs above, and not committed):
  graph_drive.npz        driving-speed graph (scipy.sparse, npz)
  graph_walk.npz         walking-speed graph (uniform 4.5 km/h)
  node_coords.npy        (n_nodes, 2) array of node coordinates
  edges.parquet          per-edge table (endpoints, length, source, speed-
                          derived minutes for both modes) used by
                          07_flood_scenario.py and 09_case_studies.py

Speed assumptions (Table 1 in the report):
  trunk / primary   80 km/h        secondary   60 km/h
  tertiary          40 km/h        unclassified 20 km/h
  WFP "Road"        50 km/h        WFP "Trail"  15 km/h
  walking           4.5 km/h, uniform across every class, both scenarios
"""
import argparse
import glob
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import sparse

from utils_routing import build_graph_and_edges

SNAP_TOLERANCE_M = 5.0

DRIVE_SPEED_KMH = {
    "osm_trunk": 80, "osm_primary": 80, "osm_secondary": 60,
    "osm_tertiary": 40, "osm_unclassified": 20,
    "zwe_roads_Road": 50, "zwe_roads_Trail": 15,
}
WALK_SPEED_KMH = 4.5  # uniform, every road/track class, both scenarios


def load_roads(path_or_glob):
    paths = sorted(glob.glob(path_or_glob)) if "*" in path_or_glob else [path_or_glob]
    parts = [gpd.read_file(p) for p in paths]
    return pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roads", default="data/processed/roads_network_part*.gpkg",
                     help="Path or glob to the merged road network (falls back to "
                          "data/processed/roads_network.gpkg if no parts found)")
    ap.add_argument("--out-dir", default=".cache")
    args = ap.parse_args()

    roads_path = args.roads
    if "*" in roads_path and not glob.glob(roads_path):
        roads_path = "data/processed/roads_network.gpkg"
    roads = load_roads(roads_path)
    print(f"Loaded {len(roads):,} road segments")

    # driving graph doubles as the source of node coordinates / edge geometry
    graph_drive, node_coords, edges = build_graph_and_edges(
        roads, DRIVE_SPEED_KMH, snap_tolerance_m=SNAP_TOLERANCE_M)
    print(f"Graph: {graph_drive.shape[0]:,} nodes, {graph_drive.nnz:,} directed edge entries "
          f"({len(edges):,} unique undirected edges before symmetrizing)")

    n_components, labels = sparse.csgraph.connected_components(graph_drive, directed=False)
    main_component_share = (np.bincount(labels).max() / graph_drive.shape[0])
    print(f"{n_components:,} connected components; "
          f"{main_component_share:.1%} of nodes in the largest one")

    # walking graph: identical topology, uniform speed
    walk_minutes = (edges["length_m"] / 1000.0) / WALK_SPEED_KMH * 60.0
    graph_walk = sparse.csr_matrix(
        (walk_minutes, (edges["a"], edges["b"])), shape=graph_drive.shape)
    graph_walk = graph_walk.maximum(graph_walk.T)

    edges["walk_minutes"] = walk_minutes.values

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(out_dir / "graph_drive.npz", graph_drive)
    sparse.save_npz(out_dir / "graph_walk.npz", graph_walk)
    np.save(out_dir / "node_coords.npy", node_coords)
    edges.to_parquet(out_dir / "edges.parquet")

    print(f"Wrote graph_drive.npz, graph_walk.npz, node_coords.npy, edges.parquet to {out_dir}/")


if __name__ == "__main__":
    main()
