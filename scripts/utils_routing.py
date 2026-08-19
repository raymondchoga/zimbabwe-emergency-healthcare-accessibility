"""
utils_routing.py

Shared helpers for building a routable graph from a road network and running
multi-source shortest-path accessibility analysis on it.

Method summary
--------------
Every vertex of every road LineString becomes a node in a scipy.sparse graph.
Coordinates are snapped to a tolerance (default 5m, in the CRS's own units,
so use a projected metres-based CRS) to merge near-duplicate nodes that
appear when combining road segments from different source datasets. Edge
weights are travel time in minutes, from each edge's length and an assumed
speed (km/h) looked up by the edge's `source` class.

For a graph of this size (Zimbabwe's merged road network produces roughly
1.7 million nodes and 1.8 million edges), scipy.sparse.csgraph.dijkstra with
min_only=True and a list of source node indices computes, in one call and
about a second of compute, the travel time AND identity of the nearest
source for every node in the network. This is dramatically faster than
either QGIS's native Processing network-analysis tools (which are
one-origin-at-a-time) or a pure-Python/networkx graph (which struggles to
even build a graph this size quickly).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
from scipy.spatial import cKDTree


def build_graph_and_edges(roads_gdf, speed_kmh_by_source, snap_tolerance_m=5.0,
                           default_speed_kmh=20.0):
    """
    Build an undirected routing graph from a road network GeoDataFrame.

    Parameters
    ----------
    roads_gdf : GeoDataFrame
        Road segments (LineString / MultiLineString), in a projected,
        metres-based CRS, with a `source` column used to look up speed.
    speed_kmh_by_source : dict[str, float]
        Maps a `source` value (road class) to an assumed travel speed, km/h.
    snap_tolerance_m : float
        Node-merging tolerance. Coordinates are rounded to this grid before
        being treated as the same graph node.
    default_speed_kmh : float
        Speed used for any `source` value not present in the lookup dict.

    Returns
    -------
    graph : scipy.sparse.csr_matrix, shape (n_nodes, n_nodes)
        Symmetric (undirected) node-by-node travel-time graph, in minutes.
    node_coords : np.ndarray, shape (n_nodes, 2)
        (x, y) coordinate of every graph node, same CRS as roads_gdf.
    edges : pandas.DataFrame
        One row per directed edge with columns: a (node idx), b (node idx),
        ax, ay, bx, by (endpoint coords), length_m, minutes, source. Kept
        separately from the sparse graph because later steps (flood-depth
        sampling, route-cut detection) need per-edge geometry and identity,
        which a sparse matrix alone doesn't preserve conveniently.
    """
    node_idx: dict[tuple[float, float], int] = {}
    rows, cols, weights = [], [], []
    edge_a, edge_b, edge_ax, edge_ay, edge_bx, edge_by = [], [], [], [], [], []
    edge_len, edge_min, edge_src = [], [], []

    def get_node(xy):
        key = (round(xy[0] / snap_tolerance_m) * snap_tolerance_m,
               round(xy[1] / snap_tolerance_m) * snap_tolerance_m)
        idx = node_idx.get(key)
        if idx is None:
            idx = len(node_idx)
            node_idx[key] = idx
        return idx

    for geom, source in zip(roads_gdf.geometry, roads_gdf["source"]):
        if geom is None or geom.is_empty:
            continue
        speed = speed_kmh_by_source.get(source, default_speed_kmh)
        lines = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
        for line in lines:
            coords = list(line.coords)
            for a, b in zip(coords[:-1], coords[1:]):
                ia, ib = get_node(a), get_node(b)
                if ia == ib:
                    continue
                length_m = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
                if length_m == 0:
                    continue
                minutes = (length_m / 1000.0) / speed * 60.0
                rows.append(ia); cols.append(ib); weights.append(minutes)
                edge_a.append(ia); edge_b.append(ib)
                edge_ax.append(a[0]); edge_ay.append(a[1])
                edge_bx.append(b[0]); edge_by.append(b[1])
                edge_len.append(length_m); edge_min.append(minutes); edge_src.append(source)

    n = len(node_idx)
    graph = csr_matrix((weights, (rows, cols)), shape=(n, n))
    graph = graph.maximum(graph.T)  # symmetrize -> undirected

    node_coords = np.zeros((n, 2))
    for xy, idx in node_idx.items():
        node_coords[idx] = xy

    edges = pd.DataFrame({
        "a": edge_a, "b": edge_b,
        "ax": edge_ax, "ay": edge_ay, "bx": edge_bx, "by": edge_by,
        "length_m": edge_len, "minutes": edge_min, "source": edge_src,
    })
    return graph, node_coords, edges


def snap_points_to_graph(points_xy, node_coords, graph, walking_kmh=4.5):
    """
    Snap each (x, y) point to its nearest graph node, preferring nodes in the
    graph's largest connected component so a point never lands on a small,
    disconnected island of road data and gets wrongly marked unreachable.

    Returns
    -------
    node_idx : np.ndarray[int]      nearest graph node index, per point
    snap_dist_m : np.ndarray[float] straight-line distance to that node
    snap_time_min : np.ndarray[float] walking time to cover snap_dist_m
    """
    n_components, labels = connected_components(graph, directed=False)
    main_label = np.bincount(labels).argmax()
    main_mask = labels == main_label

    tree_all = cKDTree(node_coords)
    dist_all, idx_all = tree_all.query(points_xy)

    off_main = ~main_mask[idx_all]
    if off_main.any():
        tree_main = cKDTree(node_coords[main_mask])
        main_indices = np.where(main_mask)[0]
        dist_main, idx_main = tree_main.query(points_xy[off_main])
        idx_all[off_main] = main_indices[idx_main]
        dist_all[off_main] = dist_main

    snap_time_min = (dist_all / 1000.0) / walking_kmh * 60.0
    return idx_all, dist_all, snap_time_min


def multi_source_travel_time(graph, source_node_idx):
    """
    One multi-source Dijkstra search, seeded simultaneously from every node
    in source_node_idx. Returns, for every node in the graph, the travel
    time to and the identity of its nearest source, in a single pass.

    Returns
    -------
    dist : np.ndarray            minutes to the nearest source (inf if none)
    nearest_source_node : np.ndarray  graph node index of the nearest source
    predecessors : np.ndarray    predecessor array; see reconstruct_route()
    """
    dist, predecessors, nearest_source_node = dijkstra(
        graph, directed=False, indices=source_node_idx,
        min_only=True, return_predecessors=True,
    )
    return dist, nearest_source_node, predecessors


def reconstruct_route(predecessors, start_node):
    """
    Walk a predecessor array (from multi_source_travel_time) back from
    start_node to the source it was assigned to. Returns the list of graph
    node indices making up the route, in source -> start_node order.
    scipy's sentinel for "no predecessor" (source reached) is -9999.
    """
    path = [start_node]
    cur = start_node
    while predecessors[cur] != -9999:
        cur = predecessors[cur]
        path.append(cur)
    return path[::-1]


def classify_time(minutes):
    """Standard three-band accessibility classification used throughout this
    project: well-served (<30 min), underserved (30-60 min), critical
    (>60 min or unreachable). See README.md / docs for the rationale and the
    international benchmark (Lancet Commission on Global Surgery, 2015) this
    is a finer-grained, sub-national adaptation of."""
    if not np.isfinite(minutes):
        return "critical"
    if minutes < 30:
        return "well-served"
    if minutes < 60:
        return "underserved"
    return "critical"
