"""
04_build_road_network.py

Merges the road-network source datasets into a single network with a
`source` field used downstream (05_build_routing_graph.py) to assign travel
speed by road class. Reprojects everything to EPSG:32736 (WGS 84 / UTM Zone
36S), the metres-based CRS used throughout this project.

Inputs (see README "Data Sources" for where to get each one; not included
in this repository — raw OSM/WFP extracts are large and OSM's ODbL license
requires attribution but permits redistribution of derived data, which is
what data/processed/roads_network_part*.gpkg already are):

  data_raw/roads/osm_major_roads.gpkg   3 layers: highway=trunk/primary/secondary
                                         (OSM, pulled via Overpass / QuickOSM)
  data_raw/roads/tertiary.gpkg          OSM highway=tertiary
  data_raw/roads/unclassified.gpkg      OSM highway=unclassified
  data_raw/roads/ZWE_roads.shp          WFP/humanitarian roads+trails dataset
                                         (F_CODE_DES = "Road" or "Trail")

Output:
  data/processed/roads_network.gpkg   layer default, columns: source, geometry

Why merge multiple sources at all: an OSM-only pull of trunk/primary/
secondary left most health facilities several kilometres from the nearest
mapped road (unusable for network analysis in rural areas). Adding OSM's
tertiary/unclassified classes and a WFP humanitarian roads+trails dataset
brought the median facility-to-road distance down to ~166m. See README for
the full before/after story.
"""
import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

TARGET_CRS = "EPSG:32736"

OSM_MAJOR_LAYERS = {
    "highway_trunk_Zimbabwe_8b7797": "osm_trunk",
    "highway_primary_Zimbabwe_e9267f": "osm_primary",
    "highway_secondary_Zimbabwe_820beb": "osm_secondary",
}


def load_osm_major(path):
    parts = []
    for layer, source_label in OSM_MAJOR_LAYERS.items():
        try:
            g = gpd.read_file(path, layer=layer)
        except Exception:
            continue
        parts.append(gpd.GeoDataFrame({"source": source_label, "geometry": g.geometry}, crs=g.crs))
    return pd.concat(parts, ignore_index=True)


def load_simple(path, source_label):
    g = gpd.read_file(path)
    return gpd.GeoDataFrame({"source": source_label, "geometry": g.geometry}, crs=g.crs)


def load_wfp(path):
    g = gpd.read_file(path)
    label = g["F_CODE_DES"].map({"Road": "zwe_roads_Road", "Trail": "zwe_roads_Trail"}).fillna("zwe_roads_Road")
    return gpd.GeoDataFrame({"source": label, "geometry": g.geometry}, crs=g.crs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--osm-major", default="data_raw/roads/osm_major_roads.gpkg")
    ap.add_argument("--osm-tertiary", default="data_raw/roads/tertiary.gpkg")
    ap.add_argument("--osm-unclassified", default="data_raw/roads/unclassified.gpkg")
    ap.add_argument("--wfp-roads", default="data_raw/roads/ZWE_roads.shp")
    ap.add_argument("--out", default="data/processed/roads_network.gpkg")
    args = ap.parse_args()

    parts = [
        load_osm_major(args.osm_major),
        load_simple(args.osm_tertiary, "osm_tertiary"),
        load_simple(args.osm_unclassified, "osm_unclassified"),
        load_wfp(args.wfp_roads),
    ]
    merged = pd.concat(parts, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, crs=parts[0].crs).to_crs(TARGET_CRS)

    # drop empty/invalid geometries
    merged = merged[merged.geometry.notna() & ~merged.geometry.is_empty]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged[["source", "geometry"]].to_file(args.out, driver="GPKG")

    print(f"Merged road network: {len(merged):,} segments")
    print(merged["source"].value_counts().to_string())
    total_km = merged.to_crs(TARGET_CRS).geometry.length.sum() / 1000
    print(f"Total length: {total_km:,.0f} km")


if __name__ == "__main__":
    main()
