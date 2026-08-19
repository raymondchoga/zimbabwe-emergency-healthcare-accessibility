"""
02_filter_health_facilities.py

Filters the Zimbabwe MoHCC master health facility list down to public,
operational, non-restricted-access facilities -- the set used as
destinations throughout this analysis.

This script exists mainly to document a data-quality lesson learned the
hard way during this project: the source CSV's OWNERSHIP field is an
undocumented numeric code (0-9), and it does NOT reliably distinguish
restricted-access facilities (prison/police clinics, serving inmates or
staff only, not the public) from ordinary public ones -- both commonly get
the same code. An initial ownership-code-based sweep missed 14 police
clinics that were only caught later, during case-study selection, when one
resolved as the "nearest facility" for a point named after a police
station. Before that correction, 164,084 people (1.1% of the national
population) had one of those 14 facilities as their computed nearest
facility. Name-pattern matching on top of the ownership code is what caught
them -- and per the project owner's explicit instruction, ANY facility
whose name suggests restricted (inmate/staff-only) access should be
excluded, not just ones literally named "ZRP" (Zimbabwe Republic Police).
If you re-run this against an updated facility list, re-check the
RESTRICTED_NAME_PATTERNS list below and treat a name match as a prompt to
verify, not an infallible rule -- and don't assume the ownership code alone
is sufficient, per the lesson above.

Input:
  health-facility-list-with-geo-codes.csv (Zimbabwe MoHCC master facility
  list; not redistributed in this repo -- see README "Data Sources")

Output:
  data/processed/health_facilities.gpkg           included facilities only
  data/processed/health_facilities_classified_ALL.csv   every row + reasoning
  data/processed/health_facilities_EXCLUDED.csv         excluded rows + reason
"""
import argparse
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd

# Reverse-engineered by cross-tabulating OWNERSHIP codes against TYPEOFACI text.
# Codes not in this map are treated as unknown/needs-review, not auto-excluded.
OWNERSHIP_CODE_MAP = {
    "0": "unspecified", "1": "unspecified",
    "2": "government", "3": "mission", "4": "mission",
    "5": "government/council", "6": "government/council",
    "7": "private", "8": "unspecified", "9": "unspecified",
    "govt": "government", "council": "government/council", "council ": "government/council",
    "mission": "mission", "private": "private", " mine": "private", "industrial": "private",
    "prison": "restricted", " zrp": "restricted", "zrp": "restricted", " zps": "restricted",
}

RESTRICTED_NAME_PATTERNS = [
    r"\bZRP\b",          # Zimbabwe Republic Police
    r"\bZPS\b",          # Zimbabwe Prisons (and Correctional) Service
    r"\bprison\b",
    r"\bcorrectional\b",
    r"\bpolice\b",
]

NON_OPERATIONAL_PATTERNS = [
    r"\bnot yet open\b", r"\bunder construction\b", r"\badmin(istration)? office\b",
    r"\bnon[- ]operational\b", r"\bclosed\b",
]


def classify_row(row):
    name = str(row.get("NAMEOFFACI", ""))
    ownership_raw = str(row.get("OWNERSHIP", "")).strip().lower()
    ownership_category = OWNERSHIP_CODE_MAP.get(ownership_raw, "unknown")

    for pat in RESTRICTED_NAME_PATTERNS:
        if re.search(pat, name, re.IGNORECASE):
            return False, "Excluded - restricted access (prison/police, not public)", ownership_category

    if ownership_category == "restricted":
        return False, "Excluded - restricted access (prison/police, not public)", ownership_category

    if ownership_category == "private":
        return False, "Excluded - private facility", ownership_category

    for pat in NON_OPERATIONAL_PATTERNS:
        if re.search(pat, name, re.IGNORECASE) or re.search(pat, str(row.get("COMMENTS", "")), re.IGNORECASE):
            return False, "Excluded - non-operational / administrative, not a service point", ownership_category

    lon = pd.to_numeric(row.get("LONGITUDE"), errors="coerce")
    lat = pd.to_numeric(row.get("LATITUDE"), errors="coerce")
    if pd.isna(lon) or pd.isna(lat):
        return False, "Excluded - missing or invalid coordinates", ownership_category

    return True, "Included - public facility", ownership_category


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--facilities-csv", default="health-facility-list-with-geo-codes.csv")
    ap.add_argument("--out-gpkg", default="data/processed/health_facilities.gpkg")
    ap.add_argument("--out-all-csv", default="data/processed/health_facilities_classified_ALL.csv")
    ap.add_argument("--out-excluded-csv", default="data/processed/health_facilities_EXCLUDED.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.facilities_csv)
    # first row of this HDX-style export is a HXL tag row (e.g. "#adm1+name"), not data
    df = df[~df["ID1"].isna() | ~df["Province"].astype(str).str.startswith("#")].copy()
    df = df[df["NAMEOFFACI"].notna()].copy()

    results = df.apply(classify_row, axis=1, result_type="expand")
    df["included"], df["exclusion_reason"], df["ownership_category"] = results[0], results[1], results[2]

    Path(args.out_all_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_all_csv, index=False)
    df[~df["included"]].to_csv(args.out_excluded_csv, index=False)

    included = df[df["included"]].copy()
    included["LONGITUDE"] = pd.to_numeric(included["LONGITUDE"], errors="coerce")
    included["LATITUDE"] = pd.to_numeric(included["LATITUDE"], errors="coerce")
    gdf = gpd.GeoDataFrame(
        included,
        geometry=gpd.points_from_xy(included["LONGITUDE"], included["LATITUDE"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:32736")
    gdf.to_file(args.out_gpkg, driver="GPKG")

    print(f"{len(df):,} total rows -> {len(included):,} included, {len(df) - len(included):,} excluded")
    print(df.loc[~df["included"], "exclusion_reason"].value_counts().to_string())
    restricted = df[df["exclusion_reason"] == "Excluded - restricted access (prison/police, not public)"]
    print(f"\n{len(restricted)} restricted-access facilities excluded by name/ownership pattern:")
    print(restricted["NAMEOFFACI"].to_string(index=False))


if __name__ == "__main__":
    main()
