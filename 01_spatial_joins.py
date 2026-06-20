"""
01_spatial_joins.py
====================
Author: Emmanuel Oyekanlu — Principal Data Engineer

Demonstrates spatial joins in GeoPandas:
  - Left spatial join: assign county attributes to each farm parcel
  - Count fields per county (GROUP BY equivalent after join)
  - Polygon-to-polygon join (intersects predicate)
  - Handling many-to-many spatial relationships

Spatial join vs tabular join:
    A tabular join matches rows using a shared key column (e.g., parcel_id).
    A spatial join matches rows using geometric relationships:
      - 'within'     : geometry A is entirely inside geometry B
      - 'intersects' : any part of A touches or overlaps B
      - 'contains'   : geometry A entirely contains B
      - 'crosses'    : A crosses B (shares interior but not entirely inside)
      - 'touches'    : A and B share boundary but not interior

Use case context:
    USDA administrative boundaries (county, crop reporting district, HUC watershed)
    need to be joined to field-level data for:
      - Program eligibility determination (by county)
      - Reporting rollups to state/federal agencies
      - Tax assessment cross-referencing
      - Market analysis by geographic region

Run:
    python 01_spatial_joins.py
"""

import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PARCELS_PATH = os.path.join(DATA_DIR, "farm_parcels.geojson")
COUNTIES_PATH = os.path.join(DATA_DIR, "counties.geojson")


def load_data() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load farm parcels and county polygons into GeoDataFrames.

    GeoDataFrame is a pandas DataFrame with an additional 'geometry' column
    containing Shapely geometry objects. All standard pandas operations
    (filtering, groupby, merge, etc.) work on the non-geometry columns.

    Returns
    -------
    tuple[GeoDataFrame, GeoDataFrame]
        (parcels GeoDataFrame, counties GeoDataFrame)
    """
    parcels = gpd.read_file(PARCELS_PATH)
    counties = gpd.read_file(COUNTIES_PATH)

    print("=" * 65)
    print("DATA LOADED")
    print("=" * 65)
    print(f"\nFarm Parcels: {len(parcels)} features, CRS: {parcels.crs}")
    print(parcels[['parcel_id', 'crop_type', 'area_ha', 'county']].to_string())

    print(f"\nCounties: {len(counties)} features, CRS: {counties.crs}")
    print(counties[['county_name', 'fips', 'population']].to_string())
    print()
    return parcels, counties


def left_spatial_join_within(
    parcels: gpd.GeoDataFrame,
    counties: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Perform a left spatial join: for each parcel, find the county it lies within.

    gpd.sjoin() parameters:
      - left_df  : The 'primary' GeoDataFrame — one row per left feature in output
      - right_df : The 'reference' GeoDataFrame — attributes are joined from here
      - how      : 'left' keeps all left rows (unmatched right = NaN)
                   'inner' keeps only matched rows
                   'right' keeps all right rows
      - predicate: Spatial relationship test. 'within' means left geometry is
                   entirely inside right geometry.

    Performance note:
        gpd.sjoin() uses an R-tree spatial index (via PyGEOS/Shapely) for
        efficient lookup — O(n log n) rather than O(n²) brute-force.

    Parameters
    ----------
    parcels : GeoDataFrame
        Farm parcel polygons.
    counties : GeoDataFrame
        County polygons.

    Returns
    -------
    GeoDataFrame
        Parcels enriched with county attributes. '_left' and '_right' suffixes
        are added when both GeoDataFrames have columns of the same name.
    """
    print("=" * 65)
    print("LEFT SPATIAL JOIN: PARCELS WITHIN COUNTIES")
    print("=" * 65)

    # Ensure both GeoDataFrames use the same CRS before joining
    # sjoin will raise a CRSMismatch error if CRS differs
    if parcels.crs != counties.crs:
        counties = counties.to_crs(parcels.crs)
        print(f"  Reprojected counties to match parcels CRS: {parcels.crs}")

    # Perform the spatial join
    # 'within' predicate: parcel centroid/geometry must be inside county polygon
    # Use 'intersects' if parcels cross county boundaries
    joined = gpd.sjoin(
        parcels,
        counties[['county_name', 'fips', 'population', 'geometry']],
        how='left',
        predicate='intersects'  # Use intersects since polygons touch boundaries
    )

    # Drop the auxiliary right-index column added by sjoin
    joined = joined.drop(columns=['index_right'], errors='ignore')

    print(f"\nJoin result: {len(joined)} rows (left had {len(parcels)})")
    print(f"Columns added from counties: county_name, fips, population\n")

    # Show join results
    display_cols = ['parcel_id', 'crop_type', 'area_ha', 'county_name', 'fips']
    print(joined[display_cols].to_string())

    # Identify any parcels that didn't match a county (would have NaN county_name)
    unmatched = joined[joined['county_name'].isna()]
    if len(unmatched) > 0:
        print(f"\nWarning: {len(unmatched)} parcels did not match any county:")
        print(unmatched[['parcel_id']].to_string())
    else:
        print(f"\nAll {len(joined)} parcels matched a county successfully.")

    return joined


def count_fields_per_county(joined: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Compute aggregate statistics per county using a GROUP BY pattern.

    After a spatial join, the joined GeoDataFrame has county attributes
    on every parcel row. Standard pandas groupby() then computes county-level
    summaries — this is equivalent to SQL:
        SELECT county_name, COUNT(*), SUM(area_ha), AVG(yield_ton_ha)
        FROM parcels
        GROUP BY county_name

    Parameters
    ----------
    joined : GeoDataFrame
        Result of the spatial join with county_name column.

    Returns
    -------
    DataFrame
        County-level aggregate statistics.
    """
    print("\n" + "=" * 65)
    print("FIELDS PER COUNTY — GROUP BY SUMMARY")
    print("=" * 65)

    county_stats = joined.groupby('county_name').agg(
        field_count=('parcel_id', 'count'),
        total_area_ha=('area_ha', 'sum'),
        avg_area_ha=('area_ha', 'mean'),
        avg_yield_ton_ha=('yield_ton_ha', 'mean'),
        crop_types=('crop_type', lambda x: ', '.join(sorted(x.unique())))
    ).reset_index()

    county_stats = county_stats.sort_values('total_area_ha', ascending=False)

    print(f"\n{'County':<12} {'Fields':>7} {'Total Area':>12} "
          f"{'Avg Area':>10} {'Avg Yield':>10} {'Crops'}")
    print("-" * 80)
    for _, row in county_stats.iterrows():
        print(f"{row['county_name']:<12} {row['field_count']:>7} "
              f"{row['total_area_ha']:>10.1f}ha "
              f"{row['avg_area_ha']:>9.1f}ha "
              f"{row['avg_yield_ton_ha']:>9.2f}t/ha  "
              f"{row['crop_types']}")

    return county_stats


def crop_distribution_per_county(joined: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Build a cross-tabulation: how many fields of each crop type per county.

    Uses pandas pivot_table — a common pattern in agricultural reporting
    to show spatial distribution of crop types across administrative areas.
    """
    print("\n" + "=" * 65)
    print("CROP TYPE DISTRIBUTION PER COUNTY (cross-tabulation)")
    print("=" * 65)

    # Count fields per county × crop_type combination
    cross_tab = pd.crosstab(
        index=joined['county_name'],
        columns=joined['crop_type'],
        values=joined['area_ha'],
        aggfunc='sum',
        margins=True,
        margins_name='TOTAL'
    ).fillna(0)

    print(f"\nTotal area (ha) per county × crop type:\n")
    print(cross_tab.round(1).to_string())
    return cross_tab


def polygon_to_polygon_join(
    parcels: gpd.GeoDataFrame,
    counties: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Demonstrate polygon-to-polygon join using 'intersects' predicate.

    Unlike 'within' (which requires the parcel to be entirely inside the county),
    'intersects' returns a match whenever any part of the parcel touches the county.

    This produces a many-to-many result if polygons span multiple counties —
    each matching pair becomes a separate row.

    Use case: Find which counties overlap a given watershed polygon,
    or which fields are partially within a flood risk area.
    """
    print("\n" + "=" * 65)
    print("POLYGON-TO-POLYGON JOIN (intersects predicate)")
    print("=" * 65)

    # Create a flood risk zone by buffering county boundary (simulated)
    # In production: load actual flood zone polygons from FEMA/USGS
    flood_zone = counties[counties['county_name'] == 'Kings'].copy()
    flood_zone['zone_type'] = 'flood_risk'
    flood_zone['risk_level'] = 'moderate'

    # Find all parcels that intersect the Kings County flood zone
    at_risk = gpd.sjoin(
        parcels,
        flood_zone[['county_name', 'zone_type', 'risk_level', 'geometry']],
        how='inner',
        predicate='intersects'
    )

    print(f"\nParcels intersecting Kings County flood zone: {len(at_risk)}")
    if len(at_risk) > 0:
        print(at_risk[['parcel_id', 'crop_type', 'area_ha', 'county_name']].to_string())
        total_at_risk_area = at_risk['area_ha'].sum()
        print(f"\nTotal at-risk area: {total_at_risk_area:.1f} ha")

    return at_risk


def demonstrate_centroid_join(
    parcels: gpd.GeoDataFrame,
    counties: gpd.GeoDataFrame
) -> None:
    """
    Demonstrate centroid-based point-in-polygon join.

    Useful when parcel polygons span county boundaries and you want
    to assign each parcel to exactly one county (based on where most
    of it lies — approximated by centroid location).

    This avoids the many-to-many issue that arises with 'intersects'
    when polygons straddle boundaries.
    """
    print("\n" + "=" * 65)
    print("CENTROID-BASED POINT-IN-POLYGON JOIN")
    print("=" * 65)

    # Create a GeoDataFrame of parcel centroids
    centroids = parcels.copy()
    centroids['geometry'] = parcels.geometry.centroid
    centroids['original_geom_type'] = 'Polygon (centroid used for join)'

    # Point-in-polygon join
    centroid_join = gpd.sjoin(
        centroids,
        counties[['county_name', 'geometry']],
        how='left',
        predicate='within'
    )

    print(f"\nCentroid join result ({len(centroid_join)} rows):")
    print(centroid_join[['parcel_id', 'crop_type', 'county_name']].to_string())
    print(f"\nNote: Centroid join assigns each polygon to exactly one county,")
    print(f"      even if the polygon spans multiple county boundaries.")


def main():
    print("\n" + "=" * 65)
    print("SPATIAL JOINS DEMONSTRATION")
    print("Author: Emmanuel Oyekanlu — Principal Data Engineer")
    print("=" * 65 + "\n")

    # Load data
    parcels, counties = load_data()

    # Left spatial join: assign county to each parcel
    joined = left_spatial_join_within(parcels, counties)

    # Aggregate: fields and area per county
    county_stats = count_fields_per_county(joined)

    # Cross-tabulation: crop distribution per county
    cross_tab = crop_distribution_per_county(joined)

    # Polygon-to-polygon join
    at_risk = polygon_to_polygon_join(parcels, counties)

    # Centroid-based join
    demonstrate_centroid_join(parcels, counties)

    # Save join result
    output_path = os.path.join(OUTPUT_DIR, "parcels_with_county.geojson")
    save_cols = [c for c in joined.columns if c != 'geometry']
    joined_save = joined[save_cols + ['geometry']].copy()
    joined_save.to_file(output_path, driver='GeoJSON')
    print(f"\nSaved spatial join result: {output_path}")


if __name__ == "__main__":
    main()
