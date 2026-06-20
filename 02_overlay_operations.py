"""
02_overlay_operations.py
=========================
Author: Emmanuel Oyekanlu — Principal Data Engineer

Demonstrates all four overlay operations in GeoPandas:
  - union        : combine both layers, split at overlaps
  - intersection : keep only areas in both layers
  - difference   : remove flood zone areas from farm parcels
  - symmetric_difference : areas in either layer but not both

Why overlays matter in agricultural data engineering:
    - Intersection: compute exact field area within a flood zone, EQIP zone,
      or conservation area for eligibility calculations
    - Difference: remove riparian buffer setback areas from plantable acreage
    - Union: combine field layers from two data providers for deduplication

Overlay vs spatial join:
    sjoin() adds attributes from one layer to another based on spatial relationship
    but does NOT split geometries. overlay() actually MODIFIES the geometries,
    splitting polygons where they overlap — producing new geometry features.

Run:
    python 02_overlay_operations.py
"""

import os
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for CI/server environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import Polygon, box

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PARCELS_PATH = os.path.join(DATA_DIR, "farm_parcels.geojson")


def create_flood_risk_zones() -> gpd.GeoDataFrame:
    """
    Create simulated flood risk zone polygons overlapping the farm parcels.

    In production: load from FEMA National Flood Hazard Layer (NFHL),
    NOAA storm surge inundation maps, or state floodplain GIS datasets.

    The zones partially overlap the farm parcel area to demonstrate
    all four overlay operations effectively.
    """
    # Two flood risk polygons at different risk levels
    # Positioned to overlap different parts of the parcel dataset
    zone_100yr = Polygon([
        (-120.0300, 36.7280), (-120.0100, 36.7280),
        (-120.0100, 36.7150), (-120.0300, 36.7150),
        (-120.0300, 36.7280)
    ])

    zone_500yr = Polygon([
        (-120.0450, 36.7400), (-120.0200, 36.7400),
        (-120.0200, 36.7270), (-120.0450, 36.7270),
        (-120.0450, 36.7400)
    ])

    flood_gdf = gpd.GeoDataFrame(
        {
            'zone_id': ['FZ-001', 'FZ-002'],
            'risk_level': ['100-year', '500-year'],
            'flood_depth_m': [2.4, 1.1],
            'last_updated': ['2024-01-15', '2024-01-15'],
        },
        geometry=[zone_100yr, zone_500yr],
        crs='EPSG:4326'
    )

    return flood_gdf


def perform_intersection(
    parcels: gpd.GeoDataFrame,
    flood_zones: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Intersection overlay: keep only the parts of parcels that fall within flood zones.

    This computes the exact geometry of overlap between farm fields and flood zones.
    Critical for:
      - Accurate crop loss assessment after a flood event
      - Insurance claim area calculation
      - Federal disaster program eligibility (USDA Emergency Loans)

    The result contains one row per parcel × flood zone overlap, with the geometry
    being the actual intersection polygon — NOT the original full parcel.

    Attributes from BOTH input GeoDataFrames are preserved in the result.
    """
    print("=" * 65)
    print("OVERLAY: INTERSECTION (parcels ∩ flood zones)")
    print("=" * 65)

    intersection = gpd.overlay(
        parcels,
        flood_zones,
        how='intersection',
        keep_geom_type=True  # Discard non-polygon results (points/lines from edge cases)
    )

    print(f"\nInput:  {len(parcels)} farm parcels")
    print(f"        {len(flood_zones)} flood zone polygons")
    print(f"Result: {len(intersection)} intersection polygons\n")

    if len(intersection) > 0:
        # Project to metric CRS to compute accurate areas
        intersection_proj = intersection.to_crs('EPSG:32610')
        intersection['intersection_area_ha'] = (
            intersection_proj.geometry.area / 10_000
        ).round(2)

        display_cols = ['parcel_id', 'crop_type', 'area_ha',
                        'zone_id', 'risk_level', 'intersection_area_ha']
        print(intersection[display_cols].to_string())

        # Summary: what fraction of each parcel is at risk?
        print("\n  At-risk area summary by crop type:")
        risk_by_crop = intersection.groupby('crop_type').agg(
            at_risk_ha=('intersection_area_ha', 'sum'),
            parcel_count=('parcel_id', 'nunique')
        ).sort_values('at_risk_ha', ascending=False)
        print(risk_by_crop.to_string())

    return intersection


def perform_difference(
    parcels: gpd.GeoDataFrame,
    flood_zones: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Difference overlay: subtract flood zone areas from parcels.

    Result: parcel geometries with flood-prone portions removed.
    This represents the PLANTABLE, non-flood-risk area of each field.

    Agricultural application:
      - Compute net plantable acres after removing flood setbacks
      - Identify safe zones for infrastructure placement
      - Remove wetland delineation areas from field boundaries

    Note: 'difference' is asymmetric — A.difference(B) ≠ B.difference(A)
    Here: parcels minus flood_zones → remaining safe parcel area
    """
    print("\n" + "=" * 65)
    print("OVERLAY: DIFFERENCE (parcels − flood zones)")
    print("=" * 65)

    difference = gpd.overlay(
        parcels,
        flood_zones,
        how='difference',
        keep_geom_type=True
    )

    print(f"\nResult: {len(difference)} polygon features")
    print("(Parcels with flood-risk portions removed)\n")

    if len(difference) > 0:
        # Compare original vs remaining area
        diff_proj = difference.to_crs('EPSG:32610')
        difference = difference.copy()
        difference['remaining_ha'] = (diff_proj.geometry.area / 10_000).round(2)

        # Merge with original areas to compute loss
        orig_areas = parcels.set_index('parcel_id')['area_ha']
        difference['original_ha'] = difference['parcel_id'].map(orig_areas)
        difference['area_lost_ha'] = (
            difference['original_ha'] - difference['remaining_ha']
        ).round(2)
        difference['pct_lost'] = (
            difference['area_lost_ha'] / difference['original_ha'] * 100
        ).round(1)

        display_cols = ['parcel_id', 'crop_type', 'original_ha',
                        'remaining_ha', 'area_lost_ha', 'pct_lost']
        print(difference[display_cols].to_string())

        total_original = difference['original_ha'].sum()
        total_remaining = difference['remaining_ha'].sum()
        print(f"\n  Total original area : {total_original:.1f} ha")
        print(f"  Total remaining     : {total_remaining:.1f} ha")
        print(f"  Total lost to flood : {total_original - total_remaining:.1f} ha")

    return difference


def perform_union(
    parcels: gpd.GeoDataFrame,
    flood_zones: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Union overlay: combine both layers, splitting geometries at boundaries.

    Result contains:
      - Parcel-only areas (outside flood zones): flood zone attributes = NaN
      - Flood-zone-only areas (outside parcels): parcel attributes = NaN
      - Overlap areas: attributes from both layers

    Use case: Create a comprehensive risk map that shows all parcels and
    flood zones in a single layer, with split geometries at boundaries.
    Useful for spatial data products delivered to downstream consumers
    who need a single unified layer.
    """
    print("\n" + "=" * 65)
    print("OVERLAY: UNION (parcels ∪ flood zones)")
    print("=" * 65)

    union = gpd.overlay(
        parcels[['parcel_id', 'crop_type', 'area_ha', 'geometry']],
        flood_zones[['zone_id', 'risk_level', 'geometry']],
        how='union',
        keep_geom_type=True
    )

    print(f"\nInput : {len(parcels)} parcels + {len(flood_zones)} flood zones")
    print(f"Result: {len(union)} union polygons")
    print(f"        (more than input because geometries are split at boundaries)\n")

    # Classify each resulting polygon by its origin
    union['origin'] = 'parcel_only'
    union.loc[union['parcel_id'].notna() & union['zone_id'].notna(), 'origin'] = 'overlap'
    union.loc[union['parcel_id'].isna() & union['zone_id'].notna(), 'origin'] = 'flood_zone_only'

    origin_counts = union['origin'].value_counts()
    print("  Union polygon classification:")
    for origin, count in origin_counts.items():
        print(f"    {origin:<25}: {count} polygons")

    return union


def perform_symmetric_difference(
    parcels: gpd.GeoDataFrame,
    flood_zones: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Symmetric difference: areas in either layer but NOT in both.

    Equivalent to (A ∪ B) − (A ∩ B) — removes the overlap areas from the union.

    Use case: Identify field areas that are exclusively parcel (not flood-affected)
    OR exclusively flood zone (not agricultural) — useful for mapping
    non-agricultural flood-affected areas for emergency response.
    """
    print("\n" + "=" * 65)
    print("OVERLAY: SYMMETRIC DIFFERENCE (parcels △ flood zones)")
    print("=" * 65)

    sym_diff = gpd.overlay(
        parcels[['parcel_id', 'crop_type', 'geometry']],
        flood_zones[['zone_id', 'risk_level', 'geometry']],
        how='symmetric_difference',
        keep_geom_type=True
    )

    print(f"\nResult: {len(sym_diff)} symmetric difference polygons")
    print("(Areas in parcels OR flood zones, but NOT both)\n")

    # Classify origin
    sym_diff['origin'] = sym_diff.apply(
        lambda r: 'parcel_only' if pd.notna(r.get('parcel_id')) else 'flood_zone_only',
        axis=1
    )

    origin_counts = sym_diff['origin'].value_counts()
    for origin, count in origin_counts.items():
        print(f"  {origin}: {count} polygons")

    return sym_diff


def visualize_overlays(
    parcels: gpd.GeoDataFrame,
    flood_zones: gpd.GeoDataFrame,
    intersection: gpd.GeoDataFrame,
    difference: gpd.GeoDataFrame
) -> None:
    """
    Create a 2×2 visualization of the overlay results.

    Saves to output/overlay_visualization.png
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        'Geospatial Overlay Operations — Farm Parcels vs Flood Risk Zones\n'
        'Central Valley, CA | Author: Emmanuel Oyekanlu',
        fontsize=13, fontweight='bold', y=0.98
    )

    # Color palette
    parcel_color = '#4CAF50'     # Green for farm parcels
    flood_color = '#2196F3'      # Blue for flood zones
    intersect_color = '#FF5722'  # Orange-red for intersection
    diff_color = '#8BC34A'       # Light green for safe remaining area

    # --- Plot 1: Input layers ---
    ax = axes[0, 0]
    parcels.plot(ax=ax, color=parcel_color, alpha=0.6, edgecolor='darkgreen', linewidth=0.5)
    flood_zones.plot(ax=ax, color=flood_color, alpha=0.4, edgecolor='navy', linewidth=1.5)
    ax.set_title('Input Layers\n(Green=Parcels, Blue=Flood Zones)', fontsize=10)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    p_patch = mpatches.Patch(color=parcel_color, alpha=0.6, label='Farm Parcels')
    f_patch = mpatches.Patch(color=flood_color, alpha=0.4, label='Flood Zones')
    ax.legend(handles=[p_patch, f_patch], loc='upper right', fontsize=8)

    # --- Plot 2: Intersection ---
    ax = axes[0, 1]
    parcels.plot(ax=ax, color=parcel_color, alpha=0.2, edgecolor='darkgreen', linewidth=0.5)
    flood_zones.plot(ax=ax, color=flood_color, alpha=0.2, edgecolor='navy', linewidth=1.5)
    if len(intersection) > 0:
        intersection.plot(ax=ax, color=intersect_color, alpha=0.8,
                          edgecolor='darkred', linewidth=1)
    ax.set_title('Intersection (A ∩ B)\nArea in BOTH layers', fontsize=10)
    ax.set_xlabel('Longitude')

    i_patch = mpatches.Patch(color=intersect_color, alpha=0.8, label='Intersection')
    ax.legend(handles=[i_patch], loc='upper right', fontsize=8)

    # --- Plot 3: Difference ---
    ax = axes[1, 0]
    flood_zones.plot(ax=ax, color=flood_color, alpha=0.3, edgecolor='navy', linewidth=1.5)
    if len(difference) > 0:
        difference.plot(ax=ax, color=diff_color, alpha=0.7,
                        edgecolor='darkgreen', linewidth=0.5)
    ax.set_title('Difference (A − B)\nParcels minus flood zones', fontsize=10)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    d_patch = mpatches.Patch(color=diff_color, alpha=0.7, label='Safe Parcel Area')
    ax.legend(handles=[d_patch], loc='upper right', fontsize=8)

    # --- Plot 4: All operations summary ---
    ax = axes[1, 1]
    parcels.plot(ax=ax, color=parcel_color, alpha=0.3, edgecolor='darkgreen', linewidth=0.5)
    flood_zones.plot(ax=ax, color=flood_color, alpha=0.3, edgecolor='navy', linewidth=1.5)
    if len(intersection) > 0:
        intersection.plot(ax=ax, color=intersect_color, alpha=0.7,
                          edgecolor='darkred', linewidth=0.8)
    ax.set_title('All Operations Summary\nOverlap areas highlighted', fontsize=10)
    ax.set_xlabel('Longitude')

    for ax in axes.flat:
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "overlay_visualization.png")
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved: {out_path}")


# Need pandas for notna
import pandas as pd


def main():
    print("\n" + "=" * 65)
    print("OVERLAY OPERATIONS DEMONSTRATION")
    print("Author: Emmanuel Oyekanlu — Principal Data Engineer")
    print("=" * 65 + "\n")

    # Load data
    parcels = gpd.read_file(PARCELS_PATH)
    print(f"Loaded {len(parcels)} farm parcels")

    # Create simulated flood zones
    flood_zones = create_flood_risk_zones()
    print(f"Created {len(flood_zones)} flood risk zones\n")

    # Perform all four overlay operations
    intersection = perform_intersection(parcels, flood_zones)
    difference = perform_difference(parcels, flood_zones)
    union = perform_union(parcels, flood_zones)
    sym_diff = perform_symmetric_difference(parcels, flood_zones)

    # Save outputs
    if len(intersection) > 0:
        intersection.to_file(
            os.path.join(OUTPUT_DIR, "flood_risk_intersection.geojson"),
            driver='GeoJSON'
        )

    # Visualize
    visualize_overlays(parcels, flood_zones, intersection, difference)

    print("\n" + "=" * 65)
    print("OVERLAY SUMMARY")
    print("=" * 65)
    print(f"  intersection : {len(intersection)} polygons (parcel ∩ flood)")
    print(f"  difference   : {len(difference)} polygons (parcel − flood)")
    print(f"  union        : {len(union)} polygons (parcel ∪ flood)")
    print(f"  sym_diff     : {len(sym_diff)} polygons (parcel △ flood)")


if __name__ == "__main__":
    main()
