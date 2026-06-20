"""
05_dissolve_and_aggregate.py
=============================
Author: Emmanuel Oyekanlu — Principal Data Engineer

Demonstrates dissolve and spatial aggregation in GeoPandas:
  - Dissolve by crop_type: merge polygons sharing same crop into one geometry
  - Aggregate statistics: sum area, mean yield, count fields per crop
  - Multi-column dissolve (crop_type × county)
  - Plotting dissolved result with annotated labels

Why dissolve matters in agricultural data engineering:
    Field-level GeoDataFrames often need to be rolled up to crop-level or
    regional summaries. dissolve() is the spatial equivalent of SQL GROUP BY:
    it combines geometries (via union) while aggregating attribute values.

    Use cases:
      - Reporting total corn area per county to USDA NASS
      - Creating crop type maps for remote sensing validation
      - Simplifying data for web mapping (fewer polygons = faster rendering)
      - Pipeline aggregation: field → farm → region hierarchy

Run:
    python 05_dissolve_and_aggregate.py
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PARCELS_PATH = os.path.join(DATA_DIR, "farm_parcels.geojson")


def load_data() -> gpd.GeoDataFrame:
    """Load farm parcels and display initial statistics."""
    parcels = gpd.read_file(PARCELS_PATH)
    parcels_utm = parcels.to_crs('EPSG:32610')

    # Compute actual area from geometry (projected CRS)
    parcels_utm['geom_area_ha'] = parcels_utm.geometry.area / 10_000

    print("=" * 65)
    print("FARM PARCELS LOADED")
    print("=" * 65)
    print(f"\n{len(parcels_utm)} parcels, {parcels_utm['crop_type'].nunique()} crop types")
    print(f"Total area (attributes) : {parcels_utm['area_ha'].sum():.1f} ha")
    print(f"Total area (computed)   : {parcels_utm['geom_area_ha'].sum():.2f} ha\n")

    print("Individual parcels:")
    print(parcels_utm[['parcel_id', 'crop_type', 'area_ha', 'yield_ton_ha',
                         'owner', 'county']].to_string())

    return parcels_utm


def dissolve_by_crop_type(parcels: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Dissolve field polygons by crop_type, aggregating multiple statistics.

    dissolve() groups all features with the same value in the 'by' column,
    merges their geometries via shapely union, and applies aggfunc to all
    other numeric columns.

    aggfunc options (same as pandas groupby):
      'sum'   — total area per crop type
      'mean'  — average yield per crop type
      'count' — number of parcels per crop type
      'min'/'max' — range of values
      'first'/'last' — pick one value
      dict mapping {column: aggfunc} for per-column control

    Parameters
    ----------
    parcels : GeoDataFrame
        Farm parcel features in metric CRS.

    Returns
    -------
    GeoDataFrame
        Dissolved GeoDataFrame with one row per crop type.
    """
    print("\n" + "=" * 65)
    print("DISSOLVE BY CROP TYPE")
    print("=" * 65)

    print(f"\nInput : {len(parcels)} parcels")

    # Dissolve with per-column aggregation functions
    dissolved = parcels.dissolve(
        by='crop_type',
        aggfunc={
            'area_ha':      'sum',    # Total cultivated area per crop
            'geom_area_ha': 'sum',    # Total computed area
            'yield_ton_ha': 'mean',   # Average yield (simple mean)
            'parcel_id':    'count',  # Number of fields per crop type
            'owner':        lambda x: ', '.join(sorted(x.unique())),  # All owners
            'county':       lambda x: '/'.join(sorted(x.unique())),   # Counties
        }
    ).reset_index()

    # Rename for clarity
    dissolved = dissolved.rename(columns={
        'parcel_id': 'field_count',
        'area_ha': 'total_area_ha',
        'yield_ton_ha': 'avg_yield_ton_ha',
    })

    # Compute total production potential per crop type
    dissolved['total_production_t'] = (
        dissolved['total_area_ha'] * dissolved['avg_yield_ton_ha']
    ).round(1)

    # Compute actual dissolved geometry area
    dissolved['dissolved_area_ha'] = (dissolved.geometry.area / 10_000).round(2)

    print(f"Output: {len(dissolved)} dissolved crop type polygons\n")

    print(f"{'Crop Type':<12} {'Fields':>7} {'Total Ha':>10} "
          f"{'Avg Yld':>9} {'Production(t)':>14} {'Counties'}")
    print("-" * 75)
    for _, row in dissolved.sort_values('total_area_ha', ascending=False).iterrows():
        print(f"{row['crop_type']:<12} {row['field_count']:>7} "
              f"{row['total_area_ha']:>10.1f} "
              f"{row['avg_yield_ton_ha']:>8.2f}t "
              f"{row['total_production_t']:>13.0f}t "
              f"  {row['county']}")

    return dissolved


def dissolve_by_crop_and_county(parcels: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Multi-column dissolve: group by BOTH crop_type AND county.

    This creates a separate dissolved polygon for each unique
    (crop_type, county) combination — equivalent to a 2-level GROUP BY.

    Use case: USDA county-level crop acreage reporting requires totals
    broken down by both crop type and county.

    Parameters
    ----------
    parcels : GeoDataFrame
        Farm parcel features.

    Returns
    -------
    GeoDataFrame
        Dissolved by (crop_type, county) combinations.
    """
    print("\n" + "=" * 65)
    print("MULTI-COLUMN DISSOLVE: CROP TYPE × COUNTY")
    print("=" * 65)

    dissolved_2level = parcels.dissolve(
        by=['crop_type', 'county'],
        aggfunc={
            'area_ha':      'sum',
            'yield_ton_ha': 'mean',
            'parcel_id':    'count',
        }
    ).reset_index()

    dissolved_2level = dissolved_2level.rename(columns={
        'parcel_id': 'field_count',
        'area_ha': 'total_area_ha',
        'yield_ton_ha': 'avg_yield_ton_ha',
    })

    print(f"\n{len(dissolved_2level)} (crop × county) combinations:\n")
    print(f"{'Crop Type':<12} {'County':<10} {'Fields':>7} "
          f"{'Total Ha':>10} {'Avg Yield':>10}")
    print("-" * 55)
    for _, row in dissolved_2level.sort_values(
        ['crop_type', 'county']
    ).iterrows():
        print(f"{row['crop_type']:<12} {row['county']:<10} "
              f"{row['field_count']:>7} {row['total_area_ha']:>10.1f} "
              f"{row['avg_yield_ton_ha']:>9.2f}")

    return dissolved_2level


def compute_weighted_average_yield(parcels: gpd.GeoDataFrame) -> pd.Series:
    """
    Compute area-weighted average yield per crop type.

    Simple mean yield treats all fields equally regardless of size.
    Area-weighted mean better represents actual production:
        weighted_avg = Σ(area_i × yield_i) / Σ(area_i)

    This is analogous to weighted average in a portfolio — a large field
    has more influence on the regional yield average than a tiny field.

    Parameters
    ----------
    parcels : GeoDataFrame
        Farm parcel features with area_ha and yield_ton_ha.

    Returns
    -------
    Series
        Area-weighted average yield per crop type.
    """
    print("\n" + "=" * 65)
    print("AREA-WEIGHTED AVERAGE YIELD PER CROP TYPE")
    print("=" * 65)

    # Compute weighted yield = area × yield for each field
    parcels = parcels.copy()
    parcels['weighted_yield'] = parcels['area_ha'] * parcels['yield_ton_ha']

    # Group by crop: sum weighted_yield and total area, divide
    grouped = parcels.groupby('crop_type').agg(
        simple_mean_yield=('yield_ton_ha', 'mean'),
        total_area=('area_ha', 'sum'),
        total_weighted_yield=('weighted_yield', 'sum'),
    )
    grouped['weighted_avg_yield'] = (
        grouped['total_weighted_yield'] / grouped['total_area']
    ).round(2)

    print(f"\n{'Crop Type':<12} {'Simple Mean':>12} {'Wtd Avg':>10} {'Difference':>12}")
    print("-" * 50)
    for crop, row in grouped.iterrows():
        diff = row['weighted_avg_yield'] - row['simple_mean_yield']
        print(f"{crop:<12} {row['simple_mean_yield']:>11.2f}t "
              f"{row['weighted_avg_yield']:>9.2f}t "
              f"{diff:>+11.2f}t")

    print(f"\nNote: Positive difference = larger fields have higher yield")
    print(f"      Negative difference = smaller fields have higher yield")

    return grouped['weighted_avg_yield']


def visualize_dissolved(
    parcels: gpd.GeoDataFrame,
    dissolved: gpd.GeoDataFrame
) -> None:
    """
    Plot side-by-side: original parcels colored by crop type vs dissolved result.
    """
    # Assign consistent colors per crop type
    crop_types = sorted(dissolved['crop_type'].unique())
    color_map = {
        'almond':    '#F4B942',  # Golden
        'corn':      '#8BC34A',  # Light green
        'cotton':    '#FFFFFF',  # White (cotton!)
        'pistachio': '#558B2F',  # Dark green
        'tomato':    '#E53935',  # Red
        'wheat':     '#FFD54F',  # Amber
    }
    # Fallback for any unlisted crops
    default_colors = ['#9C27B0', '#00BCD4', '#FF9800', '#607D8B']
    for i, ct in enumerate(crop_types):
        if ct not in color_map:
            color_map[ct] = default_colors[i % len(default_colors)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, (gdf, title) in zip(axes, [
        (parcels.to_crs('EPSG:4326'), 'Original Parcels (12 features)'),
        (dissolved.to_crs('EPSG:4326'), f'Dissolved by Crop Type ({len(dissolved)} features)')
    ]):
        for _, row in gdf.iterrows():
            color = color_map.get(row['crop_type'], '#CCCCCC')
            gpd.GeoDataFrame([row], geometry='geometry', crs=gdf.crs).plot(
                ax=ax, color=color, alpha=0.75, edgecolor='white', linewidth=0.8
            )

        # Labels on dissolved plot
        if 'field_count' in gdf.columns:
            for _, row in gdf.iterrows():
                centroid = row.geometry.centroid
                label = f"{row['crop_type']}\n{row['total_area_ha']:.0f}ha"
                ax.annotate(label, xy=(centroid.x, centroid.y),
                            ha='center', va='center', fontsize=7.5,
                            fontweight='bold', color='black',
                            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6))

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

    # Build legend
    legend_patches = [
        mpatches.Patch(color=color_map.get(ct, '#CCC'), label=ct)
        for ct in crop_types
    ]
    axes[1].legend(handles=legend_patches, loc='lower right',
                   title='Crop Type', fontsize=9)

    fig.suptitle(
        'GeoPandas dissolve() — Field → Crop Type Aggregation\n'
        'Central Valley, CA | Author: Emmanuel Oyekanlu',
        fontsize=12, fontweight='bold'
    )

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "dissolve_result.png")
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved: {out_path}")


def main():
    print("\n" + "=" * 65)
    print("DISSOLVE AND AGGREGATE DEMONSTRATION")
    print("Author: Emmanuel Oyekanlu — Principal Data Engineer")
    print("=" * 65 + "\n")

    # Load data
    parcels = load_data()

    # Single-column dissolve by crop type
    dissolved = dissolve_by_crop_type(parcels)

    # Two-column dissolve by crop × county
    dissolved_2level = dissolve_by_crop_and_county(parcels)

    # Area-weighted yield analysis
    weighted_yields = compute_weighted_average_yield(parcels)

    # Visualize
    visualize_dissolved(parcels, dissolved)

    # Save outputs
    dissolved.to_crs('EPSG:4326').to_file(
        os.path.join(OUTPUT_DIR, "dissolved_by_crop.geojson"),
        driver='GeoJSON'
    )
    dissolved_2level.to_crs('EPSG:4326').to_file(
        os.path.join(OUTPUT_DIR, "dissolved_crop_county.geojson"),
        driver='GeoJSON'
    )

    print("\nOutputs saved:")
    print(f"  {os.path.join(OUTPUT_DIR, 'dissolved_by_crop.geojson')}")
    print(f"  {os.path.join(OUTPUT_DIR, 'dissolved_crop_county.geojson')}")


if __name__ == "__main__":
    main()
