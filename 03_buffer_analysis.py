"""
03_buffer_analysis.py
======================
Author: Emmanuel Oyekanlu — Principal Data Engineer

Demonstrates buffer analysis in GeoPandas:
  - Creating buffers around Points (wells) at multiple distances
  - Creating buffers around LineStrings (irrigation canals)
  - Using dissolve() to merge overlapping buffers into unified zones
  - Finding field parcels within buffer zones (within-buffer query)
  - Multi-distance buffer rings (annuli) for distance zone mapping

Use case context:
    Buffer analysis is foundational to agricultural compliance and planning:
      - Pesticide application setbacks: 50m from water bodies, 100m from schools
      - Irrigation planning: which fields are within 300m of a water source?
      - Well protection zones: classify land use within 150m, 300m, 600m of well
      - AGV exclusion zones: buffer around obstacles for safe navigation corridors

Critical note on CRS for buffering:
    ALWAYS project to a metric CRS before buffering.
    buffer(300) on WGS84 coordinates creates a buffer of 300 DEGREES,
    which is enormous and meaningless. Project to UTM or State Plane first,
    apply metric buffer, then optionally reproject back to WGS84.

    Central Valley, CA → UTM Zone 10N (EPSG:32610) or
                         CA State Plane Zone III (EPSG:26943)

Run:
    python 03_buffer_analysis.py
"""

import os
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import LineString, Point

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PARCELS_PATH = os.path.join(DATA_DIR, "farm_parcels.geojson")
WATER_PATH = os.path.join(DATA_DIR, "water_sources.geojson")

# Buffer distances in meters
WELL_BUFFER_DISTANCES_M = [150, 300, 600]    # Wellhead protection zones
CANAL_BUFFER_DISTANCES_M = [50, 100, 200]    # Pesticide setback zones


def load_data() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load parcels and water sources, project to metric CRS for buffering."""
    parcels = gpd.read_file(PARCELS_PATH)
    wells = gpd.read_file(WATER_PATH)

    print("=" * 65)
    print("DATA LOADED")
    print("=" * 65)
    print(f"  Farm parcels  : {len(parcels)} features, CRS: {parcels.crs}")
    print(f"  Water sources : {len(wells)} features, CRS: {wells.crs}")

    # Project to UTM Zone 10N for accurate metric buffers
    parcels_utm = parcels.to_crs('EPSG:32610')
    wells_utm = wells.to_crs('EPSG:32610')

    print(f"\n  Projected to UTM Zone 10N (EPSG:32610) for metric buffering")
    return parcels_utm, wells_utm


def create_well_buffers(
    wells: gpd.GeoDataFrame,
    distances_m: list[int]
) -> dict[int, gpd.GeoDataFrame]:
    """
    Create buffer zones around wells at multiple protection radii.

    Wellhead Protection Areas (WHPAs) per EPA guidelines:
      - Zone 1 (150m): Immediate protection zone — no septic, no chemical storage
      - Zone 2 (300m): Inner protection — no injection wells, restricted pesticides
      - Zone 3 (600m): Outer protection — managed agricultural use

    The buffer() method returns a GeoSeries of buffer polygons.
    We build a new GeoDataFrame with the buffered geometries.

    Parameters
    ----------
    wells : GeoDataFrame
        Well point features in a metric CRS.
    distances_m : list[int]
        Buffer radii in meters.

    Returns
    -------
    dict[int, GeoDataFrame]
        Maps buffer distance → GeoDataFrame of buffer polygons.
    """
    print("\n" + "=" * 65)
    print("WELL BUFFER ZONES (Wellhead Protection Areas)")
    print("=" * 65)

    buffer_gdfs = {}

    for dist in distances_m:
        # Create a copy of the wells GeoDataFrame with buffered geometries
        buffered = wells.copy()
        buffered['geometry'] = wells.geometry.buffer(dist)
        buffered['buffer_dist_m'] = dist
        buffered['zone_name'] = f'Zone_{dist}m'

        # Compute buffer area
        buffered['buffer_area_ha'] = buffered.geometry.area / 10_000

        buffer_gdfs[dist] = buffered

        print(f"\n  Buffer at {dist}m:")
        for _, row in buffered[buffered['type'] == 'well'].iterrows():
            print(f"    {row['source_id']}: buffer area = {row['buffer_area_ha']:.2f} ha")

    return buffer_gdfs


def create_canal_buffers(distances_m: list[int]) -> dict[int, gpd.GeoDataFrame]:
    """
    Create buffer zones around irrigation canal LineStrings.

    In this demo, we simulate an irrigation canal as a LineString.
    In production: load from USDA FSA, state water board GIS, or OpenStreetMap.

    Canal buffers represent pesticide application setback zones:
      - 50m: Restricted use pesticides require permit
      - 100m: Standard pesticide setback per county ordinance
      - 200m: Expanded setback for aerial application

    Parameters
    ----------
    distances_m : list[int]
        Buffer distances in meters.

    Returns
    -------
    dict[int, GeoDataFrame]
        Buffer GeoDataFrame per distance.
    """
    print("\n" + "=" * 65)
    print("IRRIGATION CANAL BUFFER ZONES (Pesticide Setbacks)")
    print("=" * 65)

    # Simulated irrigation canal running north-south through the parcel area
    # In UTM Zone 10N coordinates (meters) — approximate for Central Valley
    canal_utm_coords = [
        (728500, 4065800),  # North end
        (728400, 4065500),
        (728300, 4065200),
        (728200, 4064900),
        (728100, 4064600),  # South end
    ]

    canal_line = LineString(canal_utm_coords)
    canal_gdf = gpd.GeoDataFrame(
        [{'canal_id': 'CANAL-001', 'name': 'West Lateral Canal',
          'flow_rate_cfs': 45.0, 'operator': 'Kings River Conservation District'}],
        geometry=[canal_line],
        crs='EPSG:32610'
    )

    buffer_gdfs = {}

    for dist in distances_m:
        buffered = canal_gdf.copy()
        buffered['geometry'] = canal_gdf.geometry.buffer(dist)
        buffered['buffer_dist_m'] = dist
        buffered['setback_type'] = f'pesticide_setback_{dist}m'
        buffered['buffer_area_ha'] = buffered.geometry.area / 10_000

        buffer_gdfs[dist] = buffered
        print(f"  Canal buffer at {dist}m: {buffered['buffer_area_ha'].sum():.2f} ha")

    return buffer_gdfs


def dissolve_overlapping_buffers(
    buffer_dict: dict[int, gpd.GeoDataFrame],
    label: str
) -> gpd.GeoDataFrame:
    """
    Merge all buffer polygons of the same distance into single unified polygons.

    When multiple wells or canal segments have overlapping buffers,
    dissolve() merges them into a single polygon per buffer distance.
    This creates the 'protection zone' as a single unified area
    rather than overlapping circles.

    dissolve() parameters:
      - by: Column to group by (buffer_dist_m)
      - aggfunc: How to aggregate non-geometry columns

    Parameters
    ----------
    buffer_dict : dict
        Distance → GeoDataFrame of individual buffers.
    label : str
        Label for print output.

    Returns
    -------
    GeoDataFrame
        Dissolved buffer zones (one polygon per distance).
    """
    print(f"\n  Dissolving overlapping {label} buffers...")

    # Concatenate all distance levels into one GeoDataFrame
    all_buffers = gpd.GeoDataFrame(
        pd.concat(buffer_dict.values(), ignore_index=True),
        crs=list(buffer_dict.values())[0].crs
    )

    # Dissolve by buffer distance: merges overlapping buffer circles into one polygon
    dissolved = all_buffers.dissolve(
        by='buffer_dist_m',
        aggfunc={
            'buffer_area_ha': 'sum',  # Note: this will overcount for overlapping zones
            'zone_name': 'first'      # Keep first zone name (or define a merged name)
        }
    ).reset_index()

    # Recompute actual dissolved area
    dissolved['dissolved_area_ha'] = dissolved.geometry.area / 10_000

    print(f"  Individual buffers: {len(all_buffers)} → Dissolved: {len(dissolved)}")
    for _, row in dissolved.iterrows():
        dist = row['buffer_dist_m']
        area = row['dissolved_area_ha']
        print(f"    {dist}m zone: {area:.2f} ha (dissolved)")

    return dissolved


def find_parcels_in_buffer_zone(
    parcels: gpd.GeoDataFrame,
    buffer_zone: gpd.GeoDataFrame,
    buffer_dist_m: int,
    source_label: str
) -> gpd.GeoDataFrame:
    """
    Find farm parcels that intersect a given buffer zone.

    This is a within-buffer query: which fields are potentially affected
    by the protection/setback zone?

    Uses sjoin with 'intersects' predicate — any parcel touching the buffer
    polygon (even partially) is flagged as affected.

    Parameters
    ----------
    parcels : GeoDataFrame
        Farm parcel polygons in metric CRS.
    buffer_zone : GeoDataFrame
        Buffer polygon(s) to test against.
    buffer_dist_m : int
        Buffer distance in meters (for labeling).
    source_label : str
        Description of the source (e.g., 'well', 'canal').

    Returns
    -------
    GeoDataFrame
        Subset of parcels that intersect the buffer zone.
    """
    # Filter to the specific distance level
    zone = buffer_zone[buffer_zone['buffer_dist_m'] == buffer_dist_m].copy()

    if len(zone) == 0:
        print(f"  No buffer zone found at {buffer_dist_m}m")
        return gpd.GeoDataFrame()

    # Spatial join: find all parcels intersecting the buffer
    in_buffer = gpd.sjoin(
        parcels,
        zone[['geometry']],
        how='inner',
        predicate='intersects'
    ).drop(columns=['index_right'], errors='ignore')

    return in_buffer


def create_multi_distance_rings(
    wells: gpd.GeoDataFrame,
    distances_m: list[int]
) -> gpd.GeoDataFrame:
    """
    Create annular rings (donuts) between buffer distances.

    A ring at 150-300m shows the area between the inner and outer buffer —
    useful for multi-zone land use regulation that treats each zone differently.

    Uses Shapely's difference operation on consecutive buffers:
        Ring(150-300m) = Buffer(300m).difference(Buffer(150m))

    Parameters
    ----------
    wells : GeoDataFrame
        Well points in metric CRS.
    distances_m : list[int]
        Sorted buffer distances defining ring boundaries.

    Returns
    -------
    GeoDataFrame
        Annular ring polygons with inner/outer distance attributes.
    """
    rings = []
    distances_m = sorted(distances_m)

    for i, well_row in wells.iterrows():
        if well_row['type'] != 'well':
            continue

        well_geom = well_row['geometry']
        outer_buffers = {d: well_geom.buffer(d) for d in distances_m}

        # Create rings: each ring = outer_buffer - inner_buffer
        for j, dist in enumerate(distances_m):
            inner_dist = distances_m[j - 1] if j > 0 else 0
            outer_buf = outer_buffers[dist]
            inner_buf = well_geom.buffer(inner_dist) if inner_dist > 0 else well_geom

            ring_geom = outer_buf.difference(inner_buf)

            rings.append({
                'source_id': well_row['source_id'],
                'inner_m': inner_dist,
                'outer_m': dist,
                'ring_label': f'{inner_dist}-{dist}m',
                'ring_area_ha': ring_geom.area / 10_000,
                'geometry': ring_geom
            })

    rings_gdf = gpd.GeoDataFrame(rings, crs=wells.crs)
    return rings_gdf


def visualize_buffers(
    parcels: gpd.GeoDataFrame,
    well_buffers_dissolved: gpd.GeoDataFrame,
    wells: gpd.GeoDataFrame
) -> None:
    """Visualize parcels with well buffer zones overlaid."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Color ramp for buffer zones (large to small so inner is visible)
    buffer_colors = {600: '#BBDEFB', 300: '#64B5F6', 150: '#1565C0'}
    buffer_alphas = {600: 0.25, 300: 0.35, 150: 0.5}

    # Draw buffer zones (largest first so smaller ones are on top)
    for dist in sorted(buffer_colors.keys(), reverse=True):
        zone = well_buffers_dissolved[
            well_buffers_dissolved['buffer_dist_m'] == dist
        ]
        if len(zone) > 0:
            zone.plot(ax=ax, color=buffer_colors[dist],
                      alpha=buffer_alphas[dist], edgecolor='navy',
                      linewidth=1.5, label=f'Well buffer {dist}m')

    # Draw parcels
    parcels.plot(ax=ax, color='#4CAF50', alpha=0.5,
                 edgecolor='darkgreen', linewidth=0.8)

    # Draw well points
    wells[wells['type'] == 'well'].plot(
        ax=ax, color='blue', markersize=60, zorder=5,
        marker='*', label='Wells'
    )

    # Annotate well IDs
    for _, row in wells[wells['type'] == 'well'].iterrows():
        ax.annotate(
            row['source_id'],
            xy=(row.geometry.x, row.geometry.y),
            xytext=(5, 5), textcoords='offset points',
            fontsize=8, color='navy', fontweight='bold'
        )

    ax.set_title(
        'Well Buffer Zones (Wellhead Protection Areas)\n'
        'Central Valley, CA — UTM Zone 10N\n'
        'Author: Emmanuel Oyekanlu',
        fontsize=12, fontweight='bold'
    )
    ax.set_xlabel('Easting (m, UTM Zone 10N)')
    ax.set_ylabel('Northing (m, UTM Zone 10N)')

    # Build legend
    handles = [
        mpatches.Patch(color=buffer_colors[d], alpha=buffer_alphas[d],
                       label=f'{d}m buffer zone')
        for d in sorted(buffer_colors.keys())
    ]
    handles.append(mpatches.Patch(color='#4CAF50', alpha=0.5, label='Farm Parcels'))
    ax.legend(handles=handles, loc='upper right', fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "buffer_analysis.png")
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved: {out_path}")


import pandas as pd


def main():
    print("\n" + "=" * 65)
    print("BUFFER ANALYSIS DEMONSTRATION")
    print("Author: Emmanuel Oyekanlu — Principal Data Engineer")
    print("=" * 65 + "\n")

    # Load and project data
    parcels_utm, wells_utm = load_data()

    # --- Well buffers ---
    well_buffer_dict = create_well_buffers(wells_utm, WELL_BUFFER_DISTANCES_M)

    # Filter to just well-type features
    wells_only = wells_utm[wells_utm['type'] == 'well'].copy()

    # Dissolve overlapping well buffers
    well_buffers_dissolved = dissolve_overlapping_buffers(
        {d: gdf[gdf['type'] == 'well'] for d, gdf in well_buffer_dict.items()},
        'well'
    )

    # Find parcels within each well buffer zone
    print("\n" + "=" * 65)
    print("PARCELS WITHIN WELL BUFFER ZONES")
    print("=" * 65)

    for dist in WELL_BUFFER_DISTANCES_M:
        in_zone = find_parcels_in_buffer_zone(
            parcels_utm, well_buffers_dissolved, dist, 'well'
        )
        print(f"\n  Parcels within {dist}m of any well: {len(in_zone)}")
        if len(in_zone) > 0:
            print(f"    {list(in_zone['parcel_id'])}")
            print(f"    Total area in zone: {in_zone['area_ha'].sum():.1f} ha")

    # --- Canal buffers ---
    canal_buffer_dict = create_canal_buffers(CANAL_BUFFER_DISTANCES_M)

    # --- Multi-distance rings ---
    print("\n" + "=" * 65)
    print("MULTI-DISTANCE ANNULAR RINGS (Wellhead Protection)")
    print("=" * 65)

    rings_gdf = create_multi_distance_rings(wells_only, WELL_BUFFER_DISTANCES_M)
    print(f"\n  Generated {len(rings_gdf)} annular ring polygons:")
    print(rings_gdf[['source_id', 'ring_label', 'ring_area_ha']].to_string())

    # Save outputs
    rings_gdf.to_crs('EPSG:4326').to_file(
        os.path.join(OUTPUT_DIR, "well_protection_rings.geojson"),
        driver='GeoJSON'
    )
    well_buffers_dissolved.to_crs('EPSG:4326').to_file(
        os.path.join(OUTPUT_DIR, "well_buffers_dissolved.geojson"),
        driver='GeoJSON'
    )

    # Visualize
    visualize_buffers(parcels_utm, well_buffers_dissolved, wells_only)

    print("\nBuffer analysis complete. Outputs in output/ directory.")


if __name__ == "__main__":
    main()
