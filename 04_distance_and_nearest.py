"""
04_distance_and_nearest.py
===========================
Author: Emmanuel Oyekanlu — Principal Data Engineer

Demonstrates distance computation and nearest neighbor analysis:
  - Full pairwise distance matrix between field centroids
  - Nearest neighbor for each field (closest other field)
  - Distance from each field to nearest water source
  - Using scipy.spatial.cKDTree for efficient nearest neighbor queries

Use case context:
    Distance analysis is central to logistics optimization in agricultural
    data pipelines:
      - Route optimization: which fields should be served by which AGV/AMR?
      - Irrigation planning: which fields are beyond economic reach of a water source?
      - Supply chain: distance from field to nearest processing facility
      - Precision ag: identifying isolated fields needing dedicated equipment

Performance note:
    For n features, a brute-force distance matrix is O(n²).
    KDTree nearest-neighbor queries are O(n log n) — preferred for large datasets.
    For very large datasets (>100k points), use Faiss or Annoy approximate NN.

Run:
    python 04_distance_and_nearest.py
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from shapely.geometry import LineString

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PARCELS_PATH = os.path.join(DATA_DIR, "farm_parcels.geojson")
WATER_PATH = os.path.join(DATA_DIR, "water_sources.geojson")


def load_and_project() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load data and project to metric CRS for accurate distance calculations."""
    parcels = gpd.read_file(PARCELS_PATH)
    water = gpd.read_file(WATER_PATH)

    # Project to UTM Zone 10N for metric distances
    parcels_utm = parcels.to_crs('EPSG:32610')
    water_utm = water.to_crs('EPSG:32610')

    # Add centroid geometry for parcel-to-parcel distance
    parcels_utm['centroid_x'] = parcels_utm.geometry.centroid.x
    parcels_utm['centroid_y'] = parcels_utm.geometry.centroid.y

    print("=" * 65)
    print("DATA LOADED AND PROJECTED TO UTM ZONE 10N")
    print("=" * 65)
    print(f"  Farm parcels  : {len(parcels_utm)}")
    print(f"  Water sources : {len(water_utm)}")
    print(f"  CRS           : {parcels_utm.crs}\n")

    return parcels_utm, water_utm


def compute_pairwise_distance_matrix(
    parcels: gpd.GeoDataFrame
) -> pd.DataFrame:
    """
    Compute the full pairwise centroid-to-centroid distance matrix.

    Uses numpy broadcasting for vectorized computation:
        dist[i,j] = sqrt((x[i]-x[j])² + (y[i]-y[j])²)

    This is O(n²) — acceptable for small datasets (<1000 features).
    For larger datasets, use KDTree or spatial indexing.

    Parameters
    ----------
    parcels : GeoDataFrame
        Farm parcel features in a metric CRS.

    Returns
    -------
    DataFrame
        n×n distance matrix (meters). Row/column labels are parcel_ids.
    """
    print("=" * 65)
    print("PAIRWISE DISTANCE MATRIX (centroid-to-centroid, meters)")
    print("=" * 65)

    # Extract centroid coordinates as numpy arrays
    coords = parcels[['centroid_x', 'centroid_y']].values  # shape (n, 2)
    parcel_ids = parcels['parcel_id'].tolist()

    # Vectorized pairwise distance using broadcasting
    # diff shape: (n, 1, 2) - (1, n, 2) = (n, n, 2)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=2))  # shape (n, n)

    # Build labeled DataFrame
    dist_df = pd.DataFrame(
        dist_matrix,
        index=parcel_ids,
        columns=parcel_ids
    )

    print(f"\nDistance matrix shape: {dist_df.shape}")
    print(f"Min non-zero distance : {dist_matrix[dist_matrix > 0].min():.0f} m")
    print(f"Max distance          : {dist_matrix.max():.0f} m")
    print(f"Mean distance         : {dist_matrix[dist_matrix > 0].mean():.0f} m\n")

    # Print formatted matrix (in km for readability)
    dist_km = dist_df / 1000
    print("Distance matrix (km):")
    pd.set_option('display.float_format', '{:.2f}'.format)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(dist_km.round(2).to_string())

    return dist_df


def find_nearest_neighbor_field(
    parcels: gpd.GeoDataFrame,
    dist_matrix: pd.DataFrame
) -> pd.DataFrame:
    """
    For each parcel, find its nearest neighboring parcel.

    Uses the precomputed distance matrix — zero out diagonal first
    to exclude self-matches, then find argmin per row.

    Returns a DataFrame with:
      - parcel_id: source parcel
      - nearest_id: ID of nearest neighbor
      - distance_m: distance in meters
      - nearest_crop: crop type of nearest neighbor

    Parameters
    ----------
    parcels : GeoDataFrame
        Farm parcel features.
    dist_matrix : DataFrame
        Pairwise distance matrix from compute_pairwise_distance_matrix().

    Returns
    -------
    DataFrame
        Nearest neighbor results per parcel.
    """
    print("\n" + "=" * 65)
    print("NEAREST NEIGHBOR — EACH FIELD TO CLOSEST OTHER FIELD")
    print("=" * 65)

    # Set diagonal to infinity so self-match is never chosen
    dist_np = dist_matrix.values.copy()
    np.fill_diagonal(dist_np, np.inf)

    parcel_ids = dist_matrix.index.tolist()
    crop_lookup = dict(zip(parcels['parcel_id'], parcels['crop_type']))
    owner_lookup = dict(zip(parcels['parcel_id'], parcels['owner']))

    results = []
    for i, src_id in enumerate(parcel_ids):
        nearest_idx = np.argmin(dist_np[i])
        nearest_id = parcel_ids[nearest_idx]
        distance_m = dist_np[i, nearest_idx]

        results.append({
            'parcel_id': src_id,
            'nearest_id': nearest_id,
            'distance_m': round(distance_m, 1),
            'distance_km': round(distance_m / 1000, 3),
            'source_crop': crop_lookup.get(src_id, '?'),
            'nearest_crop': crop_lookup.get(nearest_id, '?'),
            'same_owner': owner_lookup.get(src_id) == owner_lookup.get(nearest_id),
        })

    nn_df = pd.DataFrame(results)

    print(f"\n{'Parcel':<10} {'Nearest':<10} {'Dist(m)':>9} {'Dist(km)':>9} "
          f"{'Src Crop':<12} {'Near Crop':<12} {'Same Owner'}")
    print("-" * 80)
    for _, row in nn_df.iterrows():
        print(f"{row['parcel_id']:<10} {row['nearest_id']:<10} "
              f"{row['distance_m']:>9.1f} {row['distance_km']:>9.3f} "
              f"{row['source_crop']:<12} {row['nearest_crop']:<12} "
              f"{'YES' if row['same_owner'] else 'no'}")

    print(f"\nSummary:")
    print(f"  Average nearest neighbor distance: {nn_df['distance_m'].mean():.0f} m")
    print(f"  Minimum NN distance              : {nn_df['distance_m'].min():.0f} m")
    print(f"  Maximum NN distance              : {nn_df['distance_m'].max():.0f} m")
    print(f"  Fields with same-owner neighbor  : {nn_df['same_owner'].sum()}")

    return nn_df


def find_nearest_water_source_kdtree(
    parcels: gpd.GeoDataFrame,
    water: gpd.GeoDataFrame
) -> pd.DataFrame:
    """
    Find the nearest water source for each field using scipy cKDTree.

    cKDTree is a k-dimensional tree data structure that enables O(log n)
    nearest neighbor lookups. For n parcels and m water sources:
      - Brute force: O(n×m) — 12 × 6 = 72 distance calculations
      - cKDTree: O(n × log(m)) — scales to millions of parcels

    This pattern is essential for:
      - Assigning irrigation water sources to fields in a data pipeline
      - Determining which field each sensor reading belongs to
      - Finding nearest charging station for AGV battery management

    Parameters
    ----------
    parcels : GeoDataFrame
        Farm parcel centroids in metric CRS.
    water : GeoDataFrame
        Water source points in metric CRS.

    Returns
    -------
    DataFrame
        Each parcel with nearest water source ID, type, distance.
    """
    print("\n" + "=" * 65)
    print("NEAREST WATER SOURCE — cKDTree ALGORITHM")
    print("=" * 65)

    # Extract parcel centroid coordinates
    parcel_coords = np.array([
        [row.centroid_x, row.centroid_y]
        for _, row in parcels.iterrows()
    ])

    # Extract water source coordinates
    water_coords = np.array([
        [row.geometry.x, row.geometry.y]
        for _, row in water.iterrows()
    ])

    # Build KD-tree on water sources
    # Query with k=1 returns the single nearest neighbor
    # For k=3, returns 3 nearest in sorted order
    tree = cKDTree(water_coords)

    # Query for k=3 nearest water sources per parcel
    distances, indices = tree.query(parcel_coords, k=min(3, len(water)))

    # Build result table
    results = []
    for i, (_, parcel_row) in enumerate(parcels.iterrows()):
        # Primary (nearest) water source
        nearest_idx = indices[i] if distances.ndim == 1 else indices[i, 0]
        nearest_dist = distances[i] if distances.ndim == 1 else distances[i, 0]
        nearest_water = water.iloc[nearest_idx]

        row_result = {
            'parcel_id': parcel_row['parcel_id'],
            'crop_type': parcel_row['crop_type'],
            'nearest_water_id': nearest_water['source_id'],
            'nearest_water_type': nearest_water['type'],
            'distance_m': round(float(nearest_dist), 1),
            'distance_km': round(float(nearest_dist) / 1000, 3),
            'water_capacity_m3': nearest_water['capacity_m3'],
            'water_active': nearest_water['active'],
        }

        # Second nearest (alternative source)
        if distances.ndim > 1 and indices.shape[1] > 1:
            second_idx = indices[i, 1]
            second_water = water.iloc[second_idx]
            row_result['second_water_id'] = second_water['source_id']
            row_result['second_dist_m'] = round(float(distances[i, 1]), 1)

        results.append(row_result)

    result_df = pd.DataFrame(results)

    print(f"\nNearest water source per field:")
    print(f"\n{'Parcel':<10} {'Crop':<12} {'Water Source':<12} {'Type':<8} "
          f"{'Dist(m)':>8} {'Capacity(m3)':>13} {'Active'}")
    print("-" * 85)
    for _, row in result_df.iterrows():
        print(f"{row['parcel_id']:<10} {row['crop_type']:<12} "
              f"{row['nearest_water_id']:<12} {row['nearest_water_type']:<8} "
              f"{row['distance_m']:>8.1f} {row['water_capacity_m3']:>13,.0f} "
              f"{'YES' if row['water_active'] else 'NO'}")

    print(f"\nSummary:")
    print(f"  Average distance to water  : {result_df['distance_m'].mean():.0f} m")
    print(f"  Max distance to water      : {result_df['distance_m'].max():.0f} m")
    print(f"  Fields nearest inactive src: "
          f"{(~result_df['water_active']).sum()} "
          f"(need to check backup source)")

    # Fields beyond 2km from water — may need infrastructure investment
    far_fields = result_df[result_df['distance_m'] > 2000]
    if len(far_fields) > 0:
        print(f"\n  Fields >2km from any water source (infrastructure concern):")
        print(far_fields[['parcel_id', 'crop_type', 'distance_m']].to_string())

    return result_df


def visualize_distance_analysis(
    parcels: gpd.GeoDataFrame,
    water: gpd.GeoDataFrame,
    nn_df: pd.DataFrame,
    water_dist_df: pd.DataFrame
) -> None:
    """Visualize nearest neighbor connections and water source distances."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # --- Plot 1: Nearest neighbor connections ---
    ax1 = axes[0]
    parcels.plot(ax=ax1, color='#4CAF50', alpha=0.5,
                 edgecolor='darkgreen', linewidth=0.8)

    # Build lookup for centroid coordinates
    centroid_lookup = {
        row['parcel_id']: (row['centroid_x'], row['centroid_y'])
        for _, row in parcels.iterrows()
    }

    # Draw lines from each parcel to nearest neighbor
    for _, row in nn_df.iterrows():
        src_coords = centroid_lookup.get(row['parcel_id'])
        dst_coords = centroid_lookup.get(row['nearest_id'])
        if src_coords and dst_coords:
            ax1.plot(
                [src_coords[0], dst_coords[0]],
                [src_coords[1], dst_coords[1]],
                'b--', alpha=0.5, linewidth=1.2
            )

    # Plot centroids
    cx = [v[0] for v in centroid_lookup.values()]
    cy = [v[1] for v in centroid_lookup.values()]
    ax1.scatter(cx, cy, color='darkgreen', s=40, zorder=5)

    # Annotate
    for pid, (x, y) in centroid_lookup.items():
        ax1.annotate(pid.replace('CV-', ''), xy=(x, y),
                     xytext=(3, 3), textcoords='offset points',
                     fontsize=7, color='darkgreen')

    ax1.set_title('Nearest Neighbor Connections\n(dashed lines = NN pairs)',
                  fontsize=11, fontweight='bold')
    ax1.set_xlabel('Easting (m)')
    ax1.set_ylabel('Northing (m)')

    # --- Plot 2: Distance to nearest water source ---
    ax2 = axes[1]

    # Color parcels by distance to water
    parcels_with_dist = parcels.merge(
        water_dist_df[['parcel_id', 'distance_m']],
        on='parcel_id'
    )

    parcels_with_dist.plot(
        column='distance_m',
        ax=ax2,
        cmap='RdYlGn_r',   # Red = far, Green = close
        legend=True,
        legend_kwds={'label': 'Distance to Water (m)', 'shrink': 0.6},
        edgecolor='white',
        linewidth=0.8
    )

    # Plot water sources
    water.plot(ax=ax2, color='blue', markersize=80,
               marker='o', zorder=5, label='Water Sources')

    ax2.set_title('Distance to Nearest Water Source\n(Red = far, Green = close)',
                  fontsize=11, fontweight='bold')
    ax2.set_xlabel('Easting (m)')
    ax2.set_ylabel('Northing (m)')

    fig.suptitle(
        'Distance & Nearest Neighbor Analysis — Central Valley CA\n'
        'Author: Emmanuel Oyekanlu',
        fontsize=12, fontweight='bold'
    )

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "distance_analysis.png")
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved: {out_path}")


def main():
    print("\n" + "=" * 65)
    print("DISTANCE & NEAREST NEIGHBOR ANALYSIS")
    print("Author: Emmanuel Oyekanlu — Principal Data Engineer")
    print("=" * 65 + "\n")

    # Load and project
    parcels, water = load_and_project()

    # Compute pairwise distance matrix
    dist_matrix = compute_pairwise_distance_matrix(parcels)

    # Find nearest neighbor per field
    nn_df = find_nearest_neighbor_field(parcels, dist_matrix)

    # Find nearest water source per field using KDTree
    water_dist_df = find_nearest_water_source_kdtree(parcels, water)

    # Visualize
    visualize_distance_analysis(parcels, water, nn_df, water_dist_df)

    # Save results
    nn_df.to_csv(os.path.join(OUTPUT_DIR, "nearest_neighbor_results.csv"), index=False)
    water_dist_df.to_csv(
        os.path.join(OUTPUT_DIR, "water_source_distances.csv"), index=False
    )
    print("\nCSV results saved to output/ directory.")


if __name__ == "__main__":
    main()
