"""
06_zonal_statistics.py
=======================
Author: Emmanuel Oyekanlu — Principal Data Engineer

Demonstrates zonal statistics computation:
  - Simulate a raster grid over the study area (NDVI, soil moisture, etc.)
  - For each polygon zone (farm parcel), extract all grid cells that fall within it
  - Compute per-zone statistics: mean, min, max, std, percentiles
  - Generate a choropleth map of zonal mean values
  - Demonstrate the rasterio/rasterstats equivalent approach using numpy grids

In production, rasterstats.zonal_stats() reads actual GeoTIFF raster files.
This script simulates that approach using numpy arrays — demonstrating the
algorithm and pipeline without requiring specific raster data files.

Zonal statistics use cases in precision agriculture:
  - NDVI zonal stats per field from Sentinel-2 imagery (crop health monitoring)
  - Soil moisture zonal stats from SMAP satellite per management zone
  - Growing Degree Days (GDD) from gridded weather data per field
  - Elevation statistics from DEM per field (drainage analysis)
  - Yield monitor point density per management zone

Run:
    python 06_zonal_statistics.py
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import Point, box
from shapely.vectorized import contains  # Fast vectorized point-in-polygon

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PARCELS_PATH = os.path.join(DATA_DIR, "farm_parcels.geojson")

# ---------------------------------------------------------------------------
# Raster grid configuration
# ---------------------------------------------------------------------------
# We'll create a simulated raster covering the study area
# Grid resolution: 100m cells (in UTM Zone 10N, meters)
GRID_RESOLUTION_M = 100   # 100 meter cell size

# Simulated variables to compute zonal stats for
SIMULATED_VARIABLES = ['ndvi', 'soil_moisture', 'surface_temp_c', 'elevation_m']


def load_and_project_parcels() -> gpd.GeoDataFrame:
    """Load parcels and project to UTM for metric grid alignment."""
    parcels = gpd.read_file(PARCELS_PATH)
    parcels_utm = parcels.to_crs('EPSG:32610')

    print("=" * 65)
    print("PARCELS LOADED AND PROJECTED")
    print("=" * 65)
    print(f"  {len(parcels_utm)} parcels in UTM Zone 10N")
    print(f"  Bounds: {parcels_utm.total_bounds.round(0)}")
    return parcels_utm


def create_simulated_raster(
    bounds: tuple[float, float, float, float],
    resolution_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Create a simulated raster grid of multiple environmental variables.

    Uses numpy to generate spatially correlated random fields that mimic
    real remote sensing data characteristics:
      - NDVI: 0.1 (bare soil) to 0.9 (dense vegetation), crop-type dependent
      - Soil moisture: 0.1 to 0.5 m³/m³, higher near water sources
      - Surface temperature: 25-45°C, inversely related to NDVI
      - Elevation: smooth gradient (typical of Central Valley alluvial plain)

    Spatial correlation is introduced via a 2D Gaussian filter that creates
    patches of similar values — more realistic than purely random noise.

    Parameters
    ----------
    bounds : tuple
        (minx, miny, maxx, maxy) in metric CRS units.
    resolution_m : float
        Grid cell size in meters.

    Returns
    -------
    tuple
        (xx, yy, raster_data, metadata)
        xx/yy: coordinate meshgrids
        raster_data: dict of variable_name → 2D array
        metadata: grid metadata dict
    """
    minx, miny, maxx, maxy = bounds

    # Add 500m padding around the parcel extent
    pad = 500
    minx -= pad; miny -= pad; maxx += pad; maxy += pad

    # Generate grid coordinates
    x_coords = np.arange(minx, maxx, resolution_m)
    y_coords = np.arange(miny, maxy, resolution_m)
    xx, yy = np.meshgrid(x_coords, y_coords)

    nrows, ncols = xx.shape

    print(f"\n  Grid dimensions: {nrows} rows × {ncols} cols "
          f"({nrows * ncols:,} cells at {resolution_m}m resolution)")
    print(f"  Grid extent: ({minx:.0f}, {miny:.0f}) to ({maxx:.0f}, {maxy:.0f})")

    np.random.seed(42)  # Reproducibility

    def smooth_field(base_values, sigma_cells=3):
        """Apply spatial smoothing to create realistic spatial correlation."""
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(base_values, sigma=sigma_cells)

    # --- NDVI: 0.1 (bare soil) to 0.85 (dense crop) ---
    # Higher in center (irrigated fields), lower at edges
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2
    dist_from_center = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)
    ndvi_base = 0.7 - 0.3 * (dist_from_center / dist_from_center.max())
    ndvi_noise = np.random.normal(0, 0.08, (nrows, ncols))
    ndvi = np.clip(smooth_field(ndvi_base + ndvi_noise), 0.05, 0.92)

    # --- Soil Moisture: inversely correlated with distance from south edge ---
    # South end dryer (simulating irrigation gradient)
    y_gradient = (yy - miny) / (maxy - miny)
    sm_base = 0.15 + 0.25 * y_gradient
    sm_noise = np.random.normal(0, 0.03, (nrows, ncols))
    soil_moisture = np.clip(smooth_field(sm_base + sm_noise, sigma_cells=4), 0.05, 0.55)

    # --- Surface Temperature: inversely related to NDVI (vegetation cools) ---
    temp_base = 42 - 18 * ndvi
    temp_noise = np.random.normal(0, 1.2, (nrows, ncols))
    surface_temp = smooth_field(temp_base + temp_noise, sigma_cells=2)

    # --- Elevation: gentle north-south gradient (alluvial plain) ---
    elevation_base = 80 + 15 * ((yy - miny) / (maxy - miny))
    elevation_noise = np.random.normal(0, 2, (nrows, ncols))
    elevation = smooth_field(elevation_base + elevation_noise, sigma_cells=5)

    raster_data = {
        'ndvi': ndvi,
        'soil_moisture': soil_moisture,
        'surface_temp_c': surface_temp,
        'elevation_m': elevation,
    }

    metadata = {
        'minx': minx, 'miny': miny, 'maxx': maxx, 'maxy': maxy,
        'resolution_m': resolution_m,
        'nrows': nrows, 'ncols': ncols,
        'x_coords': x_coords,
        'y_coords': y_coords,
    }

    return xx, yy, raster_data, metadata


def compute_zonal_statistics(
    parcels: gpd.GeoDataFrame,
    xx: np.ndarray,
    yy: np.ndarray,
    raster_data: dict,
    metadata: dict
) -> pd.DataFrame:
    """
    Compute zonal statistics: for each polygon, extract raster cells within it
    and compute summary statistics.

    Algorithm:
        1. Get bounding box of polygon
        2. Subset grid to bounding box (fast pre-filter)
        3. For each cell in the subset, test if cell center is within polygon
        4. Collect values of cells passing the point-in-polygon test
        5. Compute statistics on collected values

    This algorithm runs in O(n_parcels × n_cells_per_bbox) time.
    For production with large rasters, use rasterstats or rioxarray which
    use optimized C implementations via GDAL.

    Parameters
    ----------
    parcels : GeoDataFrame
        Polygon zones in the same CRS as the grid.
    xx, yy : ndarray
        Coordinate meshgrids.
    raster_data : dict
        Variable name → 2D array of values.
    metadata : dict
        Grid metadata from create_simulated_raster().

    Returns
    -------
    DataFrame
        One row per parcel with mean/min/max/std for each variable.
    """
    print("\n" + "=" * 65)
    print("COMPUTING ZONAL STATISTICS")
    print("=" * 65)

    results = []
    x_res = metadata['resolution_m']
    x_coords = metadata['x_coords']
    y_coords = metadata['y_coords']

    for idx, parcel_row in parcels.iterrows():
        parcel_geom = parcel_row['geometry']
        parcel_id = parcel_row['parcel_id']

        # Step 1: Get polygon bounding box
        minx, miny, maxx, maxy = parcel_geom.bounds

        # Step 2: Find grid indices within bounding box
        x_mask = (x_coords >= minx - x_res) & (x_coords <= maxx + x_res)
        y_mask = (y_coords >= miny - x_res) & (y_coords <= maxy + x_res)

        x_indices = np.where(x_mask)[0]
        y_indices = np.where(y_mask)[0]

        if len(x_indices) == 0 or len(y_indices) == 0:
            print(f"  WARNING: No grid cells found for {parcel_id}")
            continue

        # Step 3: Extract subset grid coordinates
        sub_xx = xx[np.ix_(y_indices, x_indices)]  # Note: rows=y, cols=x
        sub_yy = yy[np.ix_(y_indices, x_indices)]

        # Step 4: Point-in-polygon test for each cell center
        # Use shapely vectorized contains for speed
        flat_x = sub_xx.flatten()
        flat_y = sub_yy.flatten()

        # Test which cell centers are inside the polygon
        # shapely.vectorized.contains is much faster than iterating
        try:
            in_polygon = contains(parcel_geom, flat_x, flat_y)
        except Exception:
            # Fallback: test each point individually
            in_polygon = np.array([
                parcel_geom.contains(Point(x, y))
                for x, y in zip(flat_x, flat_y)
            ])

        n_cells = int(in_polygon.sum())

        if n_cells == 0:
            print(f"  WARNING: Zero cells inside {parcel_id} — polygon may be smaller than grid")
            # Use centroid cell as fallback
            centroid = parcel_geom.centroid
            n_cells = 1
            fallback = True
        else:
            fallback = False

        # Step 5: Compute statistics for each variable
        row_stats = {
            'parcel_id': parcel_id,
            'crop_type': parcel_row['crop_type'],
            'area_ha': parcel_row['area_ha'],
            'n_cells': n_cells,
            'cell_coverage_ha': n_cells * (x_res ** 2) / 10_000,
        }

        for var_name, var_grid in raster_data.items():
            # Extract subset of this variable's grid
            sub_var = var_grid[np.ix_(y_indices, x_indices)].flatten()

            if not fallback:
                # Use only cells within the polygon
                zone_values = sub_var[in_polygon]
            else:
                # Fallback: use nearest cell to centroid
                zone_values = np.array([sub_var.mean()])

            # Compute statistics
            row_stats[f'{var_name}_mean'] = round(float(np.mean(zone_values)), 4)
            row_stats[f'{var_name}_min']  = round(float(np.min(zone_values)), 4)
            row_stats[f'{var_name}_max']  = round(float(np.max(zone_values)), 4)
            row_stats[f'{var_name}_std']  = round(float(np.std(zone_values)), 4)
            row_stats[f'{var_name}_p25']  = round(float(np.percentile(zone_values, 25)), 4)
            row_stats[f'{var_name}_p75']  = round(float(np.percentile(zone_values, 75)), 4)

        results.append(row_stats)
        print(f"  {parcel_id}: {n_cells} cells | "
              f"NDVI={row_stats['ndvi_mean']:.3f} | "
              f"SM={row_stats['soil_moisture_mean']:.3f} | "
              f"T={row_stats['surface_temp_c_mean']:.1f}°C | "
              f"Elev={row_stats['elevation_m_mean']:.0f}m")

    return pd.DataFrame(results)


def print_zonal_stats_report(zonal_df: pd.DataFrame) -> None:
    """Print a formatted zonal statistics report."""
    print("\n" + "=" * 65)
    print("ZONAL STATISTICS SUMMARY REPORT")
    print("=" * 65)

    for var in SIMULATED_VARIABLES:
        mean_col = f'{var}_mean'
        if mean_col not in zonal_df.columns:
            continue

        print(f"\n  [{var.upper().replace('_', ' ')}]")
        print(f"  {'Parcel':<10} {'Crop':<12} {'Mean':>8} {'Min':>8} "
              f"{'Max':>8} {'Std':>8}")
        print(f"  {'-'*10} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        for _, row in zonal_df.sort_values(mean_col, ascending=False).iterrows():
            print(f"  {row['parcel_id']:<10} {row['crop_type']:<12} "
                  f"{row[f'{var}_mean']:>8.3f} "
                  f"{row[f'{var}_min']:>8.3f} "
                  f"{row[f'{var}_max']:>8.3f} "
                  f"{row[f'{var}_std']:>8.3f}")

        regional_mean = zonal_df[mean_col].mean()
        print(f"\n  Regional mean {var}: {regional_mean:.4f}")

    # Crop-type level statistics (dissolve equivalent)
    print("\n" + "=" * 65)
    print("CROP-TYPE LEVEL ZONAL STATISTICS (aggregated)")
    print("=" * 65)

    crop_stats = zonal_df.groupby('crop_type').agg(
        field_count=('parcel_id', 'count'),
        mean_ndvi=('ndvi_mean', 'mean'),
        mean_soil_moisture=('soil_moisture_mean', 'mean'),
        mean_temp=('surface_temp_c_mean', 'mean'),
        mean_elevation=('elevation_m_mean', 'mean'),
    ).round(3)

    print(f"\n{'Crop':<12} {'Fields':>7} {'NDVI':>7} {'SoilMoist':>10} "
          f"{'Temp(°C)':>9} {'Elev(m)':>8}")
    print("-" * 60)
    for crop, row in crop_stats.iterrows():
        print(f"{crop:<12} {int(row['field_count']):>7} "
              f"{row['mean_ndvi']:>7.3f} "
              f"{row['mean_soil_moisture']:>10.3f} "
              f"{row['mean_temp']:>9.1f} "
              f"{row['mean_elevation']:>8.1f}")


def visualize_zonal_stats(
    parcels: gpd.GeoDataFrame,
    zonal_df: pd.DataFrame,
    xx: np.ndarray,
    yy: np.ndarray,
    raster_data: dict
) -> None:
    """
    Visualize zonal statistics: raster background with parcel choropleth overlay.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(
        'Zonal Statistics — Simulated Environmental Variables\n'
        'Central Valley, CA | Author: Emmanuel Oyekanlu',
        fontsize=13, fontweight='bold', y=0.98
    )

    variables = [
        ('ndvi', 'NDVI (Vegetation Index)', 'RdYlGn', (0.1, 0.9)),
        ('soil_moisture', 'Soil Moisture (m³/m³)', 'Blues', (0.05, 0.55)),
        ('surface_temp_c', 'Surface Temp (°C)', 'RdBu_r', (25, 45)),
        ('elevation_m', 'Elevation (m)', 'terrain', (75, 100)),
    ]

    # Merge zonal stats onto parcels GeoDataFrame
    parcels_stats = parcels.merge(zonal_df, on='parcel_id', how='left',
                                   suffixes=('', '_stats'))

    for ax, (var, title, cmap, vrange) in zip(axes.flat, variables):
        # Background: plot the raster grid
        ax.pcolormesh(
            xx, yy, raster_data[var],
            cmap=cmap, vmin=vrange[0], vmax=vrange[1],
            alpha=0.5, shading='auto'
        )

        # Foreground: choropleth of parcel zonal means
        mean_col = f'{var}_mean'
        if mean_col in parcels_stats.columns:
            parcels_stats.plot(
                column=mean_col,
                ax=ax,
                cmap=cmap,
                vmin=vrange[0], vmax=vrange[1],
                edgecolor='white',
                linewidth=1.2,
                legend=True,
                legend_kwds={
                    'label': f'Zonal mean {var}',
                    'shrink': 0.5,
                    'orientation': 'horizontal'
                },
                alpha=0.85
            )

            # Label each parcel with its zonal mean
            for _, row in parcels_stats.iterrows():
                if pd.notna(row.get(mean_col)):
                    centroid = row['geometry'].centroid
                    ax.annotate(
                        f"{row[mean_col]:.2f}",
                        xy=(centroid.x, centroid.y),
                        ha='center', va='center',
                        fontsize=6.5, color='black', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.5)
                    )

        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Easting (m)', fontsize=8)
        ax.set_ylabel('Northing (m)', fontsize=8)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "zonal_statistics.png")
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved: {out_path}")


def main():
    print("\n" + "=" * 65)
    print("ZONAL STATISTICS DEMONSTRATION")
    print("Author: Emmanuel Oyekanlu — Principal Data Engineer")
    print("=" * 65 + "\n")

    from scipy.ndimage import gaussian_filter  # Ensure available

    # Load parcels
    parcels = load_and_project_parcels()

    # Create simulated raster
    print("\n" + "=" * 65)
    print("CREATING SIMULATED RASTER GRIDS")
    print("=" * 65)
    xx, yy, raster_data, metadata = create_simulated_raster(
        bounds=tuple(parcels.total_bounds),
        resolution_m=GRID_RESOLUTION_M
    )
    for var, grid in raster_data.items():
        print(f"  {var:<20}: shape={grid.shape}, "
              f"min={grid.min():.3f}, max={grid.max():.3f}, mean={grid.mean():.3f}")

    # Compute zonal statistics
    zonal_df = compute_zonal_statistics(parcels, xx, yy, raster_data, metadata)

    # Print report
    print_zonal_stats_report(zonal_df)

    # Visualize
    visualize_zonal_stats(parcels, zonal_df, xx, yy, raster_data)

    # Save results
    zonal_df.to_csv(
        os.path.join(OUTPUT_DIR, "zonal_statistics.csv"),
        index=False
    )

    # Merge and save as GeoJSON
    parcels_enriched = parcels.merge(zonal_df, on='parcel_id', how='left',
                                      suffixes=('', '_stat'))
    parcels_enriched.to_crs('EPSG:4326').to_file(
        os.path.join(OUTPUT_DIR, "parcels_with_zonal_stats.geojson"),
        driver='GeoJSON'
    )

    print(f"\nOutputs saved to output/ directory.")
    print(f"  zonal_statistics.csv")
    print(f"  parcels_with_zonal_stats.geojson")
    print(f"  zonal_statistics.png")


if __name__ == "__main__":
    main()
