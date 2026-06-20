import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon, FancyArrow, FancyBboxPatch
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import os

os.makedirs('images', exist_ok=True)

# ── Colour helpers ──────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

# ============================================================
# 1.  spatial_join_counties.png
# ============================================================
def make_spatial_join():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 9))
    fig.patch.set_facecolor('#f8f9fa')
    for ax in (ax1, ax2):
        ax.set_facecolor('#eef3f7')

    # --- 12 irregular farm parcel polygons (hand-crafted vertices) ----------
    parcels_raw = [
        # Fresno county area  (x 0..6, y 5..10)
        [(0.3,5.2),(1.5,5.0),(1.8,6.3),(0.8,6.8),(0.2,6.1)],
        [(1.9,5.1),(3.1,4.9),(3.4,6.0),(2.2,6.5),(1.7,5.8)],
        [(3.3,5.3),(4.8,5.0),(5.1,6.2),(3.9,6.7),(3.0,6.2)],
        [(0.5,7.0),(2.0,6.9),(2.2,8.2),(1.1,8.7),(0.3,7.8)],
        # Kings county area (x 0..6, y 0..5)
        [(0.2,2.5),(1.8,2.2),(2.0,3.8),(0.9,4.3),(0.1,3.5)],
        [(2.1,2.0),(3.6,1.8),(3.9,3.2),(2.7,3.7),(1.9,3.0)],
        [(4.0,2.3),(5.5,2.0),(5.8,3.4),(4.5,3.9),(3.7,3.2)],
        # Tulare county area (x 6..12, y 0..10)
        [(6.3,7.0),(7.8,6.8),(8.2,8.3),(7.0,8.9),(6.1,8.0)],
        [(8.3,7.2),(9.8,7.0),(10.2,8.4),(9.0,8.8),(8.0,7.8)],
        [(6.5,3.5),(8.0,3.2),(8.4,4.8),(7.2,5.3),(6.2,4.5)],
        [(8.5,3.8),(10.0,3.5),(10.4,5.0),(9.2,5.5),(8.2,4.6)],
        [(10.2,3.0),(11.7,2.8),(12.0,4.3),(10.9,4.8),(9.9,3.9)],
    ]

    county_county = ['Fresno','Fresno','Fresno','Fresno',
                     'Kings','Kings','Kings',
                     'Tulare','Tulare','Tulare','Tulare','Tulare']

    county_colors = {'Fresno':'coral','Kings':'lightblue','Tulare':'lightgreen'}

    # County boundary polygons
    county_polys = {
        'Fresno': [(0.0,4.8),(6.0,4.8),(6.0,10.0),(0.0,10.0)],
        'Kings':  [(0.0,0.0),(6.0,0.0),(6.0,4.8),(0.0,4.8)],
        'Tulare': [(6.0,0.0),(12.2,0.0),(12.2,10.0),(6.0,10.0)],
    }
    county_label_pos = {'Fresno':(3.0,9.3),'Kings':(3.0,0.5),'Tulare':(9.1,9.3)}

    for ax_idx, ax in enumerate([ax1, ax2]):
        ax.set_xlim(-0.3, 12.5)
        ax.set_ylim(-0.3, 10.3)
        ax.set_aspect('equal')
        ax.set_xlabel('Easting (km)', fontsize=10)
        ax.set_ylabel('Northing (km)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4, color='gray')

        # Draw county outlines
        for cname, verts in county_polys.items():
            poly = Polygon(verts, closed=True, fill=False,
                           edgecolor='#333333', linewidth=2.5,
                           linestyle='--', zorder=3)
            ax.add_patch(poly)
            lx, ly = county_label_pos[cname]
            ax.text(lx, ly, cname, ha='center', va='center',
                    fontsize=12, fontweight='bold', color='#333333',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

        # Draw parcels
        for i, verts in enumerate(parcels_raw):
            if ax_idx == 0:
                fc = 'lightblue'
                ec = '#1565c0'
            else:
                fc = county_colors[county_county[i]]
                ec = '#555555'
            poly = Polygon(verts, closed=True, facecolor=fc,
                           edgecolor=ec, linewidth=1.2, alpha=0.8, zorder=4)
            ax.add_patch(poly)
            cx = np.mean([v[0] for v in verts])
            cy = np.mean([v[1] for v in verts])
            ax.text(cx, cy, f'P{i+1}', ha='center', va='center',
                    fontsize=7, color='#111111', fontweight='bold', zorder=5)

    ax1.set_title('Input: Parcels + County Boundaries', fontsize=13, fontweight='bold', pad=10)
    ax2.set_title('Result: Parcels Attributed by County', fontsize=13, fontweight='bold', pad=10)

    # Legend for ax2
    legend_patches = [mpatches.Patch(facecolor=v, edgecolor='#555', label=k)
                      for k, v in county_colors.items()]
    ax2.legend(handles=legend_patches, title='County', loc='lower right',
               fontsize=10, title_fontsize=10, framealpha=0.9)

    fig.suptitle('Spatial Join: Farm Parcels → County Boundaries', fontsize=15,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('images/spatial_join_counties.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  spatial_join_counties.png  ✓')


# ============================================================
# 2.  overlay_operations.png
# ============================================================
def make_overlay_operations():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#f8f9fa')
    titles = ['Intersection', 'Union', 'Difference (A − B)', 'Symmetric Difference']
    fill_colors = ['gold', 'teal', 'darkorange', 'mediumpurple']

    # Define rectangle A and approximate circle B as polygon
    rect_A = np.array([(1.0,1.5),(4.5,1.5),(4.5,5.0),(1.0,5.0)])
    theta = np.linspace(0, 2*np.pi, 60)
    cx, cy, r = 3.8, 3.5, 1.8
    circle_B = np.column_stack([cx + r*np.cos(theta), cy + r*np.sin(theta)])

    def poly_clip_intersection(rect, circ):
        """Approximate intersection polygon."""
        pts = []
        for p in circ:
            x, y = p
            if rect[0,0] <= x <= rect[1,0] and rect[0,1] <= y <= rect[2,1]:
                pts.append(p)
        for p in rect:
            x, y = p
            dist = np.sqrt((x-cx)**2 + (y-cy)**2)
            if dist <= r:
                pts.append(p)
        if len(pts) < 3:
            return None
        pts = np.array(pts)
        angles = np.arctan2(pts[:,1]-pts[:,1].mean(), pts[:,0]-pts[:,0].mean())
        return pts[np.argsort(angles)]

    inter = poly_clip_intersection(rect_A, circle_B)

    def draw_panel(ax, mode, color):
        ax.set_facecolor('#eef3f7')
        ax.set_xlim(-0.2, 7.0)
        ax.set_ylim(0.5, 6.5)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(labelsize=8)

        # Dashed outlines A & B always
        pA = Polygon(rect_A, closed=True, fill=False, edgecolor='#1565c0',
                     linewidth=2, linestyle='--', zorder=3)
        pB = Polygon(circle_B, closed=True, fill=False, edgecolor='#b71c1c',
                     linewidth=2, linestyle='--', zorder=3)
        ax.add_patch(pA)
        ax.add_patch(pB)
        ax.text(1.3, 4.7, 'A', fontsize=14, color='#1565c0', fontweight='bold')
        ax.text(5.0, 5.0, 'B', fontsize=14, color='#b71c1c', fontweight='bold')

        if mode == 'intersection' and inter is not None:
            p = Polygon(inter, closed=True, facecolor=color, edgecolor='#333',
                        linewidth=1.5, alpha=0.85, zorder=4)
            ax.add_patch(p)

        elif mode == 'union':
            # Draw rect A filled
            pAf = Polygon(rect_A, closed=True, facecolor=color, edgecolor='#333',
                          linewidth=1.5, alpha=0.7, zorder=2)
            pBf = Polygon(circle_B, closed=True, facecolor=color, edgecolor='#333',
                          linewidth=1.5, alpha=0.7, zorder=2)
            ax.add_patch(pAf)
            ax.add_patch(pBf)

        elif mode == 'difference':
            # A without intersection — draw A filled, then blank intersection
            pAf = Polygon(rect_A, closed=True, facecolor=color, edgecolor='#333',
                          linewidth=1.5, alpha=0.85, zorder=2)
            ax.add_patch(pAf)
            if inter is not None:
                pI = Polygon(inter, closed=True, facecolor='#eef3f7',
                             edgecolor='none', zorder=3)
                ax.add_patch(pI)

        elif mode == 'sym_diff':
            # A filled, B filled, intersection blanked
            pAf = Polygon(rect_A, closed=True, facecolor=color, edgecolor='#333',
                          linewidth=1.5, alpha=0.7, zorder=2)
            pBf = Polygon(circle_B, closed=True, facecolor=color, edgecolor='#333',
                          linewidth=1.5, alpha=0.7, zorder=2)
            ax.add_patch(pAf)
            ax.add_patch(pBf)
            if inter is not None:
                pI = Polygon(inter, closed=True, facecolor='#eef3f7',
                             edgecolor='none', zorder=3)
                ax.add_patch(pI)

        # Re-draw dashed outlines on top
        pA2 = Polygon(rect_A, closed=True, fill=False, edgecolor='#1565c0',
                      linewidth=2, linestyle='--', zorder=5)
        pB2 = Polygon(circle_B, closed=True, fill=False, edgecolor='#b71c1c',
                      linewidth=2, linestyle='--', zorder=5)
        ax.add_patch(pA2)
        ax.add_patch(pB2)

        # Patch for legend
        patch = mpatches.Patch(facecolor=color, edgecolor='#333', alpha=0.85,
                               label=f'Result region')
        patchA = mpatches.Patch(facecolor='none', edgecolor='#1565c0',
                                linestyle='--', label='Shape A')
        patchB = mpatches.Patch(facecolor='none', edgecolor='#b71c1c',
                                linestyle='--', label='Shape B')
        ax.legend(handles=[patchA, patchB, patch], fontsize=8, loc='lower left',
                  framealpha=0.9)

    modes = ['intersection', 'union', 'difference', 'sym_diff']
    for ax, mode, title, color in zip(axes.flat, modes, titles, fill_colors):
        draw_panel(ax, mode, color)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=8)

    fig.suptitle('Polygon Overlay Operations', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('images/overlay_operations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  overlay_operations.png  ✓')


# ============================================================
# 3.  buffer_and_distance.png
# ============================================================
def make_buffer_and_distance():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 8))
    fig.patch.set_facecolor('#f8f9fa')
    for ax in (ax1, ax2):
        ax.set_facecolor('#eef3f7')

    # --- Left panel: Multi-Ring Buffer ---
    ax1.set_xlim(-100, 1100)
    ax1.set_ylim(-100, 900)
    ax1.set_aspect('equal')
    ax1.set_xlabel('Easting (m)', fontsize=10)
    ax1.set_ylabel('Northing (m)', fontsize=10)
    ax1.set_title('Multi-Ring Buffer Analysis', fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.4)

    well_pts = np.array([(150,200),(400,150),(700,220),(200,550),(500,600),(800,500)])
    farm_fields = [
        [(50,720),(250,710),(270,800),(60,810)],
        [(300,700),(530,690),(550,790),(310,800)],
        [(600,730),(850,720),(870,820),(610,820)],
        [(50,380),(230,370),(250,500),(60,505)],
        [(900,350),(1050,340),(1060,480),(905,485)],
        [(620,350),(800,340),(820,450),(630,455)],
        [(300,300),(480,290),(500,400),(310,405)],
        [(50,50),(240,40),(260,160),(60,165)],
        [(650,50),(830,40),(850,170),(660,165)],
        [(370,50),(560,40),(580,160),(380,155)],
        [(830,600),(1000,590),(1010,710),(840,715)],
        [(880,130),(1040,120),(1050,250),(890,255)],
    ]

    # Draw farm fields
    for verts in farm_fields:
        poly = Polygon(verts, closed=True, facecolor='#c8e6c9',
                       edgecolor='#2e7d32', linewidth=1.5, alpha=0.7, zorder=2)
        ax1.add_patch(poly)

    # Draw buffers (largest first for visual layering)
    buffer_dists = [600, 300, 150]
    buffer_colors = ['#bbdefb', '#90caf9', '#42a5f5']
    buffer_alphas = [0.25, 0.35, 0.5]
    theta = np.linspace(0, 2*np.pi, 120)
    for bd, bc, ba in zip(buffer_dists, buffer_colors, buffer_alphas):
        for wx, wy in well_pts:
            xs = wx + bd * np.cos(theta)
            ys = wy + bd * np.sin(theta)
            ax1.fill(xs, ys, color=bc, alpha=ba, zorder=1)
            ax1.plot(xs, ys, color='#1565c0', linewidth=0.6, alpha=0.5, zorder=1)

    # Well points
    ax1.scatter(well_pts[:,0], well_pts[:,1], s=80, color='#0d47a1',
                zorder=5, marker='o', edgecolors='white', linewidths=1.2)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor='#42a5f5', alpha=0.9, label='150 m buffer'),
        mpatches.Patch(facecolor='#90caf9', alpha=0.9, label='300 m buffer'),
        mpatches.Patch(facecolor='#bbdefb', alpha=0.9, label='600 m buffer'),
        plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#0d47a1',
                   markersize=9, label='Well / Water point'),
        mpatches.Patch(facecolor='#c8e6c9', edgecolor='#2e7d32', label='Farm field'),
    ]
    ax1.legend(handles=legend_handles, loc='upper right', fontsize=9, framealpha=0.9)

    # --- Right panel: Nearest Water Source ---
    ax2.set_xlim(-100, 1100)
    ax2.set_ylim(-100, 900)
    ax2.set_aspect('equal')
    ax2.set_xlabel('Easting (m)', fontsize=10)
    ax2.set_ylabel('Northing (m)', fontsize=10)
    ax2.set_title('Nearest Water Source Distance', fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.4)

    # Field centroids from farm_fields
    centroids = np.array([(np.mean([v[0] for v in f]),
                           np.mean([v[1] for v in f])) for f in farm_fields])
    water_pts = np.array([(150,200),(450,170),(750,220),(200,550),
                           (550,600),(850,510)])

    # Compute nearest water source and distance
    dists = []
    nearest = []
    for cx, cy in centroids:
        d = np.sqrt((water_pts[:,0]-cx)**2 + (water_pts[:,1]-cy)**2)
        idx = np.argmin(d)
        dists.append(d[idx])
        nearest.append(water_pts[idx])
    dists = np.array(dists)

    # Color lines by distance
    norm = mcolors.Normalize(vmin=dists.min(), vmax=dists.max())
    cmap = cm.get_cmap('plasma')
    for i, (cx, cy) in enumerate(centroids):
        wx, wy = nearest[i]
        color = cmap(norm(dists[i]))
        ax2.plot([cx, wx], [cy, wy], color=color, linewidth=2, alpha=0.8, zorder=3)

    # Draw farm fields lightly
    for verts in farm_fields:
        poly = Polygon(verts, closed=True, facecolor='#f3e5f5',
                       edgecolor='#7b1fa2', linewidth=1.2, alpha=0.6, zorder=2)
        ax2.add_patch(poly)

    # Parcel centroids
    ax2.scatter(centroids[:,0], centroids[:,1], s=60, color='#7b1fa2',
                zorder=5, edgecolors='white', linewidths=1)
    # Water sources as stars
    ax2.scatter(water_pts[:,0], water_pts[:,1], s=140, color='#0d47a1',
                zorder=5, marker='*', edgecolors='white', linewidths=0.8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax2, shrink=0.75, pad=0.02)
    cbar.set_label('Distance (m)', fontsize=10)

    legend2 = [
        plt.Line2D([0],[0], marker='*', color='w', markerfacecolor='#0d47a1',
                   markersize=12, label='Water source'),
        plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#7b1fa2',
                   markersize=8, label='Field centroid'),
        plt.Line2D([0],[0], color='gray', linewidth=2, label='Nearest-source link'),
    ]
    ax2.legend(handles=legend2, loc='upper right', fontsize=9, framealpha=0.9)

    fig.suptitle('Buffer Analysis & Distance Calculations', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('images/buffer_and_distance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  buffer_and_distance.png  ✓')


# ============================================================
# 4.  dissolve_aggregation.png
# ============================================================
def make_dissolve_aggregation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor('#f8f9fa')
    for ax in (ax1, ax2):
        ax.set_facecolor('#eef3f7')

    crop_types = ['corn','corn','wheat','soy','corn','wheat','soy','sunflower',
                  'corn','soy','wheat','sunflower']
    crop_colors = {'corn':'gold','wheat':'#d4a017','soy':'#66bb6a',
                   'sunflower':'#ff7043'}

    # 12 parcel polygons (4 columns × 3 rows)
    parcels = []
    areas = []
    for row in range(3):
        for col in range(4):
            x0 = col * 3.0 + 0.2
            y0 = row * 2.5 + 0.2
            # Add slight irregularity
            jitter = rng.uniform(-0.15, 0.15, (4, 2))
            verts = np.array([(x0,y0),(x0+2.6,y0),(x0+2.6,y0+2.1),(x0,y0+2.1)]) + jitter
            parcels.append(verts)
            areas.append(round(rng.uniform(12, 28), 1))

    # Left: individual parcels
    ax1.set_xlim(-0.3, 12.5)
    ax1.set_ylim(-0.3, 8.0)
    ax1.set_aspect('equal')
    ax1.set_xlabel('Easting (km)', fontsize=10)
    ax1.set_ylabel('Northing (km)', fontsize=10)
    ax1.set_title('Individual Parcels (12)', fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.4)

    for i, (verts, crop) in enumerate(zip(parcels, crop_types)):
        poly = Polygon(verts, closed=True, facecolor=crop_colors[crop],
                       edgecolor='#333', linewidth=1.2, alpha=0.8, zorder=2)
        ax1.add_patch(poly)
        cx = verts[:,0].mean()
        cy = verts[:,1].mean()
        ax1.text(cx, cy, f'P{i+1}', ha='center', va='center',
                 fontsize=7, fontweight='bold', color='#111')

    legend_patches1 = [mpatches.Patch(facecolor=v, edgecolor='#333', label=k.capitalize())
                       for k, v in crop_colors.items()]
    ax1.legend(handles=legend_patches1, title='Crop Type', loc='upper right',
               fontsize=9, title_fontsize=9, framealpha=0.9)

    # Right: dissolved by crop type (merged large polygons)
    ax2.set_xlim(-0.3, 12.5)
    ax2.set_ylim(-0.3, 8.0)
    ax2.set_aspect('equal')
    ax2.set_xlabel('Easting (km)', fontsize=10)
    ax2.set_ylabel('Northing (km)', fontsize=10)
    ax2.set_title('Dissolved by Crop Type', fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.4)

    # Build dissolved bounding hulls per crop
    from collections import defaultdict
    crop_verts = defaultdict(list)
    crop_area = defaultdict(float)
    for verts, crop, area in zip(parcels, crop_types, areas):
        crop_verts[crop].extend(verts.tolist())
        crop_area[crop] += area

    # Convex hull approximation per crop
    for crop, pts in crop_verts.items():
        pts = np.array(pts)
        # Compute convex hull via angle sort around centroid
        cx, cy = pts.mean(axis=0)
        angles = np.arctan2(pts[:,1]-cy, pts[:,0]-cx)
        hull_pts = pts[np.argsort(angles)]
        # Expand slightly
        scale = 1.08
        hull_pts = (hull_pts - [cx, cy]) * scale + [cx, cy]
        poly = Polygon(hull_pts, closed=True, facecolor=crop_colors[crop],
                       edgecolor='#333', linewidth=2, alpha=0.75, zorder=2)
        ax2.add_patch(poly)
        total_area = crop_area[crop]
        count = crop_types.count(crop)
        ax2.text(cx, cy,
                 f'{crop.capitalize()}\n{count} parcels\n{total_area:.0f} ha total',
                 ha='center', va='center', fontsize=9, fontweight='bold',
                 color='#111', zorder=5,
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.75))

    legend_patches2 = [mpatches.Patch(facecolor=v, edgecolor='#333', label=k.capitalize())
                       for k, v in crop_colors.items()]
    ax2.legend(handles=legend_patches2, title='Crop Type', loc='upper right',
               fontsize=9, title_fontsize=9, framealpha=0.9)

    fig.suptitle('Dissolve + Aggregate by Crop Type', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('images/dissolve_aggregation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  dissolve_aggregation.png  ✓')


# ── Run all ──────────────────────────────────────────────────────────────────
print('Generating images for 05_spatial_analysis_geopandas …')
make_spatial_join()
make_overlay_operations()
make_buffer_and_distance()
make_dissolve_aggregation()
print('Done.')
