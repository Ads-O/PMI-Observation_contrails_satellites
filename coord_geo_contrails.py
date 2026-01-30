# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 19:28:53 2026

@author: aoyin
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.ndimage import binary_opening
import pandas as pd

# === PARAMÈTRES GÉOGRAPHIQUES ===
lat_center = 43.6045
lon_center = 1.4440
km_per_pixel = 2.0

# Seuils RGB pour contrails
R_thr, G_thr, B_thr = 0.6, 0.4, 0.6

times = ["13h10", "13h20", "13h30", "13h40"]
channels = [105, 123, 87]

all_coords = []  # Pour stocker toutes les coordonnées pour le plot combiné

def read_and_correct(time):
    """Lecture et correction orientation pour les trois canaux"""
    da_105 = xr.open_dataarray(f"bt_105_zoom_toulouse_{time}_radius_100.nc")
    da_123 = xr.open_dataarray(f"bt_123_zoom_toulouse_{time}_radius_100.nc")
    da_87  = xr.open_dataarray(f"bt_87_zoom_toulouse_{time}_radius_100.nc")

    bt_105 = np.fliplr(np.rot90(da_105.values, 2))
    bt_123 = np.rot90(da_123.values, 2)
    bt_87  = np.rot90(da_87.values, 2)

    n_rows, n_cols = bt_105.shape
    deg_lat_per_km = 1 / 111.0
    deg_lon_per_km = 1 / (111.0 * np.cos(np.deg2rad(lat_center)))
    lat_extent = (n_rows / 2) * km_per_pixel * deg_lat_per_km
    lon_extent = (n_cols / 2) * km_per_pixel * deg_lon_per_km

    lat_1d = np.linspace(lat_center + lat_extent, lat_center - lat_extent, n_rows)
    lon_1d = np.linspace(lon_center - lon_extent, lon_center + lon_extent, n_cols)
    lon, lat = np.meshgrid(lon_1d, lat_1d)

    lat_corr = np.fliplr(np.rot90(lat, 2))
    lon_corr = np.fliplr(np.rot90(lon, 2))

    return bt_105, bt_123, bt_87, lat_corr, lon_corr

def create_rgb(bt_105, bt_123, bt_87):
    """Création du composite RGB"""
    R_diff = bt_123 - bt_105
    R = np.clip((R_diff - np.percentile(R_diff, 2)) / (np.percentile(R_diff, 98)-np.percentile(R_diff, 2)), 0,1)

    G_diff = bt_105 - bt_87
    G = np.clip((G_diff - np.percentile(G_diff, 2)) / (np.percentile(G_diff, 98)-np.percentile(G_diff, 2)),0,1)

    B = bt_105
    B = np.clip((B - np.percentile(B, 2)) / (np.percentile(B, 98)-np.percentile(B, 2)),0,1)
    B = 1 - B

    return np.stack([R,G,B], axis=-1), R, G, B

def detect_contrails(R,G,B):
    """Détection des contrails en magenta"""
    mask = (R>R_thr) & (B>B_thr) & (G<G_thr)
    mask = binary_opening(mask, structure=np.ones((3,3)))
    return mask

def save_csv(time, lat_corr, lon_corr, R,G,B, mask):
    iy, ix = np.where(mask)
    df = pd.DataFrame({
        "latitude": lat_corr[iy, ix],
        "longitude": lon_corr[iy, ix],
        "R": R[iy, ix],
        "G": G[iy, ix],
        "B": B[iy, ix]
    })
    df.to_csv(f"contrails_{time}.csv", index=False)
    return df

def plot_rgb_contrails(time, rgb, mask):
    iy, ix = np.where(mask)
    plt.figure(figsize=(12,10))
    plt.imshow(rgb)
    plt.scatter(ix, iy, s=2, c='cyan', alpha=0.8)
    plt.scatter(rgb.shape[1]//2, rgb.shape[0]//2, s=200, c='red', marker='*', edgecolors='white', linewidths=3)
    plt.title(f"{time} - RGB + Contrails")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"RGB_contrails_{time}.png", dpi=300)
    plt.show()

# === BOUCLE PRINCIPALE ===
for t in times:
    bt_105, bt_123, bt_87, lat_corr, lon_corr = read_and_correct(t)
    rgb, R, G, B = create_rgb(bt_105, bt_123, bt_87)
    mask = detect_contrails(R,G,B)
    df = save_csv(t, lat_corr, lon_corr, R,G,B, mask)

    # Ajout des coordonnées pour le plot combiné
    all_coords.append(df[['latitude','longitude']])

    plot_rgb_contrails(t, rgb, mask)

# === PLOT COMBINÉ DE TOUS LES CONTRAILS ===
plt.figure(figsize=(10,10))
for i, df_coords in enumerate(all_coords):
    plt.scatter(df_coords['longitude'], df_coords['latitude'], s=5, alpha=0.5, label=times[i])
plt.scatter(lon_center, lat_center, s=200, c='red', marker='*', edgecolors='white', linewidths=3, label='Toulouse')
plt.xlabel("Longitude (°E)")
plt.ylabel("Latitude (°N)")
plt.title("Contrails détectés - Tous les créneaux horaires")
plt.legend()
plt.grid(True, alpha=0.4, linestyle='--')
plt.axis('equal')
plt.tight_layout()
plt.savefig("Contrails_tous_creneaux.png", dpi=300)
plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy import ndimage
import json

#%% [... Code de chargement et création RGB identique ...]

#%% DÉTECTION AUTOMATIQUE DES CONTRAILS

print("\n🔍 Pré-détection automatique des contrails...")

# Détection magenta
R_thr = 0.6
B_thr = 0.6
G_thr = 0.4

contrail_mask = (R > R_thr) & (B > B_thr) & (G < G_thr)
contrail_mask = ndimage.binary_opening(contrail_mask, structure=np.ones((3,3)))

# Labelliser les objets
labeled, num_features = ndimage.label(contrail_mask)

print(f"   ✅ {num_features} zones détectées automatiquement")

#%% CRÉER DES ANNOTATIONS LABELME AUTOMATIQUES

from skimage import measure

shapes = []

for i in range(1, num_features + 1):
    # Masque pour cet objet
    mask_i = (labeled == i)
    
    # Extraire les contours
    contours = measure.find_contours(mask_i, 0.5)
    
    if len(contours) == 0:
        continue
    
    # Prendre le plus grand contour
    contour = max(contours, key=len)
    
    # Convertir en format LabelMe (x, y au lieu de row, col)
    points = [[float(point[1]), float(point[0])] for point in contour]
    
    # Simplifier le contour (garder 1 point sur 5)
    points = points[::5]
    
    # Créer l'annotation
    shape = {
        "label": f"contrail_{i}",
        "points": points,
        "group_id": None,
        "shape_type": "polygon",
        "flags": {}
    }
    
    shapes.append(shape)

print(f"   ✅ {len(shapes)} annotations créées")

#%% SAUVEGARDER LE FICHIER LABELME AVEC ANNOTATIONS

labelme_data = {
    "version": "5.0.1",
    "flags": {},
    "shapes": shapes,
    "imagePath": "composite_rgb_toulouse_pour_labelme.png",
    "imageData": None,
    "imageHeight": rgb.shape[0],
    "imageWidth": rgb.shape[1]
}

with open('composite_rgb_toulouse_ANNOTE.json', 'w') as f:
    json.dump(labelme_data, f, indent=2)

print("   ✅ composite_rgb_toulouse_ANNOTE.json créé avec pré-annotations")

#%% VISUALISER LES ANNOTATIONS

fig, ax = plt.subplots(figsize=(14, 12))
ax.imshow(rgb)

# Dessiner les contours détectés
for shape in shapes:
    points = np.array(shape['points'])
    ax.plot(points[:, 0], points[:, 1], 'c-', linewidth=2, alpha=0.8)
    ax.fill(points[:, 0], points[:, 1], 'cyan', alpha=0.3)
    
    # Centre
    center_x = np.mean(points[:, 0])
    center_y = np.mean(points[:, 1])
    ax.plot(center_x, center_y, 'y*', markersize=15, 
           markeredgecolor='black', markeredgewidth=1)
    ax.text(center_x, center_y, shape['label'], 
           color='white', fontsize=9, weight='bold',
           ha='center', va='bottom',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

ax.set_title(f'Pré-annotations automatiques\n{len(shapes)} contrails détectées', 
            fontsize=14, weight='bold')
ax.axis('off')

plt.tight_layout()
plt.savefig('composite_rgb_PRE_ANNOTE.png', dpi=200, bbox_inches='tight')
plt.show()

print("   ✅ composite_rgb_PRE_ANNOTE.png créé")

print("\n📝 Tu peux maintenant :")
print("   1. Ouvrir dans LabelMe : labelme composite_rgb_toulouse_pour_labelme.png")
print("   2. Charger les pré-annotations : File → Load → composite_rgb_toulouse_ANNOTE.json")
print("   3. Corriger/Ajuster les annotations automatiques")
print("   4. Sauvegarder : File → Save")
