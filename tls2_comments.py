# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 11:29:54 2025

@author: aoyin
"""

import xml.etree.ElementTree as ET  # Bibliothèque pour lire les fichiers XML
import xarray as xr  # Bibliothèque pour manipuler les fichiers NetCDF
import os  # Pour gérer les chemins de fichiers
import numpy as np  # Calculs numériques
import matplotlib.pyplot as plt  # Créer des graphiques
import cartopy.crs as ccrs  # Projections cartographiques
import cartopy.feature as cfeature  # Éléments géographiques (frontières, côtes...)
from pyproj import Proj, transform  # Conversions de projections géographiques

#%% ÉTAPE 1 : RÉCUPÉRER LES FICHIERS
# Cette section lit le fichier manifest.xml pour obtenir la liste de tous les fichiers .nc

dossier = r"C:\Users\aoyin\OneDrive - IPSA\IPSA\5e_annee\Semestre1\PMI-Contrails\Python\MTG_FCI_guide_louise\13h"

# Parser le fichier XML manifest
tree = ET.parse(os.path.join(dossier, "manifest.xml"))
root = tree.getroot()  # Obtenir la racine du document XML
ns = {'eum': 'http://www.eumetsat.int/sip'}  # Namespace utilisé par EUMETSAT dans leurs XML

# Extraire tous les chemins de fichiers .nc depuis le manifest
# findall cherche tous les éléments 'path' dans le XML
# On ne garde que ceux qui se terminent par .nc
paths = [elem.text for elem in root.findall('.//eum:path', ns) if elem.text.endswith('.nc')]
print("Nombre de fichiers trouvés :", len(paths))

#%% ÉTAPE 2 : TEST SUR LE PREMIER FICHIER
# On teste d'abord avec un seul fichier pour vérifier que tout fonctionne

fichier_test = os.path.join(dossier, paths[0])  # Chemin complet du premier fichier
ds = xr.open_dataset(fichier_test)  # Ouvrir le fichier NetCDF

# Ouvrir le groupe spécifique qui contient les données IR10.5 mesurées
# Les fichiers MTG sont organisés en groupes hiérarchiques
ds_ir105 = xr.open_dataset(fichier_test, group="data/ir_105_hr/measured")

# Récupérer toutes les variables nécessaires pour le calcul de température de brillance
eff = ds_ir105["effective_radiance"]  # Radiance effective (valeur brute du capteur)
k_rad = ds_ir105["radiance_unit_conversion_coefficient"]  # Coefficient de conversion
wn  = ds_ir105["radiance_to_bt_conversion_coefficient_wavenumber"]  # Nombre d'onde central
c1  = ds_ir105["radiance_to_bt_conversion_constant_c1"]  # Constante de Planck c1
c2  = ds_ir105["radiance_to_bt_conversion_constant_c2"]  # Constante de Planck c2
a   = ds_ir105["radiance_to_bt_conversion_coefficient_a"]  # Coefficient de correction linéaire a
b   = ds_ir105["radiance_to_bt_conversion_coefficient_b"]  # Coefficient de correction linéaire b

# CONVERSION : Radiance effective → Radiance physique
# La radiance effective est une valeur normalisée qu'il faut multiplier par k_rad
L = eff * k_rad

# CALCUL DE LA TEMPÉRATURE DE BRILLANCE (Brightness Temperature)
# Étape 1 : Inverser la loi de Planck pour obtenir la température
# Formule : T = (c2 * wn) / ln(1 + (c1 * wn³) / L)
BT_planck = c2 * wn / np.log(1 + (c1 * wn**3) / L)

# Étape 2 : Appliquer la correction linéaire
# La correction linéaire ajuste la température pour tenir compte des non-linéarités du capteur
# Formule : BT_finale = a * BT_planck + b
BT_105 = a * BT_planck + b
BT_105.name = "BT_105"  # Donner un nom à la variable
print("BT 10.5 µm (K) :", float(BT_105.min()), "à", float(BT_105.max()))

#%% ÉTAPE 3 : TRAITER TOUS LES FICHIERS
# Les données MTG sont divisées en plusieurs fichiers (chunks)
# Il faut les traiter tous et les combiner ensuite

bt_105_list = []  # Liste pour stocker toutes les températures de brillance

for p in paths:  # Boucle sur tous les fichiers .nc
    fichier = os.path.join(dossier, p)
    ds_ir105_all = xr.open_dataset(fichier, group="data/ir_105_hr/measured")
    
    # Vérifier si le fichier contient des données mesurées
    # Certains fichiers contiennent seulement des tables de calibration (LUT)
    if "effective_radiance" not in ds_ir105_all:
        print(f"Fichier {p} sans 'effective_radiance', on le saute")
        ds_ir105_all.close()
        continue  # Passer au fichier suivant

    # Récupérer les mêmes variables que pour le test
    eff_all = ds_ir105_all["effective_radiance"]
    k_rad_all = ds_ir105_all["radiance_unit_conversion_coefficient"]
    wn_all  = ds_ir105_all["radiance_to_bt_conversion_coefficient_wavenumber"]
    c1_all  = ds_ir105_all["radiance_to_bt_conversion_constant_c1"]
    c2_all  = ds_ir105_all["radiance_to_bt_conversion_constant_c2"]
    a_all   = ds_ir105_all["radiance_to_bt_conversion_coefficient_a"]
    b_all   = ds_ir105_all["radiance_to_bt_conversion_coefficient_b"]

    # Même calcul que précédemment
    L_all = eff_all * k_rad_all
    BT_planck_all = c2_all * wn_all / np.log(1 + (c1_all * wn_all**3) / L_all)
    BT_105_all = a_all * BT_planck_all + b_all
    BT_105_all.name = "BT_105"
    
    bt_105_list.append(BT_105_all)  # Ajouter à la liste
    ds_ir105_all.close()  # Fermer le fichier pour libérer la mémoire

print(f"Nombre total de fichiers traités : {len(bt_105_list)}")

#%% ÉTAPE 4 : COMBINER TOUS LES FICHIERS
# Les fichiers représentent différentes parties de l'image complète
# Il faut les assembler en une seule grande image

# Convertir chaque DataArray en Dataset (requis par combine_by_coords)
bt_105_ds_list = [da.to_dataset() for da in bt_105_list]

# Combiner tous les chunks en utilisant leurs coordonnées x,y
# xarray comprend automatiquement comment les assembler
bt_105_full_ds = xr.combine_by_coords(bt_105_ds_list)

# Récupérer la DataArray combinée
bt_105_full = bt_105_full_ds["BT_105"]

# IMPORTANT : Inverser l'ordre des x pour corriger l'orientation
# En projection géostationnaire, x augmente vers l'ouest
# Mais matplotlib affiche x vers l'est → il faut inverser
#bt_105_full = bt_105_full.sortby('x', ascending=False)
#bt_105_full = bt_105_full.sortby('y', ascending=False)  # ← AJOUT


print(bt_105_full)

#%% ÉTAPE 5 : AFFICHAGE DE L'IMAGE COMPLÈTE

plt.figure(figsize=(10,8))
bt_105_full.plot.imshow(cmap='RdYlBu_r')  # Colormap : rouge=chaud, bleu=froid
plt.gca().invert_xaxis()  # Inverser l'axe x pour avoir la bonne orientation
plt.xlabel("")  
plt.ylabel("")
plt.title("BT 10.5 µm (K) - Image complète MTG FCI")
plt.show()

#%% ÉTAPE 6 : CALCUL DES COORDONNÉES GÉOGRAPHIQUES
# Les coordonnées x,y sont en radians dans la projection géostationnaire
# Il faut les convertir en latitude/longitude

print("\n🌍 Calcul des coordonnées géographiques depuis la projection MTG...")

# Récupérer les coordonnées x et y (en radians)
x_coords = bt_105_full.x.values  # Array 1D des coordonnées x
y_coords = bt_105_full.y.values  # Array 1D des coordonnées y

# Créer une grille 2D à partir des coordonnées 1D
# meshgrid crée deux matrices 2D où chaque point (i,j) contient ses coordonnées (x,y)
X, Y = np.meshgrid(x_coords, y_coords)

# PARAMÈTRES DU SATELLITE MTG
satellite_height = 35786023.0  # Altitude du satellite en mètres (~35786 km)
satellite_longitude = 0.0  # Le satellite MTG-I1 est positionné à 0°E

# CONVERSION : Radians → Mètres
# Les coordonnées x,y sont des angles (radians)
# Pour la projection, on a besoin de distances (mètres)
# Formule : distance = angle × rayon
X_m = X * satellite_height
Y_m = Y * satellite_height

# DÉFINIR LES PROJECTIONS
# proj_geos : projection géostationnaire (système du satellite)
# +proj=geos : type de projection (geostationary)
# +lon_0 : longitude du satellite
# +h : altitude du satellite
# +sweep=y : direction du balayage du capteur
proj_geos = Proj(f'+proj=geos +lon_0={satellite_longitude} +h={satellite_height} +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs +sweep=y')

# proj_latlon : projection géographique classique (latitude/longitude)
proj_latlon = Proj('+proj=latlong +ellps=WGS84')

# CONVERSION : Projection géostationnaire → Lat/Lon
print("⏳ Conversion en cours (peut prendre quelques secondes)...")
# transform convertit les coordonnées d'un système à l'autre
# flatten() : on transforme les matrices 2D en vecteurs 1D pour la conversion
lon, lat = transform(proj_geos, proj_latlon, X_m.flatten(), Y_m.flatten())

# RESHAPE : Revenir à une grille 2D
lon = lon.reshape(X.shape)
lat = lat.reshape(X.shape)

print(f"✅ Coordonnées calculées !")
print(f"   Latitude : [{np.nanmin(lat):.2f}°, {np.nanmax(lat):.2f}°]")
print(f"   Longitude : [{np.nanmin(lon):.2f}°, {np.nanmax(lon):.2f}°]")


# ✅ MAINTENANT INVERSER bt_105_full POUR QU'IL CORRESPONDE AUX LAT/LON
bt_105_full = bt_105_full.sortby('x', ascending=False)
bt_105_full = bt_105_full.sortby('y', ascending=False)


#%% ÉTAPE 7 : ZOOM SUR TOULOUSE
# Maintenant qu'on a les coordonnées géographiques, on peut localiser Toulouse

lat_tlse = 43.6045  # Latitude de Toulouse
lon_tlse = 1.4440   # Longitude de Toulouse

print(f"\n📍 Recherche de Toulouse ({lat_tlse}°N, {lon_tlse}°E)...")

# TROUVER LE PIXEL LE PLUS PROCHE DE TOULOUSE
# Calculer la distance euclidienne entre chaque pixel et Toulouse
# Formule : distance = √((lat - lat_tlse)² + (lon - lon_tlse)²)
dist = np.sqrt((lat - lat_tlse)**2 + (lon - lon_tlse)**2)

# Trouver l'indice du pixel avec la distance minimale
# argmin donne l'indice dans le tableau aplati
# unravel_index convertit cet indice en coordonnées 2D (iy, ix)
iy, ix = np.unravel_index(np.argmin(dist), dist.shape)

print(f"   Pixel trouvé → ix={ix}, iy={iy}")
print(f"   Coordonnées réelles : lat={lat[iy, ix]:.4f}°, lon={lon[iy, ix]:.4f}°")

# DÉFINIR UNE FENÊTRE AUTOUR DE TOULOUSE
radius = 50  # Nombre de pixels de part et d'autre de Toulouse

# Calculer les limites du zoom en s'assurant de rester dans l'image
x_min = max(ix - radius, 0)  # Ne pas descendre sous 0
x_max = min(ix + radius, bt_105_full.sizes["x"])  # Ne pas dépasser la taille de l'image
y_min = max(iy - radius, 0)
y_max = min(iy + radius, bt_105_full.sizes["y"])

# EXTRAIRE LES DONNÉES DU ZOOM
# isel = index selection (sélection par indices)
# slice = crée une tranche d'indices (comme [x_min:x_max])
bt_zoom = bt_105_full.isel(x=slice(x_min, x_max), y=slice(y_min, y_max))
lat_zoom = lat[y_min:y_max, x_min:x_max]  # Extraire les lat/lon correspondantes
lon_zoom = lon[y_min:y_max, x_min:x_max]

print(f"   Taille du zoom : {bt_zoom.shape}")

#%% ÉTAPE 8 : AFFICHAGE SIMPLE DU ZOOM

plt.figure(figsize=(10, 10))

# pcolormesh : affiche une grille 2D avec des couleurs
# shading="nearest" : chaque pixel garde sa couleur (pas d'interpolation)
#plt.pcolormesh(lon_zoom, lat_zoom, bt_zoom, shading="nearest", cmap='RdYlBu_r')
plt.pcolormesh(lon_zoom, lat_zoom, bt_zoom, shading="nearest")


# Ajouter un point rouge pour Toulouse
plt.scatter(lon_tlse, lat_tlse, color="red", s=100, marker='*', 
            edgecolors='white', linewidths=2, label="Toulouse", zorder=10)

plt.colorbar(label="BT 10.5 µm (K)")
plt.xlabel("Longitude (°)")
plt.ylabel("Latitude (°)")
plt.title("Zoom radiances IR10.5 – Région de Toulouse")
plt.legend()
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()

#%% ÉTAPE 9 : CARTE CARTOPY PROFESSIONNELLE

print("\n🗺️ Création de la carte géographique avec Cartopy...")

fig = plt.figure(figsize=(14, 12))


# Créer des axes avec une projection cartographique
# PlateCarree = projection équirectangulaire (lat/lon directement)
ax = plt.axes(projection=ccrs.PlateCarree())

# ✅ AJOUTER LE TITRE ICI (avant ou après la création des axes)
plt.title(f"Zoom de {radius} km autour de Toulouse – Radiances IR10.5", 
          fontsize=14, weight='bold', pad=20)

# DÉFINIR L'ÉTENDUE DE LA CARTE
# extent = [lon_min, lon_max, lat_min, lat_max]
extent = [float(np.nanmin(lon_zoom)), float(np.nanmax(lon_zoom)), 
          float(np.nanmin(lat_zoom)), float(np.nanmax(lat_zoom))]
ax.set_extent(extent, crs=ccrs.PlateCarree())

# AJOUTER LES DONNÉES DE RADIANCE
# vmin/vmax : limiter les valeurs pour améliorer le contraste
# percentile(2) et percentile(98) : ignorer les valeurs extrêmes
im = ax.pcolormesh(lon_zoom, lat_zoom, bt_zoom, 
                   transform=ccrs.PlateCarree(),  # Indiquer la projection des données
                   #cmap='RdYlBu_r',  # Colormap
                   alpha=0.7,  # Transparence pour voir la carte dessous
                   shading='nearest',
                   vmin=np.nanpercentile(bt_zoom, 2),
                   vmax=np.nanpercentile(bt_zoom, 98))

# AJOUTER LES ÉLÉMENTS CARTOGRAPHIQUES
ax.add_feature(cfeature.BORDERS, linewidth=1.5, edgecolor='black')  # Frontières
ax.add_feature(cfeature.COASTLINE, linewidth=1.5)  # Côtes
ax.add_feature(cfeature.RIVERS, edgecolor='blue', alpha=0.5)  # Rivières
ax.add_feature(cfeature.LAKES, edgecolor='blue', facecolor='lightblue', alpha=0.5)  # Lacs

# MARQUER TOULOUSE
ax.plot(lon_tlse, lat_tlse, 'r*', markersize=20, 
        transform=ccrs.PlateCarree(), 
        markeredgecolor='white', markeredgewidth=2, zorder=10)
ax.text(lon_tlse + 0.1, lat_tlse + 0.1, 'Toulouse', 
        transform=ccrs.PlateCarree(),
        fontsize=12, weight='bold', color='red',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# AJOUTER D'AUTRES VILLES
villes = {
    'Montpellier': (3.8767, 43.6108),
    'Bordeaux': (-0.5792, 44.8378),
    'Pau': (-0.3708, 43.2951),
    'Carcassonne': (2.3488, 43.2130),
    'Albi': (2.1479, 43.9298)
}

# Ne garder que les villes dans la zone de zoom
for ville, (lon_v, lat_v) in villes.items():
    if extent[0] <= lon_v <= extent[1] and extent[2] <= lat_v <= extent[3]:
        ax.plot(lon_v, lat_v, 'ko', markersize=5, transform=ccrs.PlateCarree(), zorder=5)
        ax.text(lon_v + 0.05, lat_v + 0.05, ville, 
               transform=ccrs.PlateCarree(),
               fontsize=9, style='italic',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))

# AJOUTER UNE GRILLE LAT/LON
gl = ax.gridlines(draw_labels=True, linewidth=0.5, 
                  color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False  # Pas de labels en haut
gl.right_labels = False  # Pas de labels à droite
gl.xlabel_style = {'size': 10}

