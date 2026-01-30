from satpy import Scene
from satpy import find_files_and_readers
import os
from glob import glob
from pyresample import geometry

# --- 1. DÉFINITION DE LA ZONE (Toulouse) ---
# On définit une grille fixe. C'est le "cadre" de ta photo.
area_id = 'toulouse_zoom'
description = 'Zoom sur Toulouse et environs'
proj_id = 'latlon' # Projection simple Latitude/Longitude
projection = {'proj': 'latlong', 'datum': 'WGS84'}

# Taille de l'image en pixels (Largeur x Hauteur)
# 1000x1000 pixels pour un zoom de quelques degrés, c'est de la haute résolution (~500m/pixel)
width = 1000
height = 1000

# Les bornes géographiques [Lon Min, Lat Min, Lon Max, Lat Max]
min_lat, max_lat = 41.1, 47.1
min_lon, max_lon = -1.9, 3.9
area_extent = (min_lon, min_lat, max_lon, max_lat)

# Création de l'objet de définition de zone
area_def = geometry.AreaDefinition(
    area_id,
    description,
    proj_id,
    projection,
    width,
    height,
    area_extent
)

# --- 2. GESTION DES FICHIERS ---
# On adapte pour lire ton dossier 'data1' actuel
script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, "data") 
output_dir = os.path.join(script_dir, "Resultats_Satpy")

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# On cherche tous les fichiers NC
all_files = glob(os.path.join(input_dir, "*.nc"))
# On filtre pour ne garder que les fichiers EUMETSAT (pas les fichiers TRAIL)
files_to_load = [f for f in all_files if "EUMETSAT" in f and "TRAIL" not in f]

if not files_to_load:
    print("❌ Erreur : Aucun fichier trouvé dans 'data1'.")
    exit()

print(f"📂 Fichiers trouvés : {len(files_to_load)}")

# --- 3. CHARGEMENT ET TRAITEMENT ---
print("🛠️ Création de la Scène...")
scn = Scene(reader="fci_l1c_nc", filenames=files_to_load)

# On charge les canaux nécessaires pour 'ash' (ou 'dust')
# Note : 'ash' utilise IR8.7, IR10.8 (ou 10.5), IR12.0 (ou 12.3)
# Satpy va trouver les noms correspondants tout seul (ir_87, ir_105, ir_123)
print("⏳ Chargement du composite 'ash' (Cendres/Traînées)...")
scn.load(['ash']) 

# --- 4. RE-PROJECTION (L'étape clé de ton code) ---
print("🔄 Projection sur la zone Toulouse...")
# On force l'image à entrer dans notre cadre défini plus haut (area_def)
local_scn = scn.resample(area_def)

# --- 5. SAUVEGARDE ---
output_filename = os.path.join(output_dir, "satpy_ash_toulouse.png")
print(f"💾 Sauvegarde en cours : {output_filename} ...")

# save_dataset fait tout le travail (normalisation, couleurs, sauvegarde)
local_scn.save_dataset('ash', filename=output_filename)

print("✅ Terminé !")