import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


bt_105 = xr.open_dataarray("bt_105_zoom_toulouse_13h30_radius_50.nc").values
bt_123 = xr.open_dataarray("bt_123_zoom_toulouse_13h30_radius_50.nc").values
bt_87 = xr.open_dataarray("bt_87_zoom_toulouse_13h30_radius_50.nc").values


# Rotation + flip pour annuler effet miroir
bt_105_corr = np.fliplr(np.rot90(bt_105, k=2))
bt_123_corr = np.rot90(bt_123, k=2)
bt_87_corr = np.rot90(bt_87, k=2)


# Utilisez des percentiles au lieu de valeurs fixes pour normaliser
R_diff = bt_123_corr - bt_105_corr
R = (R_diff - np.percentile(R_diff, 2)) / (np.percentile(R_diff, 98) - np.percentile(R_diff, 2))
R = np.clip(R, 0, 1)

G_diff = bt_105_corr - bt_87_corr
G = (G_diff - np.percentile(G_diff, 2)) / (np.percentile(G_diff, 98) - np.percentile(G_diff, 2))
G = np.clip(G, 0, 1)

B = bt_105_corr
B = (B - np.percentile(B, 2)) / (np.percentile(B, 98) - np.percentile(B, 2))
B = np.clip(B, 0, 1)
B = 1 - B  # inversion (froid = clair)


# combiner
rgb = np.stack([R, G, B], axis=-1)


# Ajouter le point rouge sur Toulouse
x_center = rgb.shape[1] // 2  # Centre horizontal
y_center = rgb.shape[0] // 2  # Centre vertical
plt.figure(figsize=(12, 10))
plt.imshow(rgb)
plt.scatter(x_center, y_center, color="red", s=100, marker='.', 
            edgecolors='white', linewidths=2, label="Toulouse", zorder=10)


# Graphique
plt.title("Composite_RGB_13h30_radius_50")
plt.axis('off')
plt.savefig('Composite_RGB_13h30_radius_50.png') 
plt.show()

