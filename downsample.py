import numpy as np
import torch
import torch.nn as nn
from functions.z_order import xyz2key

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load data
data = np.load("test_data/Area_1_conferenceRoom_2.npy")
print(f"Original shape: {data.shape}")

coord = data[:, :3].copy()

# Voxel downsample with z-order sort
voxel_size = 0.02
coord_tensor = torch.from_numpy(coord).to(device).float()
grid_coord = torch.floor(coord_tensor / voxel_size).long()

key = xyz2key(grid_coord[:, 0], grid_coord[:, 1], grid_coord[:, 2])
keys_unique, inverse, counts = torch.unique(key, return_inverse=True, return_counts=True)
index = torch.argsort(key)
grid_coord = grid_coord[index][nn.functional.pad(torch.cumsum(counts, dim=0), (1, 0))[:len(keys_unique)]]
coord_tensor = coord_tensor[index][nn.functional.pad(torch.cumsum(counts, dim=0), (1, 0))[:len(keys_unique)]]

print(f"After voxelization: {len(coord_tensor)} points")

coord_down = coord_tensor.cpu().numpy()

# Save back to original file
np.save("test_data/Area_1_conferenceRoom_2.npy", coord_down)
print("Saved downsampled point cloud back to test_data/Area_1_conferenceRoom_2.npy")
