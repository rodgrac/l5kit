import matplotlib.pyplot as plt

import numpy as np

from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset, AgentDataset

from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR
from l5kit.geometry import transform_points
from tqdm import tqdm
from collections import Counter
from l5kit.data import PERCEPTION_LABELS
from prettytable import PrettyTable

import os

# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "/scratch/rodney/datasets/lyft_prediction"
# get config
cfg = load_config_data("./visualisation_config.yaml")
print(cfg)

print(f'current raster_param:\n')
for k,v in cfg["raster_params"].items():
    print(f"{k}:{v}")

dm = LocalDataManager()
dataset_path = dm.require(cfg["val_data_loader"]["key"])
zarr_dataset = ChunkedDataset(dataset_path)
zarr_dataset.open()
print(zarr_dataset)

# frames = zarr_dataset.frames
# coords = np.zeros((len(frames), 2))
# for idx_coord, idx_data in enumerate(tqdm(range(len(frames)), desc="getting centroid to plot trajectory")):
#     frame = zarr_dataset.frames[idx_data]
#     coords[idx_coord] = frame["ego_translation"][:2]
#
# plt.figure()
# plt.scatter(coords[:, 0], coords[:, 1], marker='.')
# axes = plt.gca()
# axes.set_xlim([-2500, 1600])
# axes.set_ylim([-2500, 1600])

agents = zarr_dataset.agents
probabilities = agents["label_probabilities"]
labels_indexes = np.argmax(probabilities, axis=1)
counts = []
for idx_label, label in enumerate(PERCEPTION_LABELS):
    counts.append(np.sum(labels_indexes == idx_label))

table = PrettyTable(field_names=["label", "counts"])
for count, label in zip(counts, PERCEPTION_LABELS):
    table.add_row([label, count])
print(table)

rast = build_rasterizer(cfg, dm)
dataset = EgoDataset(cfg, zarr_dataset, rast)


data = dataset[10000]

im = data["image"].transpose(1, 2, 0)
im = dataset.rasterizer.to_rgb(im)
target_positions_pixels = transform_points(data["target_positions"], data["raster_from_agent"])
draw_trajectory(im, target_positions_pixels, TARGET_POINTS_COLOR, yaws=data["target_yaws"])

plt.figure()
plt.imshow(im[::-1])


cfg["raster_params"]["map_type"] = "py_satellite"
rast = build_rasterizer(cfg, dm)
dataset = EgoDataset(cfg, zarr_dataset, rast)
data = dataset[50]

im = data["image"].transpose(1, 2, 0)
im = dataset.rasterizer.to_rgb(im)
target_positions_pixels = transform_points(data["target_positions"], data["raster_from_agent"])
draw_trajectory(im, target_positions_pixels, TARGET_POINTS_COLOR, yaws=data["target_yaws"])

plt.figure()
plt.imshow(im[::-1])

dataset = AgentDataset(cfg, zarr_dataset, rast)
data = dataset[0]

im = data["image"].transpose(1, 2, 0)
im = dataset.rasterizer.to_rgb(im)
target_positions_pixels = transform_points(data["target_positions"], data["raster_from_agent"])
draw_trajectory(im, target_positions_pixels, TARGET_POINTS_COLOR, yaws=data["target_yaws"])

plt.figure()
plt.imshow(im[::-1])

# from IPython.display import display, clear_output
# import PIL
#
# cfg["raster_params"]["map_type"] = "py_semantic"
# rast = build_rasterizer(cfg, dm)
# dataset = EgoDataset(cfg, zarr_dataset, rast)
# scene_idx = 1
# indexes = dataset.get_scene_indices(scene_idx)
# images = []
#
# for idx in indexes:
#     data = dataset[idx]
#     im = data["image"].transpose(1, 2, 0)
#     im = dataset.rasterizer.to_rgb(im)
#     target_positions_pixels = transform_points(data["target_positions"], data["raster_from_agent"])
#     center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]
#     draw_trajectory(im, target_positions_pixels, TARGET_POINTS_COLOR, yaws=data["target_yaws"])
#     clear_output(wait=True)
#     display(PIL.Image.fromarray(im[::-1]))

plt.show()