from typing import Callable

import numpy as np
import pytest
from torch.utils.data import DataLoader, Dataset, Subset

from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import RenderContext, StubRasterizer, build_rasterizer


def check_sample(cfg: dict, dataset: Dataset) -> None:
    iterator = iter(dataset)  # type: ignore
    for i in range(10):
        el = next(iterator)
        assert el["image"].shape[1:] == tuple(cfg["raster_params"]["raster_size"])
        assert len(el["target_positions"]) == cfg["model_params"]["future_num_frames"]
        assert len(el["target_yaws"]) == cfg["model_params"]["future_num_frames"]
        assert len(el["target_availabilities"]) == cfg["model_params"]["future_num_frames"]
        assert el["world_to_image"].shape == (3, 3)


def check_torch_loading(dataset: Dataset) -> None:
    # test dataloader
    iterator = iter(DataLoader(dataset, batch_size=4))  # type: ignore
    next(iterator)


@pytest.mark.parametrize("rast_name", ["py_satellite", "py_semantic", "box_debug", "satellite_debug"])
@pytest.mark.parametrize("dataset_cls", [EgoDataset, AgentDataset])
def test_dataset_rasterizer(
    rast_name: str, dataset_cls: Callable, zarr_dataset: ChunkedDataset, dmg: LocalDataManager, cfg: dict
) -> None:
    rasterizer = build_rasterizer(cfg, dmg)

    dataset = dataset_cls(cfg=cfg, zarr_dataset=zarr_dataset, rasterizer=rasterizer, perturbation=None)
    check_sample(cfg, dataset)
    check_torch_loading(dataset)


@pytest.mark.parametrize("frame_idx", [0, 10, 247, pytest.param(775, marks=pytest.mark.xfail)])
@pytest.mark.parametrize("dataset_cls", [EgoDataset, AgentDataset])
def test_frame_index_interval(dataset_cls: Callable, frame_idx: int, zarr_dataset: ChunkedDataset, cfg: dict) -> None:
    render_context = RenderContext(np.asarray((100, 100)), np.asarray((0.25, 0.25)), np.asarray((0.5, 0.5)))
    rasterizer = StubRasterizer(render_context, 0)
    dataset = dataset_cls(cfg, zarr_dataset, rasterizer, None)
    indices = dataset.get_frame_indices(frame_idx)
    subdata = Subset(dataset, indices)
    for _ in subdata:  # type: ignore
        pass


@pytest.mark.parametrize("scene_idx", [0, pytest.param(1, marks=pytest.mark.xfail)])
@pytest.mark.parametrize("dataset_cls", [EgoDataset, AgentDataset])
def test_scene_index_interval(dataset_cls: Callable, scene_idx: int, zarr_dataset: ChunkedDataset, cfg: dict) -> None:
    render_context = RenderContext(np.asarray((100, 100)), np.asarray((0.25, 0.25)), np.asarray((0.5, 0.5)))
    rasterizer = StubRasterizer(render_context, 0)
    dataset = dataset_cls(cfg, zarr_dataset, rasterizer, None)
    indices = dataset.get_scene_indices(scene_idx)
    subdata = Subset(dataset, indices)
    for _ in subdata:  # type: ignore
        pass


@pytest.mark.parametrize("history_num_frames", [1, 2, 3, 4])
@pytest.mark.parametrize("dataset_cls", [EgoDataset, AgentDataset])
def test_non_zero_history(
    history_num_frames: int, dataset_cls: Callable, zarr_dataset: ChunkedDataset, dmg: LocalDataManager, cfg: dict
) -> None:
    cfg["model_params"]["history_num_frames"] = history_num_frames
    rast_params = cfg["raster_params"]
    rast_params["map_type"] = "box_debug"
    rasterizer = build_rasterizer(cfg, dmg)

    dataset = dataset_cls(cfg, zarr_dataset, rasterizer, None)
    indexes = [0, 1, 10, -1]  # because we pad, even the first index should have an (entire black) history
    for idx in indexes:
        data = dataset[idx]
        assert data["image"].shape == (2 * (history_num_frames + 1), *rast_params["raster_size"])


@pytest.mark.parametrize("history_num_frames", [1, 2, 3, 4])
@pytest.mark.parametrize("dataset_cls", [EgoDataset, AgentDataset])
def test_no_rast_dataset(
    history_num_frames: int, dataset_cls: Callable, zarr_dataset: ChunkedDataset, dmg: LocalDataManager, cfg: dict
) -> None:
    cfg["model_params"]["history_num_frames"] = history_num_frames
    rasterizer = None
    dataset = dataset_cls(cfg, zarr_dataset, rasterizer, None)
    indexes = [0, 1, 10, -1]  # because we pad, even the first index should have an (entire black) history
    for idx in indexes:
        data = dataset[idx]
        assert "image" not in data
    check_torch_loading(dataset)
