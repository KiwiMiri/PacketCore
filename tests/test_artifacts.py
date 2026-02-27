"""Tests for artifact save/load utilities."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from packetcore.utils.artifacts import load_artifact, save_artifact


@pytest.fixture()
def tmp_dir(tmp_path):
    return tmp_path


def test_save_load_json(tmp_dir):
    data = {"key": "value", "num": 42, "list": [1, 2, 3]}
    path = tmp_dir / "data.json"
    save_artifact(data, path)
    loaded = load_artifact(path)
    assert loaded == data


def test_save_load_pkl(tmp_dir):
    data = {"array": np.arange(10), "label": "test"}
    path = tmp_dir / "data.pkl"
    save_artifact(data, path)
    loaded = load_artifact(path)
    np.testing.assert_array_equal(loaded["array"], data["array"])


def test_save_load_pt(tmp_dir):
    tensor = {"weights": torch.randn(4, 4)}
    path = tmp_dir / "model.pt"
    save_artifact(tensor, path)
    loaded = load_artifact(path, map_location="cpu")
    torch.testing.assert_close(loaded["weights"], tensor["weights"])


def test_overwrite_false_raises(tmp_dir):
    path = tmp_dir / "data.json"
    save_artifact({"a": 1}, path)
    with pytest.raises(FileExistsError):
        save_artifact({"b": 2}, path, overwrite=False)


def test_overwrite_true_replaces(tmp_dir):
    path = tmp_dir / "data.json"
    save_artifact({"a": 1}, path)
    save_artifact({"a": 99}, path, overwrite=True)
    loaded = load_artifact(path)
    assert loaded["a"] == 99


def test_load_nonexistent_raises(tmp_dir):
    with pytest.raises(FileNotFoundError):
        load_artifact(tmp_dir / "missing.pkl")


def test_creates_parent_dirs(tmp_dir):
    path = tmp_dir / "nested" / "deep" / "data.json"
    save_artifact({"x": 1}, path)
    assert path.exists()


def test_returns_path_object(tmp_dir):
    path = tmp_dir / "data.json"
    result = save_artifact({}, path)
    assert isinstance(result, Path)
