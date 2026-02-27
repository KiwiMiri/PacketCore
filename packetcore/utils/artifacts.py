"""Artefact persistence utilities.

Provides a unified interface for saving and loading all pipeline artefacts:

* PyTorch model weights (``torch.nn.Module``)
* Scikit-learn models / scalers (anything pickle-serialisable)
* Plain Python objects (dictionaries, calibrators, threshold maps, etc.)

All artefacts are stored as versioned files so that multiple training runs
can coexist without overwriting each other.

File format
-----------
* ``*.pt``   – PyTorch ``state_dict`` (saved with :func:`torch.save`)
* ``*.pkl``  – Pickle for sklearn objects and generic Python objects
* ``*.json`` – JSON for lightweight metadata (schema, thresholds, config)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".pt", ".pkl", ".json"}


def save_artifact(obj: Any, path: Union[str, Path], overwrite: bool = True) -> Path:
    """Persist *obj* to *path*.

    The serialisation format is chosen automatically based on the file
    extension:

    * ``.pt``   → :func:`torch.save` (requires PyTorch)
    * ``.json`` → :func:`json.dump` (obj must be JSON-serialisable)
    * ``.pkl``  → :mod:`pickle`

    Parameters
    ----------
    obj:
        Object to persist.
    path:
        Destination file path.  The parent directory is created if needed.
    overwrite:
        If ``False`` and *path* exists, raise :exc:`FileExistsError`.

    Returns
    -------
    path:
        Resolved :class:`pathlib.Path` of the saved file.
    """
    path = Path(path)
    if not overwrite and path.exists():
        raise FileExistsError(f"Artefact already exists at '{path}'. Set overwrite=True to replace it.")

    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()

    if ext == ".pt":
        import torch
        torch.save(obj, path)
    elif ext == ".json":
        with path.open("w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2)
    else:
        # Default to pickle for .pkl or unknown extensions
        with path.open("wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Saved artefact to '%s'.", path)
    return path


def load_artifact(path: Union[str, Path], map_location: Optional[str] = None) -> Any:
    """Load an artefact previously saved by :func:`save_artifact`.

    Parameters
    ----------
    path:
        File path to load from.
    map_location:
        Forwarded to :func:`torch.load` for ``.pt`` files (e.g.
        ``"cpu"`` when loading GPU-trained models on a CPU-only machine).

    Returns
    -------
    obj:
        Deserialised object.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artefact not found at '{path}'.")

    ext = path.suffix.lower()

    if ext == ".pt":
        import torch
        obj = torch.load(path, map_location=map_location, weights_only=False)
    elif ext == ".json":
        with path.open("r", encoding="utf-8") as fh:
            obj = json.load(fh)
    else:
        with path.open("rb") as fh:
            obj = pickle.load(fh)  # noqa: S301

    logger.info("Loaded artefact from '%s'.", path)
    return obj
