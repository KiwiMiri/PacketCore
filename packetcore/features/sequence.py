"""Sequence window builder.

Produces fixed-length sliding-window tensors of shape ``[T, F]`` from an
ordered sequence of tabular feature vectors.

Time-ordering
-------------
The caller is responsible for ensuring the input is sorted by timestamp
before calling :meth:`SequenceBuilder.build`.  This class does **not** sort
internally so that it can be composed cleanly with the time-based splitter.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class SequenceBuilder:
    """Convert a tabular feature matrix into overlapping window tensors.

    Parameters
    ----------
    window_size:
        Number of time-steps ``T`` in each window.
    stride:
        Step size between consecutive windows.  ``stride=1`` gives maximally
        overlapping windows; ``stride=window_size`` gives non-overlapping.
    pad_value:
        Scalar used to left-pad the first ``window_size - 1`` incomplete
        windows when ``pad_partial=True``.
    pad_partial:
        If ``True``, emit windows for the first ``window_size - 1`` rows by
        left-padding with *pad_value*.  If ``False`` (default), only emit
        full windows.
    """

    def __init__(
        self,
        window_size: int = 32,
        stride: int = 1,
        pad_value: float = 0.0,
        pad_partial: bool = False,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.window_size = window_size
        self.stride = stride
        self.pad_value = pad_value
        self.pad_partial = pad_partial

    def build(self, X: np.ndarray) -> np.ndarray:
        """Slide a window over *X* and return a 3-D tensor.

        Parameters
        ----------
        X:
            Feature matrix of shape ``(N, F)``.

        Returns
        -------
        windows:
            Tensor of shape ``(W, T, F)`` where ``W`` is the number of
            windows produced.
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")

        n, f = X.shape

        if self.pad_partial:
            padding = np.full((self.window_size - 1, f), self.pad_value, dtype=np.float32)
            X_padded = np.concatenate([padding, X], axis=0)
        else:
            X_padded = X

        n_padded = len(X_padded)
        if n_padded < self.window_size:
            # Not enough data for even one window → return empty tensor
            return np.empty((0, self.window_size, f), dtype=np.float32)

        # Number of windows
        n_windows = (n_padded - self.window_size) // self.stride + 1

        windows = np.empty((n_windows, self.window_size, f), dtype=np.float32)
        for i in range(n_windows):
            start = i * self.stride
            windows[i] = X_padded[start : start + self.window_size]

        return windows

    def build_single(self, X: np.ndarray, idx: int) -> np.ndarray:
        """Return the window *ending* at row *idx* (inclusive).

        Useful for per-sample scoring during inference where you want the
        window context for a specific packet.

        Parameters
        ----------
        X:
            Feature matrix ``(N, F)``.
        idx:
            Row index (0-based) of the last packet in the window.

        Returns
        -------
        window:
            Array of shape ``(T, F)``.
        """
        X = np.asarray(X, dtype=np.float32)
        start = max(0, idx - self.window_size + 1)
        window = X[start : idx + 1]
        if len(window) < self.window_size:
            pad = np.full((self.window_size - len(window), X.shape[1]), self.pad_value, dtype=np.float32)
            window = np.concatenate([pad, window], axis=0)
        return window
