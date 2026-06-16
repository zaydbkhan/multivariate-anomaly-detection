"""Shared utilities for TranAD training and inference."""

import numpy as np
import torch


def convert_to_windows(data: torch.Tensor, window_size: int) -> torch.Tensor:
    """Convert time series to sliding windows.

    For position i, window contains data[i-W:i] (padded with data[0] at front).
    Vectorized using torch.Tensor.unfold for efficiency.

    Reference: tranad/main.py, convert_to_windows().

    Args:
        data: (N, features) tensor
        window_size: window length (default 10)

    Returns:
        (N, window_size, features) tensor
    """
    n, f = data.shape
    pad = data[0:1].expand(window_size, f)
    padded = torch.cat([pad, data], dim=0)  # (N + W, F)
    windows = padded.unfold(0, window_size, 1)  # (N + 1, F, W)
    windows = windows[:n].permute(0, 2, 1)  # (N, W, F)
    return windows


def auto_device(preference: str = "auto") -> torch.device:
    """Select best available torch device.

    Args:
        preference: "auto", "cpu", "cuda", or "mps".

    Returns:
        torch.device
    """
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def subsample_data(
    data: np.ndarray, fraction: float, replace: bool = False
) -> np.ndarray:
    """Randomly subsample rows from a numpy array.

    Args:
        data: (N, ...) array.
        fraction: fraction of rows to keep (< 1.0).
        replace: whether to sample with replacement.

    Returns:
        Subsampled array of shape (max(1, int(N * fraction)), ...).
        If fraction >= 1.0, returns data unchanged.
    """
    if fraction >= 1.0:
        return data
    n_rows = data.shape[0]
    n_sample = max(1, int(n_rows * fraction))
    indices = np.random.choice(n_rows, n_sample, replace=replace)
    return data[indices]
