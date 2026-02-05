"""Utilities for multimodal analysis."""

from pathlib import Path

import h5py as h5
import numpy as np
import numpy.typing as npt


def load_instant_velocity(path: str | Path) -> npt.NDArray:
    """Load tracking data and compute instant speed from resampled position.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to an HDF5 tracking file.

    Returns
    -------
    numpy.ndarray
        1D instant velocity array.
    """

    with h5.File(path, "r") as f:
        x_resampled = f["xResampled"][:].squeeze()
        y_resampled = f["yResampled"][:].squeeze()
        t_resampled = f["tResampled"][:].squeeze()

    dx = np.diff(x_resampled)
    dy = np.diff(y_resampled)
    dt = np.diff(t_resampled).mean()

    dx = np.concatenate((dx, np.array([dx[0]])))
    dy = np.concatenate((dy, np.array([dy[0]])))

    return np.sqrt(dx**2 + dy**2) * dt


def compute_moving_time_percentage(
    velocity: npt.NDArray, window_size: int = 1200, threshold: float = 5.0
) -> npt.NDArray:
    """Compute the moving time percentage (speed above `threshold`) on sub-windows.

    Parameters
    ----------
    velocity : numpy.ndarray
        Instant mouse velocity array.
    window_size : int, optional
        Size of the windows in which to separate `v`. The moving time percentage will be
        computed in each window.
    threshold : float, optional
        Velocity threshold above which the mouse is considered moving, in cm/s. Default
        is 5.0.

    Returns
    -------
    numpy.ndarray
        Array of size ``v.size // window_size`` containing the percentage of time the
        mouse is moving with velocity above `threshold` in each window.
    """

    return 100 * (
        np.lib.stride_tricks.sliding_window_view(velocity, window_size)[::window_size]
        > threshold
    ).mean(axis=1)


def compute_max_velocities(
    velocity: npt.NDArray, window_size: int = 1200
) -> npt.NDArray:
    """Compute the max velocities in sub-windows.

    Max velocities are computed using the 99.9% percentile of the velocity in each
    window to avoid outliers.

    Parameters
    ----------
    velocity : numpy.ndarray
        Instant mouse velocity array.
    window_size : int, optional
        Size of the windows in which to separate `v`. The moving time percentage will be
        computed in each window.

    Returns
    -------
    numpy.ndarray
        Array of size ``v.size // window_size`` containing the percentage of time the
        mouse is moving with velocity above `threshold` in each window.
    """

    return np.percentile(
        np.lib.stride_tricks.sliding_window_view(velocity, window_size)[::window_size],
        99.9,
        axis=1,
    )
