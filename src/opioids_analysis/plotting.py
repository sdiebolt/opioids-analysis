from pathlib import Path
from typing import Any

import colorcet as cc
import matplotlib.pyplot as plt
import networkx as nx
import nibabel as nib
import numpy as np
import numpy.typing as npt
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from nilearn import image

from opioids_analysis.pearson import read_session_group_level_pearson
from opioids_analysis.cross_correlation import read_session_group_level_xcorr


def plot_seed_map(
    ax: plt.Axes,
    seed_map: npt.NDArray,
    threshold: float = 0,
    background: npt.NDArray | None = None,
    seed_roi: npt.NDArray | None = None,
    seed_roi_color: str = "black",
    **kwargs: Any,
) -> None:
    """Plot a thresholded seed-based map with a background image and overlaid seed ROI.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    seed_map : numpy.ndarray
        Seed-based map as a 2D array.
    threshold : float, optional
        Voxels with intensities less than `threshold` will be hidden. Default is 0.
    background : numpy.ndarray or None, optional
        Background image as a 2D array, or ``None`` to display no background. Default is
        ``None``.
    seed_roi : numpy.ndarray or None, optional
        Seed ROI as a 2D binary array, or ``None`` to display no ROI. Default is
        ``None``.
    kwargs : dict, optional
        Keywords arguments to pass to ``matplotlib.pyplot.imshow``. Keywords prefixed
        with ``background_`` will be used when plotting `background`, and the rest when
        plotting `seed_map`.
    """
    background_kwargs = {
        k.replace("background_", ""): v
        for k, v in kwargs.items()
        if k.startswith("background_")
    }
    seed_map_kwargs = {
        k: v for k, v in kwargs.items() if not k.startswith("background_")
    }

    if background is not None:
        ax.imshow(
            np.rot90(background),
            aspect=0.1 / 0.11,
            interpolation="none",
            **background_kwargs,
        )

    masked_seed_map: np.ma.MaskedArray = np.ma.masked_array(
        seed_map, mask=np.abs(seed_map) < threshold
    )
    ax.imshow(
        np.rot90(masked_seed_map),
        aspect=0.1 / 0.11,
        interpolation="none",
        **seed_map_kwargs,
    )

    if seed_roi is not None:
        ax.contour(
            np.rot90(seed_roi.squeeze()),
            levels=[0.5],
            linewidths=1,
            antialiased=False,
            corner_mask=False,
            colors=seed_roi_color,
        )


def plot_all_phases_seed_maps(
    seed_maps: npt.NDArray,
    threshold: float = 0,
    background: npt.NDArray | None = None,
    seed_roi: npt.NDArray | None = None,
    seed_roi_color: str = "black",
    axes: npt.NDArray | None = None,
    dpi: int = 100,
    output_path: str | Path | None = None,
    **kwargs,
) -> Figure | None:
    """Plot seed-based maps from all phases in a single-row figure.

    Parameters
    ----------
    seed_maps : numpy.ndarray
        Array of phase seed-based maps with shape ``(phases, x, y)``.
    threshold : float, optional
        Voxels with intensities less than `threshold` will be hidden. Default is 0.
    background : numpy.ndarray or None, optional
        Background image as a 2D array, or ``None`` to display no background. Default is
        ``None``.
    seed_roi : numpy.ndarray or None, optional
        Seed ROI as a 2D binary array, or ``None`` to display no ROI. Default is
        ``None``.
    axes : numpy.ndarray or None, optional
        Array of axes with size ``(1, phases)`` used to plot the correlation matrices.
        If ``None``, a new figure will be created. Default is ``None``.
    dpi: int, optional
        DPI of the figure. Default is 100.
    output_path : str or pathlib.Path or None, optional
        Output path to save the figure to. If not ``None``, the figure is closed after
        saving and isn't returned. If ``None``, the figure is returned instead. Default
        is ``None``.
    kwargs : dict, optional
        Keywords arguments to pass to ``matplotlib.pyplot.imshow``. Keywords prefixed
        with ``background_`` will be used when plotting `background`, and the rest when
        plotting `seed_map`.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting figure.
    """
    n_phases = seed_maps.shape[0]
    if axes is None:
        fig, axes = plt.subplots(
            1, n_phases, figsize=(n_phases * 3, 3), tight_layout=True, dpi=dpi
        )
    else:
        if axes.size != n_phases:
            raise ValueError(
                "'axes' should be a numpy array with as many axes as phases."
            )
        fig = axes[0].get_figure()

    for phase_index, ax in enumerate(axes.ravel()):
        plot_seed_map(
            ax,
            seed_maps[phase_index],
            threshold=threshold,
            background=background,
            seed_roi=seed_roi,
            seed_roi_color=seed_roi_color,
            **kwargs,
        )
        ax.axis("off")

    if not output_path:
        return fig
    fig.savefig(output_path)
    plt.close(fig)

    return None


def plot_correlation_matrix(
    ax: plt.Axes,
    correlation_matrix: npt.NDArray,
    upper_triangle: npt.NDArray | None = None,
    **kwargs: Any,
) -> None:
    """Plot a correlation matrix.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    correlation_matrix : numpy.ndarray
        2D correlation matrix.
    upper_triangle : numpy.ndarray or None, optional
        2D correlation matrix that will replace the upper triangle of
        `correlation_matrix`, or ``None`` to leave `correlation_matrix` unmodified.
        Default is ``None``.
    kwargs : dict, optional
        Keywords arguments to pass to ``matplotlib.pyplot.imshow``. Keywords prefixed
        with ``upper_triangle_`` will be used when plotting `upper_triangle`, and the
        rest when plotting `correlation_matrix`.
    """
    upper_triangle_kwargs = {
        k.replace("upper_triangle_", ""): v
        for k, v in kwargs.items()
        if k.startswith("upper_triangle_")
    }
    correlation_matrix_kwargs = {
        k: v for k, v in kwargs.items() if not k.startswith("upper_triangle_")
    }

    ax.imshow(
        correlation_matrix,
        interpolation="none",
        **correlation_matrix_kwargs,
    )

    if upper_triangle is not None:
        upper_triangle[np.tril_indices(upper_triangle.shape[0], k=-1)] = np.nan
        ax.imshow(
            upper_triangle,
            interpolation="none",
            **upper_triangle_kwargs,
        )


def plot_all_phases_correlation_matrices(
    correlation_matrices: npt.NDArray,
    upper_triangle: npt.NDArray | None = None,
    axes: npt.NDArray | None = None,
    dpi: int = 100,
    output_path: str | Path = "",
    **kwargs,
) -> Figure | None:
    """Plot correlation matrices from all phases in a single-row figure.

    Parameters
    ----------
    correlation_matrices : numpy.ndarray
        Array of phase correlation_matrices with shape ``(phases, rois, rois)``.
    upper_triangle : numpy.ndarray or None, optional
        Array of phase correlation matrices with shape ``(phases, rois, rois)`` that
        will be used to replace upper triangles from ``correlation_matrices``, or
        ``None`` to leave `correlation_matrices` unmodified. Default is ``None``.
    axes : numpy.ndarray or None, optional
        Array of axes with size ``(1, phases)`` used to plot the correlation matrices.
        If ``None``, a new figure will be created. Default is ``None``.
    dpi: int, optional
        DPI of the figure. Default is 100.
    output_path : str or pathlib.Path, optional
        Output path to save the figure to. If provided, the figure is closed after
        saving and isn't returned. If empty, the figure is returned instead. Default is
        an empty string.
    kwargs : dict, optional
        Keywords arguments to pass to ``matplotlib.pyplot.imshow``. Keywords prefixed
        with ``upper_triangle_`` will be used when plotting `upper_triangle`, and the
        rest when plotting `correlation_matrix`.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting figure.
    """
    n_phases = correlation_matrices.shape[0]
    if axes is None:
        fig, axes = plt.subplots(
            1, n_phases, figsize=(n_phases * 3, 3), tight_layout=True, dpi=dpi
        )
    else:
        if axes.size != n_phases:
            raise ValueError(
                "'axes' should be a numpy array with as many axes as phases."
            )
        fig = axes[0].get_figure()

    for phase_index, ax in enumerate(axes.ravel()):
        plot_correlation_matrix(
            ax,
            correlation_matrices[phase_index],
            upper_triangle=upper_triangle[phase_index]
            if upper_triangle is not None
            else None,
            **kwargs,
        )
        ax.axis("off")

    if not output_path:
        return fig
    fig.savefig(output_path)
    plt.close(fig)

    return None


def plot_circular_graph(
    matrix: npt.NDArray,
    edge_threshold: float = 0.0,
    edge_width_factor: int = 5,
    edge_cmap: str | Colormap = cc.cm.coolwarm,
    edge_vmax: float = 1.0,
    edge_vmin: float = -1.0,
    node_cmap: str | Colormap = cc.cm.coolwarm,
    node_vmax: float = 1.0,
    node_vmin: float = -1.0,
    labels: dict[int, int] | None = None,
    ax: plt.Axes | None = None,
    dpi: int = 100,
) -> None:
    """Plot a correlation matrix as a circular graph.

    matrix : numpy.ndarray
        Correlation matrix with shape ``(rois, rois)``.
    edge_threshold : float, optional
        Edges with weights under `edge_threshold` will not be displayed. Default is 0.
    edge_width_factor : int, optional
        Scaling factor used to set the edge width. Default is 5.
    edge_cmap : str or matplotlib.colors.Colormap
        Colormap used to map edge weights to colors. Default is
        ``colorcet.cm.coolwarm``.
    edge_vmin, edge_vmax : float, optional
        Data range that the edge colormap covers. Default are -1.0 and 1.0 respectively.
    node_cmap : str or matplotlib.colors.Colormap
        Colormap used to map node weights to colors. Default is
        ``colorcet.cm.coolwarm``.
    node_vmin, node_vmax : float, optional
        Data range that the node colormap covers. Default are -1.0 and 1.0 respectively.
    labels : dict or None, optional
        Node labels in a dictionary of text labels keyed by node. If ``None``, indices
        from 0 to ``len(matrix)`` will be used. Default is ``None``.
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot the graph on, or ``None`` to create a new figure. Default is
        ``None``.
    dpi : int, optional
        If `ax` is ``None``, DPI of the created figure. Default is 200.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True, dpi=dpi)

    G = nx.from_numpy_array(matrix)

    # When the correlation matrix is full of zeros, no edges are created.
    if G.number_of_edges() > 0:
        edges, weights = zip(
            *[
                x
                for x in nx.get_edge_attributes(G, "weight").items()
                if abs(x[1]) >= edge_threshold
            ]
        )
    else:
        edges = ()
        weights = (0,)

    np.fill_diagonal(matrix, np.nan)
    global_connectivity = np.tanh(np.nanmean(np.arctanh(matrix), axis=0))

    nx.draw_circular(
        G,
        ax=ax,
        node_size=400,
        node_color=global_connectivity,
        cmap=node_cmap,
        vmin=node_vmin,
        vmax=node_vmax,
        edgelist=edges,
        width=np.abs(weights) * edge_width_factor,
        edge_color=weights,
        edge_cmap=edge_cmap,
        edge_vmax=edge_vmax,
        edge_vmin=edge_vmin,
        labels=labels,
        font_size=10,
        with_labels=True,
    )

    ax.set_xlim((-1.2, 1.2))
    ax.set_ylim((-1.2, 1.2))
    ax.axis("off")


def plot_all_phases_circular_graphs(
    correlation_matrices: npt.NDArray,
    edge_threshold: float = 0.0,
    edge_width_factor: int = 5,
    edge_cmap: str | Colormap = cc.cm.coolwarm,
    edge_vmax: float = 1.0,
    edge_vmin: float = -1.0,
    node_cmap: str | Colormap = cc.cm.coolwarm,
    node_vmax: float = 1.0,
    node_vmin: float = -1.0,
    labels: dict[int, int] | None = None,
    axes: npt.NDArray | None = None,
    dpi: int = 100,
    output_path: str | Path = "",
) -> Figure | None:
    """Plot circular graphs from all phase correlation matrices in a single-row figure.

    Parameters
    ----------
    correlation_matrices : numpy.ndarray
        Array of phase correlation_matrices with shape ``(phases, rois, rois)``.
    edge_threshold : float, optional
        Edges with weights under `edge_threshold` will not be displayed. Default is 0.
    edge_width_factor : int, optional
        Scaling factor used to set the edge width. Default is 5.
    edge_cmap : str or matplotlib.colors.Colormap
        Colormap used to map edge weights to colors. Default is
        ``colorcet.cm.coolwarm``.
    edge_vmin, edge_vmax : float, optional
        Data range that the edge colormap covers. Default are -1.0 and 1.0 respectively.
    node_cmap : str or matplotlib.colors.Colormap
        Colormap used to map node weights to colors. Default is
        ``colorcet.cm.coolwarm``.
    node_vmin, node_vmax : float, optional
        Data range that the node colormap covers. Default are -1.0 and 1.0 respectively.
    labels : dict or None, optional
        Node labels in a dictionary of text labels keyed by node. If ``None``,
        indices from 0 to ``len(matrix)`` will be used. Default is ``None``.
    axes : numpy.ndarray or None, optional
        Array of axes with size ``(1, phases)`` used to plot the correlation matrices.
        If ``None``, a new figure will be created. Default is ``None``.
    dpi: int, optional
        DPI of the figure. Default is 100.
    output_path : str or pathlib.Path, optional
        Output path to save the figure to. If provided, the figure is closed after
        saving and isn't returned. If empty, the figure is returned instead. Default is
        an empty string.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting figure.
    """
    n_phases = correlation_matrices.shape[0]
    if axes is None:
        fig, axes = plt.subplots(
            1, n_phases, figsize=(n_phases * 3, 3), tight_layout=True, dpi=dpi
        )
    else:
        if axes.size != n_phases:
            raise ValueError(
                "'axes' should be a numpy array with as many axes as phases."
            )
        fig = axes[0].get_figure()

    for phase_index, ax in enumerate(axes.ravel()):
        plot_circular_graph(
            correlation_matrices[phase_index],
            edge_threshold=edge_threshold,
            edge_width_factor=edge_width_factor,
            edge_cmap=edge_cmap,
            edge_vmax=edge_vmax,
            edge_vmin=edge_vmin,
            node_cmap=node_cmap,
            node_vmax=node_vmax,
            node_vmin=node_vmin,
            labels=labels,
            ax=ax,
        )

    if not output_path:
        return fig
    fig.savefig(output_path)
    plt.close(fig)

    return None


def plot_group_level_pearson(
    group_level_path: str | Path,
    session: str,
    template_img: nib.nifti1.Nifti1Image,
    rois_img: nib.nifti1.Nifti1Image,
    graph_roi_order: list[int],
    graph_roi_labels: dict[int, int] | None,
    output_path: str | Path,
) -> None:
    """Plot figures from the group-level Pearson correlation analysis of a session.

    Parameters
    ----------
    group_level_path : str or pathlib.Path
        Path of the HDF5 file containing group-level results.
    session : str
        Session label.
    template_img : nibabel.nifti1.Nifti1Image
        Template image, used as background in significance seed-based maps.
    rois_img : nibabel.nifti1.Nifti1Image
        Integer ROI image used to plot ROI contours on seed-based maps. 0 corresponds to
        background.
    graph_roi_order : list of int
        Ordering of the ROIs in circular graphs.
    graph_roi_labels : dict or None, optional
        Node labels in a dictionary of labels keyed by node. If ``None``, indices from 0
        to ``len(matrix)`` will be used. Default is ``None``.
    output_path : str or pathlib.Path
        Path to the folder where figures will be saved.
    """
    session_output_path = Path(output_path) / f"{session}"
    session_output_path.mkdir(parents=True, exist_ok=True)

    group_level_pearson = read_session_group_level_pearson(group_level_path, session)
    cor_matrices, seed_maps = (
        group_level_pearson["correlation_matrices"],
        group_level_pearson["seed_maps"],
    )

    # Group-average correlation matrix.
    _ = plot_all_phases_correlation_matrices(
        cor_matrices[:, 0],
        vmax=1,
        vmin=-1,
        cmap=cc.cm.coolwarm,
        output_path=session_output_path / "mean-correlation-matrix.tiff",
    )

    # Differences between average correlation matrix and control.
    _ = plot_all_phases_correlation_matrices(
        cor_matrices[:, 1],
        vmax=0.5,
        vmin=-0.5,
        cmap=cc.cm.gwv,
        output_path=session_output_path / "differences-correlation-matrices.tiff",
    )

    # Significant differences matrix.
    thresholded_differences = cor_matrices[:, 1].copy()
    thresholded_differences[cor_matrices[:, 2] == 0] = 0
    _ = plot_all_phases_correlation_matrices(
        thresholded_differences,
        vmax=0.5,
        vmin=-0.5,
        cmap=cc.cm.gwv,
        output_path=session_output_path / "significance-correlation-matrix.tiff",
    )

    # Combined group-average and significant differences matrix.
    _ = plot_all_phases_correlation_matrices(
        cor_matrices[:, 0],
        upper_triangle=thresholded_differences,
        vmax=1,
        vmin=-1,
        cmap=cc.cm.coolwarm,
        upper_triangle_cmap=cc.cm.gwv,
        upper_triangle_vmax=0.5,
        upper_triangle_vmin=-0.5,
        output_path=session_output_path
        / "mean-and-significance-correlation-matrix.tiff",
    )

    # Group-average circular graph.
    _ = plot_all_phases_circular_graphs(
        cor_matrices[:, 0, graph_roi_order][..., graph_roi_order],
        edge_threshold=0.3,
        node_vmax=0.2,
        node_vmin=-0.2,
        labels=graph_roi_labels,
        output_path=session_output_path / "mean-circular-graph.tiff",
    )

    # Difference between average circular graph and control.
    differences = cor_matrices[:, 1].copy()
    differences[cor_matrices[:, 2] == 0] = 0
    _ = plot_all_phases_circular_graphs(
        differences[:, graph_roi_order][..., graph_roi_order],
        edge_threshold=0,
        edge_vmax=0.5,
        edge_vmin=-0.5,
        edge_cmap=cc.cm.gwv,
        node_vmax=0.2,
        node_vmin=-0.2,
        node_cmap=cc.cm.gwv,
        labels=graph_roi_labels,
        output_path=session_output_path / "differences-circular-graphs.tiff",
    )

    n_rois = cor_matrices.shape[-1]
    for roi_index in range(n_rois):
        current_roi_mask = (
            image.math_img(f"x == {roi_index + 1}", x=rois_img)
            .get_fdata()
            .squeeze()
            .astype(bool)
        )

        # Group-average seed-based map.
        plot_all_phases_seed_maps(
            seed_maps[:, roi_index, 0],
            seed_roi=current_roi_mask,
            vmax=1,
            vmin=-1,
            cmap=cc.cm.coolwarm,
            output_path=session_output_path
            / f"mean-seed-map_roi-{roi_index + 1:02d}.tiff",
        )

        # Difference between average seed-based map and control.
        plot_all_phases_seed_maps(
            seed_maps[:, roi_index, 1],
            seed_roi=current_roi_mask,
            seed_roi_color="red",
            vmax=0.5,
            vmin=-0.5,
            cmap=cc.cm.gwv,
            output_path=session_output_path
            / f"difference-seed-map_roi-{roi_index + 1:02d}.tiff",
        )

        # Significance seed-based map.
        thresholded_differences = seed_maps[:, roi_index, 1].copy()
        thresholded_differences[seed_maps[:, roi_index, 2] == 0] = np.nan
        plot_all_phases_seed_maps(
            thresholded_differences,
            background=template_img.get_fdata().squeeze(),
            background_cmap="gray",
            seed_roi=current_roi_mask,
            seed_roi_color="red",
            vmax=0.5,
            vmin=-0.5,
            cmap=cc.cm.gwv,
            output_path=session_output_path
            / f"significance-seed-map_roi-{roi_index + 1:02d}.tiff",
        )


def plot_all_lags_xcorr_maps(
    xcorr_maps: npt.NDArray,
    threshold: float = 0,
    background: npt.NDArray | None = None,
    seed_roi: npt.NDArray | None = None,
    seed_roi_color: str = "black",
    axes: npt.NDArray | None = None,
    dpi: int = 100,
    output_path: str | Path | None = None,
    **kwargs,
) -> Figure | None:
    """Plot cross-correlation maps in a single-row figure.

    Parameters
    ----------
    xcorr_maps : numpy.ndarray
        Array of cross-correlation maps with shape ``(lags, x, y)``.
    threshold : float, optional
        Voxels with intensities less than `threshold` will be hidden. Default is 0.
    background : numpy.ndarray or None, optional
        Background image as a 2D array, or ``None`` to display no background. Default is
        ``None``.
    seed_roi : numpy.ndarray or None, optional
        Seed ROI as a 2D binary array, or ``None`` to display no ROI. Default is
        ``None``.
    axes : numpy.ndarray or None, optional
        Array of axes with size ``(1, lags)`` used to plot the correlation matrices.
        If ``None``, a new figure will be created. Default is ``None``.
    dpi: int, optional
        DPI of the figure. Default is 100.
    output_path : str or pathlib.Path or None, optional
        Output path to save the figure to. If not ``None``, the figure is closed after
        saving and isn't returned. If ``None``, the figure is returned instead. Default
        is ``None``.
    kwargs : dict, optional
        Keywords arguments to pass to ``matplotlib.pyplot.imshow``. Keywords prefixed
        with ``background_`` will be used when plotting `background`, and the rest when
        plotting `seed_map`.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting figure.
    """
    n_lags = xcorr_maps.shape[0]
    if axes is None:
        fig, axes = plt.subplots(
            1, n_lags, figsize=(n_lags * 3, 3), tight_layout=True, dpi=dpi
        )
    else:
        if axes.size != n_lags:
            raise ValueError(
                "'axes' should be a numpy array with as many axes as phases."
            )
        fig = axes[0].get_figure()

    for lag_index, ax in enumerate(axes.ravel()):
        plot_seed_map(
            ax,
            xcorr_maps[lag_index],
            threshold=threshold,
            background=background,
            seed_roi=seed_roi,
            seed_roi_color=seed_roi_color,
            **kwargs,
        )
        ax.axis("off")

    if not output_path:
        return fig
    fig.savefig(output_path)
    plt.close(fig)

    return None


def plot_group_level_xcorr(
    group_level_path: str | Path,
    session: str,
    template_img: nib.nifti1.Nifti1Image,
    rois_img: nib.nifti1.Nifti1Image,
    lags_to_skip: int,
    output_path: str | Path,
) -> None:
    """Plot figures from the group-level cross-correlation analysis of a session.

    Parameters
    ----------
    group_level_path : str or pathlib.Path
        Path of the HDF5 file containing group-level results.
    session : str
        Session label.
    template_img : nibabel.nifti1.Nifti1Image
        Template image, used as background in significance seed-based maps.
    rois_img : nibabel.nifti1.Nifti1Image
        Integer ROI image used to plot ROI contours on seed-based maps. 0 corresponds to
        background.
    lags_to_skip : int
        Number of lags to skip when plotting cross-correlation map. Skipping 0 lag will
        plot all lags, while skipping 5 would result in plotting one lag every six.
    output_path : str or pathlib.Path
        Path to the folder where figures will be saved.
    """
    session_output_path = Path(output_path) / f"{session}"
    session_output_path.mkdir(parents=True, exist_ok=True)

    xcorr_maps = read_session_group_level_xcorr(group_level_path, session)["xcorr_maps"]

    n_phases = xcorr_maps.shape[0]
    n_rois = xcorr_maps.shape[1]
    n_lags = xcorr_maps.shape[2]

    # Lag zero is always at the center of the axis (thus the number of lags is always
    # odd). Plots must always show the lag zero at the center, and skipped lags must be
    # determined from there, positively and negatively.
    max_lag = int((n_lags - 1) / 2)
    retained_lags = np.arange(0, max_lag + 1, lags_to_skip + 1)
    retained_lags = np.concatenate((-retained_lags[:0:-1], retained_lags)) + max_lag
    xcorr_maps = xcorr_maps[:, :, retained_lags]

    xcorr_maps[:, :, :, 0] = (
        xcorr_maps[:, :, :, 0] - xcorr_maps[:, :, :, 0].mean((3, 4), keepdims=True)
    ) / xcorr_maps[:, :, :, 0].std((3, 4), ddof=1, keepdims=True)

    for roi_index in range(n_rois):
        for phase_index in range(n_phases):
            current_roi_mask = (
                image.math_img(f"x == {roi_index + 1}", x=rois_img)
                .get_fdata()
                .squeeze()
                .astype(bool)
            )

            # Group-average cross-correlation map.
            plot_all_lags_xcorr_maps(
                xcorr_maps[phase_index, roi_index, :, 0],
                seed_roi=current_roi_mask,
                vmax=5,
                vmin=-5,
                cmap=cc.cm.coolwarm,
                output_path=session_output_path
                / f"mean-xcorr-maps_phase-{phase_index + 1:02d}_roi-{roi_index + 1:02d}.tiff",
            )

            # Difference between average seed-based map and control.
            plot_all_lags_xcorr_maps(
                xcorr_maps[phase_index, roi_index, :, 1],
                seed_roi=current_roi_mask,
                seed_roi_color="red",
                vmax=0.5,
                vmin=-0.5,
                cmap=cc.cm.gwv,
                output_path=session_output_path
                / f"difference-xcorr-maps_phase-{phase_index + 1:02d}_roi-{roi_index + 1:02d}.tiff",
            )

            # Significance seed-based map.
            thresholded_differences = xcorr_maps[phase_index, roi_index, :, 1].copy()
            thresholded_differences[xcorr_maps[phase_index, roi_index, :, 2] == 0] = (
                np.nan
            )
            plot_all_lags_xcorr_maps(
                thresholded_differences,
                background=template_img.get_fdata().squeeze(),
                background_cmap="gray",
                seed_roi=current_roi_mask,
                seed_roi_color="red",
                vmax=0.5,
                vmin=-0.5,
                cmap=cc.cm.gwv,
                output_path=session_output_path
                / f"significance-xcorr-maps_phase-{phase_index + 1:02d}_roi-{roi_index + 1:02d}.tiff",
            )
