"""Utilities for Pearson correlation statistical analysis."""

import re
import warnings
from pathlib import Path
from typing import TypedDict, cast

import h5py as h5
import nibabel as nib
import numpy as np
import numpy.typing as npt
import statsmodels.api as sm
from joblib import Parallel, delayed
from nilearn import connectome, masking
from nilearn.signal import _standardize
from scikit_posthocs import posthoc_dunn
from scipy.stats import mannwhitneyu


class SubjectLevelPearson(TypedDict):
    seed_maps: npt.NDArray
    correlation_matrices: npt.NDArray


def compute_subject_level_pearson(
    nii_paths: list[str | Path],
    brain_mask_img: nib.nifti1.Nifti1Image,
    rois_img: nib.nifti1.Nifti1Image,
    sample_masks: npt.NDArray | None = None,
) -> SubjectLevelPearson:
    """Compute subject-level Pearson correlation matrices and seed-based maps.

    Parameters
    ----------
    nii_paths : list of str or list of pathlib.Path
        List of NIfTI files ordered by run index, corresponding to analysis phases of a
        session. `compute_subject_level_pearson` will check that indices of the ``run``
        entities in `nii_paths` are consecutive and ordered. Otherwise, an error will be
        raised.
    brain_mask_img : nibabel.nifti1.Nifti1Image
        Brain mask with same spatial shape as the functional NIfTI images, used to
        extract voxel signals.
    rois_img : nibabel.nifti1.Nifti1Image
        ROIs mask with same spatial shape as the functional NIfTI images.
    sample_masks : list of numpy.ndarray or None
        Array of sample masks with shape ``(phase, time)``, where ``phase`` is the
        number of phases. If ``None``, no scrubbing is performed. Default is ``None``.

    Returns
    -------
    dict
        Dictionary with two keys:

        * ``seed_maps``: dictionary with subject names as keys and seed-based maps as
          values, with shape ``(phases, rois, x, y)``.
        * ``correlation_matrices``: dictionary with subject names as keys and
          correlation matrices as values, with shape ``(phases, rois, rois)``.

    Raises
    ------
    ValueError
        If any NIfTI path does not contain the ``run`` entity, or if ``run`` entities
        aren't consecutive and ordered.
    """
    # We check that the run entities in nii_paths are consecutive and ordered to avoid
    # any user error, e.g. misordering and/or missing runs.
    run_entities = [re.search(r"_run-(\d+)", str(p)) for p in nii_paths]
    if any(run_entity is None for run_entity in run_entities):
        raise ValueError("NIfTI paths must contain the 'run' entity.")

    run_indices = [
        int(cast(re.Match[str], run_entity).group(1)) for run_entity in run_entities
    ]
    if run_indices != list(range(1, len(nii_paths) + 1)):
        raise ValueError(
            "NIfTI paths must be ordered by run indices, and run indices must be "
            "consecutive."
        )

    rois_img.set_sform(brain_mask_img.affine)
    rois_data = masking.apply_mask(rois_img, brain_mask_img)
    roi_labels = np.unique(rois_data[rois_data > 0])

    n_phases = len(nii_paths)
    n_rois = len(roi_labels)

    nii_img = cast(nib.nifti1.Nifti1Image, nib.load(nii_paths[0]))
    seed_maps = np.zeros(
        (
            n_phases,
            n_rois,
            nii_img.shape[0],
            nii_img.shape[2],
        )
    )
    correlation_matrices = np.zeros((n_phases, n_rois, n_rois))
    for phase_index, nii_path in enumerate(nii_paths):
        phase_sample_mask = (
            sample_masks[phase_index] if sample_masks is not None else None
        )

        # Phases that have less than 5% usable samples are discarded. The corresponding
        # seed-based maps and correlation matrices are set to NaN to remove their
        # contribution from the group-level analysis.
        if (
            phase_sample_mask is not None
            and phase_sample_mask.sum() < 0.05 * phase_sample_mask.size
        ):
            seed_maps[phase_index] = np.full(
                (n_rois,) + brain_mask_img.shape, np.nan
            ).squeeze()
            correlation_matrices[phase_index] = np.full((n_rois, n_rois), np.nan)
            continue

        cleaned_signals = masking.apply_mask(nii_path, brain_mask_img)

        if phase_sample_mask is not None:
            cleaned_signals = cleaned_signals[phase_sample_mask]

        rois_signals = np.array(
            [
                cleaned_signals[:, rois_data == roi_label].mean(axis=1)
                for roi_label in roi_labels
            ]
        ).T

        # Ignore FutureWarning from nilearn.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            correlation_matrices[phase_index] = connectome.ConnectivityMeasure(
                kind="correlation"
            ).fit_transform([rois_signals])[0]

        cleaned_signals = _standardize(cleaned_signals, standardize="zscore_sample")
        rois_signals = _standardize(rois_signals, standardize="zscore_sample")

        for roi_index in range(rois_signals.shape[1]):
            seed_to_voxel_correlations = (
                cleaned_signals.T @ rois_signals[:, roi_index]
            ) / (cleaned_signals.shape[0] - 1)
            seed_maps[phase_index, roi_index] = (
                masking.unmask(seed_to_voxel_correlations, brain_mask_img)
                .get_fdata()
                .squeeze()
            )

    return {"seed_maps": seed_maps, "correlation_matrices": correlation_matrices}


class SessionSubjectLevelPearson(TypedDict):
    seed_maps: dict[str, npt.NDArray]
    correlation_matrices: dict[str, npt.NDArray]


def write_session_subject_level_pearson(
    path: str | Path,
    session: str,
    results: SessionSubjectLevelPearson,
    overwrite: bool = False,
) -> None:
    """Write subject-level Pearson correlation results for all subjects of a session.

    Subject-level Pearson correlation results are saved as an HDF5 file containing
    sub-datasets with names ``seed_maps/session/subject`` and
    ``correlation_matrices/session/subject``, where ``session`` and ``subject`` are
    session and subject labels respectively. This file can be written using
    `write_session_subject_level_pearson`.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to an HDF5 file where the subject-level results will be saved.
    session : str
        Session label.
    results : dict
        Dictionary with two keys:

        * ``seed_maps``: dictionary with subject names as keys and seed-based maps as
          values, with shape ``(phases, rois, x, y)``.
        * ``correlation_matrices``: dictionary with subject names as keys and
          correlation matrices as values, with shape ``(phases, rois, rois)``.
    overwrite : bool, optional
        If ``True``, overwrite existing session data. If ``False`` and session data
        already exists, skip writing. Default is ``False``.
    """
    with h5.File(path, "a") as f:
        # Check if session already exists
        if "seed_maps" in f and session in f["seed_maps"]:
            if not overwrite:
                return
            # Delete existing session data to overwrite
            del f[f"seed_maps/{session}"]
            del f[f"correlation_matrices/{session}"]

        for subject in results["seed_maps"].keys():
            f.create_dataset(
                f"correlation_matrices/{session}/{subject}",
                data=results["correlation_matrices"][subject],
            )
            f.create_dataset(
                f"seed_maps/{session}/{subject}", data=results["seed_maps"][subject]
            )


def read_session_subject_level_pearson(
    path: str | Path, session: str
) -> SessionSubjectLevelPearson:
    """Read subject-level Pearson correlation results for all subjects of a session.

    Subject-level Pearson correlation results are saved as an HDF5 file containing
    sub-datasets with names ``seed_maps/session/subject`` and
    ``correlation_matrices/session/subject``, where ``session`` and ``subject`` are
    session and subject labels respectively. This file can be written using
    `write_session_subject_level_pearson`.

    Parameters
    ----------
    path : str
        Path of the HDF5 file containing subject-level results.
    session : str
        Session label.

    Returns
    -------
    dict
        Dictionary with two keys:

        * ``seed_maps``: dictionary with subject names as keys and seed-based maps as
          values, with shape ``(phases, rois, x, y)``.
        * ``correlation_matrices``: dictionary with subject names as keys and
          correlation matrices as values, with shape ``(phases, rois, rois)``.
    """
    with h5.File(path, "r") as f:
        return {
            "seed_maps": {
                subject: maps[...]
                for subject, maps in f[f"seed_maps/{session}"].items()
            },
            "correlation_matrices": {
                subject: matrices[...]
                for subject, matrices in f[f"correlation_matrices/{session}"].items()
            },
        }


def read_session_sample_masks(path: str | Path, session: str) -> dict[str, npt.NDArray]:
    """Read sample masks for all subjects of a session.

    Sample masks are saved as an HDF5 file containing sub-datasets with names
    ``sample_masks/session/subject`` where ``session`` and ``subject`` are session and
    subject labels respectively. This file can be written using
    `write_session_sample_masks`.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to an HDF5 file containing session sample masks.
    session : str
        Session label.

    Returns
    -------
    dict
        Dictionary with subject names as keys and sample masks with shape ``(phase,
        time)`` as values.
    """
    with h5.File(path, "r") as f:
        return {
            subject: mask[...] for subject, mask in f[f"sample_masks/{session}"].items()
        }


def compare_correlation_values(
    group1: npt.NDArray, group2: npt.NDArray, fdr_threshold: float = 0.05
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Compare two samples of Pearson correlation arrays.

    Comparison is performed element-wise using the Mann-Whitney *U* test. The first
    dimension of `group1` and `group2` corresponds to samples. The Benjamini-Hochberg
    procedure is applied to control the false discovery rate (FDR) at level
    `fdr_threshold`.

    Parameters
    ----------
    group1, group2 : numpy.ndarray
        Arrays of correlation values with shape ``(samples, ...)``. The number of
        samples in each group may be different.
    fdr_threshold : float, optional
        Threshold at which to control the FDR. Default is 0.05.

    Returns
    -------
    mean_group1 : numpy.ndarray
        Sample average of the `group1` array.
    difference : numpy.ndarray
        Difference of the sample means from `group1` and `group2`.
    significance : numpy.ndarray
        Integer array with same shape as `mean_group1` with +1, respectively -1, for
        significant positive, respectively negative, differences and 0 for
        non-significant differences (after applying the Benjamini-Hochberg procedure).
    """
    # Ignore warnings for the diagonals.
    with np.errstate(divide="ignore", invalid="ignore"):
        group1 = np.arctanh(group1)
        group2 = np.arctanh(group2)

    mean_group1 = np.tanh(np.nanmean(group1, axis=0))
    mean_group2 = np.tanh(np.nanmean(group2, axis=0))
    difference = mean_group1 - mean_group2

    # Considering that our samples are small, We use non parametric testing to avoid
    # making any assumption on the distribution of the correlation coefficients.
    p_values = mannwhitneyu(
        group1, group2, alternative="two-sided", nan_policy="omit"
    ).pvalue

    # FDR correction is used to account for multiple testing.
    corrected_p_values = sm.stats.multipletests(p_values, method="fdr_bh")[1]

    significance = np.sign(difference)
    significance[corrected_p_values >= fdr_threshold] = 0

    return mean_group1, difference, significance


def _matrix2tril(matrix: npt.NDArray) -> npt.NDArray:
    """Return the lower triangles from symmetric matrices.

    Parameters
    ----------
    matrix : numpy.ndarray
        Symmetric matrices to extract the lower triangle from, as an N-dimensional array
        with shape ``(..., n, n)``.

    Returns
    -------
    numpy.ndarray
        Lower triangles from `matrix`, with shape ``(..., n * (n - 1) / 2)``
    """
    return matrix[..., *np.tril_indices(matrix.shape[-1], k=-1)]  # type: ignore[syntax]


def _tril2matrix(tril: npt.NDArray) -> npt.NDArray:
    """Return a symmetric matrix from its lower triangle vector.

    Parameters
    ----------
    tril : numpy.ndarray
        1D array corresponding to the lower triangle of a symmetric matrix, obtained
        using `_matrix2tril`.

    Returns
    -------
    numpy.ndarray
        The corresponding symmetric matrix.
    """
    # The formula below was obtained by solving the quadratic formula
    # n_rows * (n_rows - 1) / 2 = tril.shape[-1].
    n_rows = int(1 + np.sqrt(1 + 8 * tril.shape[-1]) / 2)

    matrix = np.zeros(tril.shape[:-1] + (n_rows, n_rows))
    matrix[..., *np.tril_indices(n_rows, k=-1)] = tril
    return matrix + np.transpose(matrix, axes=(*range(matrix.ndim - 2), -1, -2))


def compare_correlation_matrices(
    group1: npt.NDArray,
    group2: npt.NDArray,
    fdr_threshold: float = 0.05,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Compare two samples of Pearson correlation matrices.

    Comparison is performed element-wise considering only the lower matrix triangles and
    using the Mann-Whitney *U* test. The first dimension of `group1` and `group2`
    corresponds to samples. The Benjamini-Hochberg procedure is applied to control the
    false discovery rate (FDR) at level `fdr_threshold`.

    Parameters
    ----------
    group1, group2 : numpy.ndarray
        Two samples of correlation matrices to compare, with shape ``(samples, x, x)``.
        The number of samples in each group may be different.
    fdr_threshold : float, optional
        Threshold at which to control the FDR. Default is 0.05.

    Returns
    -------
    mean_group1 : numpy.ndarray
        Sample average of the `group1` matrix.
    difference : numpy.ndarray
        Difference of the sample means from `group1` and `group2`.
    significance : numpy.ndarray
        Integer array with same shape as `mean_group1` with +1, respectively -1, for
        significant positive, respectively negative, differences and 0 for
        non-significant differences (after applying the Benjamini-Hochberg procedure).
    """
    lower_triangle_group1 = _matrix2tril(group1)
    lower_triangle_group2 = _matrix2tril(group2)

    comparison_result = compare_correlation_values(
        lower_triangle_group1, lower_triangle_group2, fdr_threshold=fdr_threshold
    )

    return cast(
        tuple[npt.NDArray, npt.NDArray, npt.NDArray],
        tuple(map(_tril2matrix, comparison_result)),
    )


def compare_correlation_values_anova(
    treatment_groups: list[npt.NDArray],
    control_group: npt.NDArray,
    fdr_threshold: float = 0.05,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Compare multiple samples of Pearson correlation arrays using ANOVA

    Comparison is performed element-wise. The first dimension of arrays in `groups`
    corresponds to samples. The Benjamini-Hochberg procedure is applied to control the
    false discovery rate (FDR) at level `fdr_threshold`.

    Parameters
    ----------
    treatment_groups : list of numpy.ndarray
        Treatment groups, as arrays of correlation values with shape ``(samples, ...)``.
        The number of samples in each group may be different.
    control_group : numpy.ndarray
        Control group, as an array of correlation values with shape ``(samples, ...)``.
        The number of samples in this group may be different from the number of samples
        in the other groups.
    fdr_threshold : float, optional
        Threshold at which to control the FDR. Default is 0.05.

    Returns
    -------
    mean_treatment_groups : list of numpy.ndarray
        Sample average of the `treatment_groups` arrays.
    differences : numpy.ndarray
        Differences of the sample means from `treatment_groups` and `control_group`.
    significances : numpy.ndarray
        Integer array with same shape as `mean_group1` with +1, respectively -1, for
        significant positive, respectively negative, differences and 0 for
        non-significant differences (after applying the Benjamini-Hochberg procedure).
    """
    # Fisher transform before computing averages and differences. Ignore warnings for
    # the diagonals.
    with np.errstate(divide="ignore", invalid="ignore"):
        treatment_groups = [np.arctanh(g) for g in treatment_groups]
        control_group = np.arctanh(control_group)

    mean_treatment_groups = [np.nanmean(g, axis=0) for g in treatment_groups]
    mean_control_group = np.nanmean(control_group, axis=0)

    # Differences are computed and then back-transformed to correlations using the
    # inverse Fisher transformation.
    differences = np.tanh([x - mean_control_group for x in mean_treatment_groups])

    mean_treatment_groups = [np.tanh(m) for m in mean_treatment_groups]
    mean_control_group = np.tanh(mean_control_group)

    n_correlation_values = control_group.shape[1]
    pvalues = np.zeros((len(treatment_groups), n_correlation_values))
    for corr_index in range(n_correlation_values):
        # We store the p-values comparing each treatment group with the control group
        # (but no the pairwise comparisons between treatment groups, nor the pairwise
        # comparison betweeen the control group and itself).
        pvalues[:, corr_index] = posthoc_dunn(
            [x[:, corr_index] for x in treatment_groups + [control_group]],
        ).values[:-1, -1]

    reject = sm.stats.multipletests(
        pvalues.ravel(), alpha=fdr_threshold, method="fdr_bh"
    )[0].reshape(pvalues.shape)

    significances = np.where(reject, np.sign(differences), 0)

    return mean_treatment_groups, differences, significances


def compare_correlation_matrices_anova(
    treatment_groups: list[npt.NDArray],
    control_group: npt.NDArray,
    fdr_threshold: float = 0.05,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Compare multiple samples of Pearson correlation matrices using ANOVA.

    Comparison is performed element-wise considering only the lower matrix triangles.
    The first dimension of arrays in `treatment_groups` and `control_group` corresponds
    to samples. The Benjamini-Hochberg procedure is applied to control the false
    discovery rate (FDR) at level `fdr_threshold` after Dunn's test.

    Parameters
    ----------
    treatment_groups : list of numpy.ndarray
        Treatment groups, as arrays of correlation matrices with shape
        ``(samples, x, x)``. The number of samples in each group may be different.
    control_group : numpy.ndarray
        Control group, as an array of correlation matrices with shape ``(samples, x,
        x)``. The number of samples in this group may be different from the number of
        samples in the other groups.
    fdr_threshold : float, optional
        Threshold at which to control the FDR. Default is 0.05.

    Returns
    -------
    mean_treatment_groups : list of numpy.ndarray
        Sample average of the `treatment_groups` matrices.
    differences : numpy.ndarray
        Difference of the sample means from `treatment_groups` and `control_group`, with
        shape ``(len(treatment_groups), x, x)``.
    significance : numpy.ndarray
        Integer array with same shape as `differences` with +1, respectively -1, for
        significant positive, respectively negative, differences and 0 for
        non-significant differences (after applying the Benjamini-Hochberg procedure).
    """
    lower_triangle_treatment = [_matrix2tril(g) for g in treatment_groups]
    lower_triangle_control = _matrix2tril(control_group)

    mean_treatment_groups, differences, significances = (
        compare_correlation_values_anova(
            treatment_groups=lower_triangle_treatment,
            control_group=lower_triangle_control,
            fdr_threshold=fdr_threshold,
        )
    )

    return (
        list(map(_tril2matrix, mean_treatment_groups)),
        _tril2matrix(differences),
        _tril2matrix(significances),
    )


def _unmask(arr: npt.NDArray, mask: npt.NDArray) -> npt.NDArray:
    """Reconstruct an array that was indexed by a boolean mask.

    Parameters
    ----------
    arr : numpy.ndarray
        An array resulting from indexing an N-D array using `mask`.
    mask : numpy.ndarray
        A boolean array.

    Returns
    -------
    numpy.ndarray
        The reconstructed array, with zeros where `mask` is ``False``.
    """
    reconstructed = np.zeros(arr.shape[:-1] + mask.shape)
    reconstructed[..., mask] = arr
    return reconstructed


def compare_seed_maps_anova(
    treatment_groups: list[npt.NDArray],
    control_group: npt.NDArray,
    mask: npt.NDArray,
    fdr_threshold: float = 0.05,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Compare two samples of seed-based maps.

    Comparison is performed element-wise using ANOVA. The first
    dimension of matrices in `treatment_groups` and `control_group` corresponds to
    samples. The Benjamini-Hochberg procedure is applied to control the false discovery
    rate (FDR) at level `fdr_threshold`.

    Parameters
    ----------
    treatment_groups : list of numpy.ndarray
        Treatment groups, as arrays of correlation matrices with shape
        ``(samples, x, y, z)``. The number of samples in each group may be different.
    control_group : numpy.ndarray
        Control group, as an array of correlation matrices with shape ``(samples, x, y,
        z)``. The number of samples in this group may be different from the number of
        samples in the other groups.
    mask : numpy.ndarray
        Boolean mask. Seed-based maps will be compared only where `mask` is ``True``.
    fdr_threshold : float, optional
        Threshold at which to control the FDR. Default is 0.05.

    Returns
    -------
    mean_treatment_groups : list of numpy.ndarray
        Sample average of the `treatment_groups` matrices.
    differences : numpy.ndarray
        Difference of the sample means from `treatment_groups` and `control_group`, with
        shape ``(len(treatment_groups), x, y, z)``.
    significance : numpy.ndarray
        Integer array with same shape as `differences` with +1, respectively -1, for
        significant positive, respectively negative, differences and 0 for
        non-significant differences (after applying the Benjamini-Hochberg procedure).
    """
    mean_treatment_groups, differences, significances = (
        compare_correlation_values_anova(
            [x[:, mask] for x in treatment_groups],
            control_group[:, mask],
            fdr_threshold=fdr_threshold,
        )
    )
    return cast(
        tuple[npt.NDArray, npt.NDArray, npt.NDArray],
        (
            list(
                map(lambda x: _unmask(x, mask), mean_treatment_groups),
            ),
            _unmask(differences, mask),
            _unmask(significances, mask),
        ),
    )


def compare_seed_maps(
    group1: npt.NDArray,
    group2: npt.NDArray,
    mask: npt.NDArray,
    fdr_threshold: float = 0.05,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Compare two samples of seed-based maps.

    Comparison is performed element-wise using the Mann-Whitney *U* test. The first
    dimension of `group1` and `group2` corresponds to samples. The Benjamini-Hochberg
    procedure is applied to control the false discovery rate (FDR) at level
    `fdr_threshold`.

    Parameters
    ----------
    group1, group2 : numpy.ndarray
        Two samples of seed-bsaed maps to compare, with shape ``(samples, x, y, z)``.
        The number of samples in each group may be different.
    mask : numpy.ndarray
        Boolean mask. Seed-based maps will be compared only where `mask` is ``True``.
    fdr_threshold : float, optional
        Threshold at which to control the FDR. Default is 0.05.

    Returns
    -------
    mean_group1 : numpy.ndarray
        Sample average of the `group1` seed-based map.
    difference : numpy.ndarray
        Difference of the sample means from `group1` and `group2`.
    significance : numpy.ndarray
        Integer array with same shape as `mean_group1` with +1, respectively -1, for
        significant positive, respectively negative, differences and 0 for
        non-significant differences (after applying the Benjamini-Hochberg procedure).
    """
    comparison_result = compare_correlation_values(
        group1[:, mask],
        group2[:, mask],
        fdr_threshold=fdr_threshold,
    )
    return cast(
        tuple[npt.NDArray, npt.NDArray, npt.NDArray],
        tuple(map(lambda x: _unmask(x, mask), comparison_result)),
    )


def compute_group_level_pearson_anova(
    subject_level_path: str | Path,
    treatment_sessions: list[str],
    control_session: str,
    brain_mask: npt.NDArray,
    fdr_threshold: float = 0.05,
    n_jobs: int = -1,
):
    """Compute group-level Pearson correlation matrices and seed-based maps.

    Group-level results are group-average correlation matrices and seed-based maps and
    their corresponding differences and statistical significance relative to the
    control, for each phase.

    Parameters
    ----------
    subject_level_path : str or pathlib.Path
        Path of the HDF5 file containing subject-level results.
    treatment_sessions : str
        Label of the treatment sessions.
    control_session : str
        Label of the control session.
    brain_mask : numpy.ndarray
        Boolean mask of the brain. Seed-based maps will be compared only where
        `brain_mask` is ``True``.
    fdr_threshold : float, optional
        Threshold at which to control the FDR. Default is 0.05.
    n_jobs : int, optional
        The maximum number of concurrently running jobs. If -1 all CPUs are used.
        Default is -1.

    Returns
    -------
    dict
        Dictionary with keys:

        * ``correlation_matrices``: group-level correlation matrices with shape
          ``(phases, 3, len(treatment_sessions), rois, rois)``. The second dimension
          corresponds to the mean correlation matrix, the difference matrix between
          session and control, and the significance matrix.
        * ``seed_maps``: group-level seed-based maps with shape ``(phases, rois, 3,
        len(treatment_sessions), x, y)``. The third dimension corresponds to the mean
        seed-based map, the difference map between session and control, and the
        significance map.
    """
    cor_matrices_treatments = []
    seed_maps_treatments = []
    for session in treatment_sessions:
        session_results = read_session_subject_level_pearson(
            subject_level_path, session
        )
        cor_matrices_treatments.append(
            np.asarray(list(session_results["correlation_matrices"].values()))
        )
        seed_maps_treatments.append(
            np.asarray(list(session_results["seed_maps"].values()))
        )

    results_control = read_session_subject_level_pearson(
        subject_level_path, control_session
    )
    cor_matrices_control = np.asarray(
        list(results_control["correlation_matrices"].values())
    )
    seed_maps_control = np.asarray(list(results_control["seed_maps"].values()))

    n_phases = cor_matrices_treatments[0].shape[1]
    n_rois = cor_matrices_treatments[0].shape[-1]
    with Parallel(n_jobs=n_jobs) as parallel:

        def compare_phase_cor_matrices(phase):
            return compare_correlation_matrices_anova(
                [x[:, phase] for x in cor_matrices_treatments],
                cor_matrices_control[:, phase],
                fdr_threshold=fdr_threshold,
            )

        group_cor_matrices = parallel(
            delayed(compare_phase_cor_matrices)(phase) for phase in range(n_phases)
        )

        def compare_phase_roi_seed_maps(phase, roi):
            return compare_seed_maps_anova(
                [x[:, phase, roi] for x in seed_maps_treatments],
                seed_maps_control[:, phase, roi],
                mask=brain_mask,
                fdr_threshold=fdr_threshold,
            )

        group_seed_maps = parallel(
            delayed(compare_phase_roi_seed_maps)(phase, roi)
            for phase in range(n_phases)
            for roi in range(n_rois)
        )

    group_cor_matrices = np.array(group_cor_matrices)
    group_seed_maps = np.array(group_seed_maps)
    group_seed_maps = group_seed_maps.reshape(
        (n_phases, n_rois) + group_seed_maps.shape[1:]
    )

    return {
        "correlation_matrices": group_cor_matrices,
        "seed_maps": group_seed_maps,
        "sessions": treatment_sessions,
    }


def compute_group_level_pearson(
    subject_level_path: str | Path,
    session_treatment: str,
    session_control: str,
    brain_mask: npt.NDArray,
    fdr_threshold: float = 0.05,
    n_jobs: int = -1,
):
    """Compute group-level Pearson correlation matrices and seed-based maps.

    Group-level results are group-average correlation matrices and seed-based maps and
    their corresponding differences and statistical significance relative to the
    control, for each phase.

    Parameters
    ----------
    subject_level_path : str or pathlib.Path
        Path of the HDF5 file containing subject-level results.
    session_treatment : str
        Label of the treatment session.
    session_control : str
        Label of the control session.
    brain_mask : numpy.ndarray
        Boolean mask of the brain. Seed-based maps will be compared only where
        `brain_mask` is ``True``.
    fdr_threshold : float, optional
        Threshold at which to control the FDR. Default is 0.05.
    n_jobs : int, optional
        The maximum number of concurrently running jobs. If -1 all CPUs are used.
        Default is -1.

    Returns
    -------
    dict
        Dictionary with keys:

        * ``correlation_matrices``: group-level correlation matrices with shape
          ``(phases, 3, rois, rois)``. The second dimension corresponds to the mean
          correlation matrix, the difference matrix between session and control, and the
          significance matrix.
        * ``seed_maps``: group-level seed-based maps with shape ``(phases, rois, 3, x,
          y)``. The third dimension corresponds to the mean seed-based map, the
          difference map between session and control, and the significance map.
    """
    results_treatment = read_session_subject_level_pearson(
        subject_level_path, session_treatment
    )
    results_control = read_session_subject_level_pearson(
        subject_level_path, session_control
    )

    cor_matrices_treatment = np.asarray(
        list(results_treatment["correlation_matrices"].values())
    )
    cor_matrices_control = np.asarray(
        list(results_control["correlation_matrices"].values())
    )
    seed_maps_treatment = np.asarray(list(results_treatment["seed_maps"].values()))
    seed_maps_control = np.asarray(list(results_control["seed_maps"].values()))

    n_phases = cor_matrices_treatment.shape[1]
    n_rois = cor_matrices_treatment.shape[-1]
    with Parallel(n_jobs=n_jobs) as parallel:

        def compare_phase_cor_matrices(phase):
            return compare_correlation_matrices(
                cor_matrices_treatment[:, phase],
                cor_matrices_control[:, phase],
                fdr_threshold=fdr_threshold,
            )

        group_cor_matrices = parallel(
            delayed(compare_phase_cor_matrices)(phase) for phase in range(n_phases)
        )

        def compare_phase_roi_seed_maps(phase, roi):
            return compare_seed_maps(
                seed_maps_treatment[:, phase, roi],
                seed_maps_control[:, phase, roi],
                mask=brain_mask,
                fdr_threshold=fdr_threshold,
            )

        group_seed_maps = parallel(
            delayed(compare_phase_roi_seed_maps)(phase, roi)
            for phase in range(n_phases)
            for roi in range(n_rois)
        )

    group_cor_matrices = np.array(group_cor_matrices)
    group_seed_maps = np.array(group_seed_maps)
    group_seed_maps = group_seed_maps.reshape(
        (n_phases, n_rois) + group_seed_maps.shape[1:]
    )

    return {
        "correlation_matrices": group_cor_matrices,
        "seed_maps": group_seed_maps,
    }


def read_session_group_level_pearson(
    path: str | Path, session: str
) -> dict[str, npt.NDArray]:
    """Read group-level Pearson correlation results from a session.

    Group-level results are group-average correlation matrices and seed-based maps and
    their corresponding differences and statistical significance relative to the
    control, for each phase.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the output HDF5 file.
    session : str
        Session label to load from the results file.

    Returns
    -------
    dict
        Dictionary with keys:

        * ``correlation_matrices``: group-level correlation matrices with shape
          ``(phases, 3, rois, rois)``. The second dimension corresponds to the mean
          correlation matrix, the difference matrix between session and control, and the
          significance matrix.
        * ``seed_maps``: group-level seed-based maps with shape ``(phases, rois, 3, x,
          y)``. The third dimension corresponds to the mean seed-based map, the
          difference map between session and control, and the significance map.
    """
    with h5.File(path, "r") as f:
        return {
            "seed_maps": f[f"seed_maps/{session}"][...],
            "correlation_matrices": f[f"correlation_matrices/{session}"][...],
        }


def read_group_level_pearson_anova(path: str | Path) -> dict[str, npt.NDArray]:
    """Read group-level Pearson correlation results from an ANOVA analysis.

    Group-level results are group-average correlation matrices and seed-based maps and
    their corresponding differences and statistical significance relative to the
    control, for each phase.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the output HDF5 file.

    Returns
    -------
    dict
        Dictionary with keys:

        * ``correlation_matrices``: group-level correlation matrices with shape
          ``(phases, 3, rois, rois)``. The second dimension corresponds to the mean
          correlation matrix, the difference matrix between session and control, and the
          significance matrix.
        * ``seed_maps``: group-level seed-based maps with shape ``(phases, rois, 3, x,
          y)``. The third dimension corresponds to the mean seed-based map, the
          difference map between session and control, and the significance map.
        * ``sessions``: list of strings with the treatment session labels.
    """
    with h5.File(path, "r") as f:
        return {
            "seed_maps": f["seed_maps"][...],
            "correlation_matrices": f["correlation_matrices"][...],
            "sessions": [x.decode() for x in f["sessions"]],
        }


def write_session_group_level_pearson(
    path: str | Path,
    session: str,
    results: dict[str, npt.NDArray],
    overwrite: bool = False,
) -> None:
    """Write group-level Pearson correlation results from a session.

    Group-level results are group-average correlation matrices and seed-based maps and
    their corresponding differences and statistical significance relative to the
    control, for each phase.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the output HDF5 file.
    session : str
        Session label.
    results : dict
        Dictionary with keys:

        * ``correlation_matrices``: group-level correlation matrices with shape
          ``(phases, 3, rois, rois)``. The second dimension corresponds to the mean
          correlation matrix, the difference matrix between session and control, and the
          significance matrix.
        * ``seed_maps``: group-level seed-based maps with shape ``(phases, rois, 3, x,
          y)``. The third dimension corresponds to the mean seed-based map, the
          difference map between session and control, and the significance map.
    overwrite : bool, optional
        If ``True``, overwrite existing session data. If ``False`` and session data
        already exists, skip writing. Default is ``False``.

    Raises
    ------
    ValueError
        If the dimensions of the ``correlation_matrices`` and ``seed_maps`` arrays don't
        match, i.e.:

        * if the number of phases do not match;
        * if the second dimension of ``correlation_matrices`` and the third dimension of
          ``seed_maps`` are not size 3;
        * if the last two dimensions of ``correlation_matrices`` and the second
          dimension of ``seed_maps`` do not match.
    """
    if results["correlation_matrices"].shape[0] != results["seed_maps"].shape[0]:
        raise ValueError(
            "The first dimensions of 'correlation_matrices' and 'seed_maps' should "
            "be equal correspond to the number of phases."
        )

    if (
        results["correlation_matrices"].shape[1] != 3
        or results["seed_maps"].shape[2] != 3
    ):
        raise ValueError(
            "The second dimension of 'correlation_matrices' and the third dimension of "
            "'seed_maps' should be equal to 3 and correspond to the group-average, "
            "difference, and statistical significance."
        )

    if any(
        x != results["seed_maps"].shape[1]
        for x in results["correlation_matrices"].shape[-2:]
    ):
        raise ValueError(
            "The last two dimensions of 'correlation_matrices' and the second "
            "dimension of 'seed_maps' should be equal and correspond to the number of "
            "ROIs."
        )

    with h5.File(path, "a") as f:
        # Check if session already exists
        if "seed_maps" in f and session in f["seed_maps"]:
            if not overwrite:
                return
            # Delete existing session data to overwrite
            del f[f"seed_maps/{session}"]
            del f[f"correlation_matrices/{session}"]

        f.create_dataset(
            f"correlation_matrices/{session}", data=results["correlation_matrices"]
        )
        f.create_dataset(f"seed_maps/{session}", data=results["seed_maps"])


def write_group_level_pearson_anova(
    path: str | Path, results: dict[str, npt.NDArray], overwrite: bool = False
) -> None:
    """Write group-level Pearson correlation results from an ANOVA analysis.

    Group-level results are group-average correlation matrices and seed-based maps and
    their corresponding differences and statistical significance relative to the
    control, for each treatment session and phase.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the output HDF5 file.
    results : dict
        Dictionary with keys:

        * ``correlation_matrices``: group-level correlation matrices with shape
          ``(phases, 3, treatment_sessions, rois, rois)``. The second dimension
          corresponds to the mean correlation matrix, the difference matrix between
          session and control, and the significance matrix. The third dimension
            corresponds to the treatment sessions.
        * ``seed_maps``: group-level seed-based maps with shape ``(phases, rois, 3,
            treatment_sessions, x, y)``. The third dimension corresponds to the mean
            seed-based map, the difference map between session and control, and the
            significance map. The fourth dimension corresponds to the treatment
            sessions.
        * ``sessions``: list of strings with the treatment session labels.
    overwrite : bool, optional
        If ``True``, overwrite existing data. If ``False`` and data already exists,
        skip writing. Default is ``False``.

    Raises
    ------
    ValueError
        If the dimensions of the ``correlation_matrices`` and ``seed_maps`` arrays don't
        match, i.e.:

        * if the number of phases do not match;
        * if the second dimension of ``correlation_matrices`` and the third dimension of
          ``seed_maps`` are not size 3;
        * if the last two dimensions of ``correlation_matrices`` and the second
          dimension of ``seed_maps`` do not match.
    """
    if results["correlation_matrices"].shape[0] != results["seed_maps"].shape[0]:
        raise ValueError(
            "The first dimensions of 'correlation_matrices' and 'seed_maps' should "
            "be equal correspond to the number of phases."
        )

    if (
        results["correlation_matrices"].shape[1] != 3
        or results["seed_maps"].shape[2] != 3
    ):
        raise ValueError(
            "The second dimension of 'correlation_matrices' and the third dimension of "
            "'seed_maps' should be equal to 3 and correspond to the group-average, "
            "difference, and statistical significance."
        )

    if any(
        x != results["seed_maps"].shape[1]
        for x in results["correlation_matrices"].shape[-2:]
    ):
        raise ValueError(
            "The last two dimensions of 'correlation_matrices' and the second "
            "dimension of 'seed_maps' should be equal and correspond to the number of "
            "ROIs."
        )

    n_sessions = len(results["sessions"])
    if (
        results["correlation_matrices"].shape[2] != n_sessions
        or results["seed_maps"].shape[3] != n_sessions
    ):
        raise ValueError(
            "The third dimension of 'correlation_matrices' and the fourth dimension of "
            "'seed_maps' should be equal and correspond to the number of treatment "
            "sessions. The number of sessions should also match the length of "
            "'sessions'."
        )

    with h5.File(path, "a") as f:
        # Check if data already exists
        if "seed_maps" in f and "correlation_matrices" in f and "sessions" in f:
            if not overwrite:
                return
            # Delete existing data to overwrite
            del f["seed_maps"]
            del f["correlation_matrices"]
            del f["sessions"]

        f.create_dataset("correlation_matrices", data=results["correlation_matrices"])
        f.create_dataset("seed_maps", data=results["seed_maps"])
        f.create_dataset("sessions", data=results["sessions"], dtype=h5.string_dtype())
