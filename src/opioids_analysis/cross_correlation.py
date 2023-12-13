"""Utilities for subject-level cross-correlation analysis."""

import re
from typing import cast, TypedDict

import h5py as h5
import nibabel as nib
import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed
from nilearn import masking
from nilearn.signal import _standardize
from pathlib import Path
from scipy.signal import correlate, correlation_lags

from opioids_analysis.pearson import compare_seed_maps


class SubjectLevelXCorrMaps(TypedDict):
    xcorr_maps: npt.NDArray


def compute_subject_level_xcorr(
    nii_paths: list[str | Path],
    brain_mask_img: nib.nifti1.Nifti1Image,
    rois_img: nib.nifti1.Nifti1Image,
    max_lag: int = 20,
) -> SubjectLevelXCorrMaps:
    """Compute subject-level cross-correlation maps.

    Parameters
    ----------
    nii_paths : list of str or list of pathlib.Path
        List of NIfTI files ordered by run index, corresponding to analysis phases of a
        session. `compute_subject_level_stats` will check that indices of the ``run``
        entities in `nii_paths` are consecutive and ordered. Otherwise, an error will be
        raised.
    brain_mask_img : nibabel.nifti1.Nifti1Image
        Brain mask with same spatial shape as the functional NIfTI images, used to
        extract voxel signals.
    rois_img : nibabel.nifti1.Nifti1Image
        ROIs mask with same spatial shape as the functional NIfTI images.
    max_lag : int, optional
        Maximum cross-correlation lag, in seconds. Default is 40.

    Returns
    -------
    dict
        Dictionary with one key:

        * ``xcorr_maps``: dictionary with subject names as keys and
          cross-correlation maps as values, with shape ``(phases, rois, lags, x, y)``.

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

    nii_img = cast(nib.nifti1.Nifti1Image, nib.load(nii_paths[0]))
    sampling_frequency = 1 / nii_img.header["pixdim"][4]
    phase_lags = (
        correlation_lags(nii_img.shape[-1], nii_img.shape[-1], mode="full")
        / sampling_frequency
    )
    retained_lags = (phase_lags >= -max_lag) & (phase_lags <= max_lag)

    xcorr_maps = np.zeros(
        (
            len(nii_paths),
            len(roi_labels),
            retained_lags.sum(),
            nii_img.shape[0],
            nii_img.shape[2],
        )
    )
    for phase_index, nii_path in enumerate(nii_paths):
        cleaned_signals = masking.apply_mask(nii_path, brain_mask_img)

        rois_signals = np.array(
            [
                cleaned_signals[:, rois_data == roi_label].mean(axis=1)
                for roi_label in roi_labels
            ]
        ).T

        cleaned_signals = _standardize(cleaned_signals, standardize="zscore_sample")
        rois_signals = _standardize(rois_signals, standardize="zscore_sample")

        for roi_index in range(len(roi_labels)):
            seed_to_voxel_xcorr = np.apply_along_axis(
                correlate,
                0,
                cleaned_signals,
                rois_signals[:, roi_index],
                mode="full",
            ) / (cleaned_signals.shape[0] - 1)

            seed_to_voxel_xcorr = seed_to_voxel_xcorr[retained_lags]
            xcorr_maps[phase_index, roi_index] = np.transpose(
                masking.unmask(seed_to_voxel_xcorr, brain_mask_img)
                .get_fdata()
                .squeeze(),
                axes=(2, 0, 1),
            )

    return {"xcorr_maps": xcorr_maps}


class SessionSubjectLevelXCorrMaps(TypedDict):
    xcorr_maps: dict[str, npt.NDArray]


def write_session_subject_level_xcorr(
    path: str | Path,
    session: str,
    results: SessionSubjectLevelXCorrMaps,
) -> None:
    """Write subject-level cross-correlation maps for all subjects of a session.

    Cross-correlation maps are saved to an HDF5 file as sub-datasets with names
    ``xcorr_maps/session/subject``, where ``session`` and ``subject`` are session and
    subject labels respectively. If `path` exists, datasets will be appended as new
    sub-datasets.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to an HDF5 file where the subject-level cross-correlation maps will be
        saved.
    session : str
        Session label.
    results : dict
        Dictionary with one key:

        * ``xcorr_maps``: dictionary with subject names as keys and
          cross-correlation maps as values, with shape ``(phases, rois, x, y, lags)``.
    """
    with h5.File(path, "a") as f:
        for subject in results["xcorr_maps"].keys():
            f.create_dataset(
                f"xcorr_maps/{session}/{subject}", data=results["xcorr_maps"][subject]
            )


def read_session_subject_level_xcorr(
    path: str | Path, session: str
) -> SessionSubjectLevelXCorrMaps:
    """Read subject-level cross-correlation maps for all subjects of a session.

    Cross-correlation maps are saved to an HDF5 file as sub-datasets with names
    ``xcorr_maps/session/subject``, where ``session`` and ``subject`` are session and
    subject labels respectively. If `path` exists, datasets will be appended as new
    sub-datasets. This file can be written using
    `write_session_subject_level_xcorr`.

    Parameters
    ----------
    path : str
        Path of the HDF5 file containing the subject-level cross-correlation maps.
    session : str
        Session label.

    Returns
    -------
    dict
        Dictionary with one key:

        * ``xcorr_maps``: dictionary with subject names as keys and
          cross-correlation maps as values, with shape ``(phases, rois, x, y, lags)``.
    """
    with h5.File(path, "r") as f:
        return {
            "xcorr_maps": {
                subject: maps[...]
                for subject, maps in f[f"xcorr_maps/{session}"].items()
            },
        }


def compute_group_level_xcorr(
    subject_level_path: str | Path,
    session_treatment: str,
    session_control: str,
    brain_mask: npt.NDArray,
    fdr_threshold: float = 0.05,
    n_jobs: int = -1,
):
    """Compute group-level cross-correlation maps.

    Group-level cross-correlation maps are group-average maps and their corresponding
    differences and statistical significance relative to the control, for each phase and
    each lag.

    Parameters
    ----------
    subject_level_path : str or pathlib.Path
        Path of the HDF5 file containing subject-level cross-correlation maps.
    session_treatment : str
        Label of the treatment session.
    session_control : str
        Label of the control session.
    brain_mask : numpy.ndarray
        Boolean mask of the brain. Cross-correlation maps will be compared only where
        `brain_mask` is ``True``.
    fdr_threshold : float, optional
        Threshold at which to control the FDR. Default is 0.05.
    n_jobs : int, optional
        The maximum number of concurrently running jobs. If -1 all CPUs are used.
        Default is -1.

    Returns
    -------
    dict
        Dictionary with one key:

        * ``xcorr_maps``: group-level correlation maps with shape ``(phases, rois, 3, x,
          y)`` dictionary with subject names as keys and
          cross-correlation maps as values, with shape ``(phases, rois, x, y, lags)``.
    """
    with h5.File(subject_level_path) as f:
        subjects_treatment = list(f["xcorr_maps"][session_treatment].keys())
        subjects_control = list(f["xcorr_maps"][session_control].keys())

        xcorr_maps_shape = f["xcorr_maps"][session_treatment][
            subjects_treatment[0]
        ].shape
        n_phases = xcorr_maps_shape[0]
        n_rois = xcorr_maps_shape[1]
        n_lags = xcorr_maps_shape[2]

    with Parallel(n_jobs=n_jobs) as parallel:

        def compare_phase_lag_xcorr(phase, roi, lag):
            with h5.File(subject_level_path) as f:
                seed_maps_treatment = np.array(
                    [
                        f["xcorr_maps"][session_treatment][subject][phase, roi, lag]
                        for subject in subjects_treatment
                    ]
                )
                seed_maps_control = np.array(
                    [
                        f["xcorr_maps"][session_control][subject][phase, roi, lag]
                        for subject in subjects_control
                    ]
                )
            return compare_seed_maps(
                seed_maps_treatment,
                seed_maps_control,
                mask=brain_mask,
                fdr_threshold=fdr_threshold,
            )

        group_xcorr_maps = parallel(
            delayed(compare_phase_lag_xcorr)(phase, roi, lag)
            for phase in range(n_phases)
            for roi in range(n_rois)
            for lag in range(n_lags)
        )

    group_xcorr_maps = np.array(group_xcorr_maps)
    group_xcorr_maps = group_xcorr_maps.reshape(
        (n_phases, n_rois, n_lags) + group_xcorr_maps.shape[1:]
    )

    return {
        "xcorr_maps": group_xcorr_maps,
    }


def read_session_group_level_xcorr(path: str | Path, session) -> dict[str, npt.NDArray]:
    """Read group-level cross-correlation maps from a session.

    Group-level cross-correlation maps are group-average maps and their corresponding
    differences and statistical significance relative to the control, for each phase and
    each lag.

    Parameters
    ----------
    path : str
        Path of the HDF5 file containing the group-level cross-correlation maps.
    session : str
        Session label.

    Returns
    -------
    dict
        Dictionary with one key:

        * ``xcorr_maps``: group-level cross-correlation maps with shape ``(phases, rois,
          lags, 3, x, y)``. The third dimension corresponds to the mean
          cross-correlation maps, the difference maps between session and control, and
          the significance maps.
    """
    with h5.File(path, "r") as f:
        return {"xcorr_maps": f[f"xcorr_maps/{session}"][...]}


def write_session_group_level_xcorr(
    path: str | Path, session: str, results: dict[str, npt.NDArray]
) -> None:
    """Write group-level cross-correlation maps from a session.

    Group-level cross-correlation maps are group-average maps and their corresponding
    differences and statistical significance relative to the control, for each phase and
    each lag.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to an HDF5 file where the group-level cross-correlation maps will be
        saved.
    session : str
        Session label.
    results : dict
        Dictionary with one key:

        * ``xcorr_maps``: group-level cross-correlation maps with shape ``(phases, rois,
          lags, 3, x, y)``. The third dimension corresponds to the mean
          cross-correlation maps, the difference maps between session and control, and
          the significance maps.
    """
    with h5.File(path, "a") as f:
        f.create_dataset(f"xcorr_maps/{session}", data=results["xcorr_maps"])
