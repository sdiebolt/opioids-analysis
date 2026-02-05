from collections import ChainMap
from operator import itemgetter
from pathlib import Path
from typing import cast

import h5py as h5
import numpy as np
from bids import BIDSLayout
from joblib import Parallel, delayed
from nibabel.nifti1 import Nifti1Image
from rich.console import Console
from rich.progress import track

from opioids_analysis.pearson import (
    SessionSubjectLevelPearson,
    SubjectLevelPearson,
    compute_group_level_pearson,
    compute_group_level_pearson_anova,
    compute_subject_level_pearson,
    read_session_sample_masks,
    read_session_subject_level_pearson,
    write_group_level_pearson_anova,
    write_session_group_level_pearson,
    write_session_subject_level_pearson,
)
from opioids_analysis.plotting import (
    plot_group_level_pearson,
    plot_group_level_pearson_anova,
)

console = Console()

########################################################################################
# Parameters
########################################################################################

# Path to the preprocessed fUS-BIDS dataset.
registered_data_path = Path(
    "/mnt/feanor/datasets/opioids/derivatives/registration/derivatives/preprocessed/"
)
layout = BIDSLayout(registered_data_path, validate=False)

# Session labels present in the dataset.
sessions = layout.get_sessions()

# Path to the parameters directory.
params_path = Path("params/")

# Analysis results root path.
opioids_results_root = Path("/mnt/feanor/home/sdiebolt/opioids-paper-results/")

# Sample masks path.
sample_masks_path = opioids_results_root / "sample_masks.h5"

# Saline sessions used for the control group.
saline_sessions = ["saline", "saline2", "WTFS1", "WTMS1"]

# Label of the session used as control in statistical comparisons.
control_session = "salineControl"

# Labels of the treatment sessions used in the analysis.
treatment_sessions = ["WTM10", "WTM20", "WTM30", "WTM70"]

# Threshold for significance after FDR correction.
fdr_threshold = 0.05

# Output HDF5 file for subject-level and group-level results.
subject_level_path = opioids_results_root / "subject_level_pearson.h5"
group_level_path = opioids_results_root / "group_level_pearson.h5"
group_level_anova_path = opioids_results_root / "group_level_pearson_anova.h5"

# Folder where figures will be saved.
figures_path = group_level_path.parent / "figures"

# ROI ordering and labels for circular graphs
graph_roi_order = (8, 9, 5, 4, 3, 2, 1, 0, 6, 7, 10, 11, 12, 16, 17, 13, 14, 15)
graph_roi_labels = {k: v + 1 for k, v in enumerate(graph_roi_order)}

# The maximum number of concurrently running jobs. If -1 all CPUs are used.
n_jobs = -1

# Whether to overwrite existing results in HDF5 files.
overwrite_results = False

########################################################################################
# Initializations
########################################################################################

with console.status("[bold cyan]Loading template and ROIs..."):
    template_path = params_path / "opioids_template.nii.gz"
    template_img = Nifti1Image.from_filename(template_path)

    # Set the sform code to 0 so that Nilearn uses qform.
    template_img.set_sform(None, code=0)

    rois_name = params_path / "Mask-autoROIs18-slim.nii.gz"
    rois_img = Nifti1Image.from_filename(rois_name)

    if template_img.shape != rois_img.shape:
        raise ValueError("Template and ROIs image shapes do not match!")

    brain_mask_img = Nifti1Image(
        (template_img.get_fdata() > 0).astype(int),
        affine=template_img.affine,
        header=template_img.header,
    )

########################################################################################
# Subject-level Pearson correlation computation
########################################################################################

with Parallel(n_jobs=n_jobs) as parallel:
    for session in track(
        sessions,
        description="[bold cyan]Computing subject-level Pearson correlation",
        transient=True,
    ):
        if subject_level_path.is_file():
            with h5.File(subject_level_path, "r") as f:
                if "seed_maps" in f and session in f["seed_maps"]:
                    continue

        sample_masks = read_session_sample_masks(sample_masks_path, session)

        # Paths are sorted to order them by run index.
        subjects = cast(list[str], layout.get_subjects(session=session))

        subject_level_pearson = cast(
            list[SubjectLevelPearson],
            parallel(
                delayed(compute_subject_level_pearson)(
                    nii_paths=sorted(
                        layout.get(subject=subject, session=session, return_type="file")
                    ),
                    brain_mask_img=brain_mask_img,
                    rois_img=rois_img,
                    sample_masks=sample_masks[subject],
                )
                for subject in subjects
            ),
        )

        # Matrices and maps are saved as numpy arrays for easier operations during the
        # group-level analysis.
        correlation_matrices = np.array(
            [res["correlation_matrices"] for res in subject_level_pearson]
        )
        seed_maps = [res["seed_maps"] for res in subject_level_pearson]
        subject_level_pearson = cast(
            SessionSubjectLevelPearson,
            {
                "correlation_matrices": dict(
                    zip(subjects, np.array(correlation_matrices))
                ),
                "seed_maps": dict(zip(subjects, np.array(seed_maps))),
            },
        )

        write_session_subject_level_pearson(
            subject_level_path,
            session,
            subject_level_pearson,
            overwrite=overwrite_results,
        )


########################################################################################
# Create the saline control session
########################################################################################

with console.status("[bold cyan]Creating saline control session..."):
    with h5.File(subject_level_path, "r") as f:
        missing_pearson = [s for s in saline_sessions if s not in f["seed_maps"]]

    if missing_pearson:
        raise RuntimeError(
            "The following saline pearson are missing from the HDF5 file: "
            f"{missing_pearson}."
        )

    saline_pearson = [
        read_session_subject_level_pearson(subject_level_path, s)
        for s in saline_sessions
    ]

    correlation_matrices = map(itemgetter("correlation_matrices"), saline_pearson)
    seed_maps = map(itemgetter("seed_maps"), saline_pearson)
    saline_control_pearson = cast(
        SessionSubjectLevelPearson,
        {
            "correlation_matrices": dict(ChainMap(*correlation_matrices)),
            "seed_maps": dict(ChainMap(*seed_maps)),
        },
    )

    write_session_subject_level_pearson(
        subject_level_path,
        control_session,
        saline_control_pearson,
        overwrite=overwrite_results,
    )

########################################################################################
# Group-level Pearson correlation computation (individual sessions)
########################################################################################

brain_mask = brain_mask_img.get_fdata().squeeze().astype(bool)

# Compute group-level results for each treatment session individually
for session in track(
    treatment_sessions,
    description="[bold cyan]Computing group-level Pearson correlation",
    transient=True,
):
    group_level_pearson = compute_group_level_pearson(
        subject_level_path=subject_level_path,
        session_treatment=session,
        session_control=control_session,
        brain_mask=brain_mask,
        fdr_threshold=fdr_threshold,
        n_jobs=n_jobs,
    )

    write_session_group_level_pearson(
        group_level_path, session, group_level_pearson, overwrite=overwrite_results
    )

########################################################################################
# Group-level Pearson correlation computation (ANOVA across sessions)
########################################################################################

with console.status("[bold cyan]Computing group-level Pearson ANOVA..."):
    group_level_pearson_anova = compute_group_level_pearson_anova(
        subject_level_path=subject_level_path,
        treatment_sessions=treatment_sessions,
        control_session=control_session,
        brain_mask=brain_mask,
        fdr_threshold=fdr_threshold,
        n_jobs=n_jobs,
    )

    write_group_level_pearson_anova(
        group_level_anova_path, group_level_pearson_anova, overwrite=overwrite_results
    )

########################################################################################
# Plotting group-level Pearson correlation results (individual sessions)
########################################################################################

for session in track(
    treatment_sessions,
    description="[bold cyan]Plotting group-level Pearson correlation results",
    transient=True,
):
    plot_group_level_pearson(
        group_level_path=group_level_path,
        session=session,
        template_img=template_img,
        rois_img=rois_img,
        graph_roi_order=graph_roi_order,
        graph_roi_labels=graph_roi_labels,
        output_path=figures_path,
    )

########################################################################################
# Plotting group-level Pearson ANOVA results (across sessions)
########################################################################################

with console.status("[bold cyan]Plotting group-level Pearson ANOVA results..."):
    plot_group_level_pearson_anova(
        group_level_path=group_level_anova_path,
        template_img=template_img,
        rois_img=rois_img,
        graph_roi_order=graph_roi_order,
        graph_roi_labels=graph_roi_labels,
        output_path=figures_path,
    )
