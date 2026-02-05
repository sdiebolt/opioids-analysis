import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bids import BIDSLayout
from joblib import Parallel, delayed
from nibabel.nifti1 import Nifti1Image
from nilearn.image import math_img
from nilearn.maskers import MultiNiftiLabelsMasker
from rich.console import Console
from rich.progress import track

from opioids_analysis.pearson import (
    read_session_sample_masks,
)

console = Console()

########################################################################################
# Parameters
########################################################################################

params_path = Path("params/")

opioids_dataset_root = Path("/mnt/feanor/datasets/opioids/")
opioids_results_root = Path("/mnt/feanor/home/sdiebolt/opioids-paper-results/")

# Sessions used to compute the relative CBV.
morphine_dose_response_sessions = ["WTM10", "WTM20", "WTM30", "WTM70"]
fentanyl_dose_response_sessions = ["WTF01", "WTF02", "WTF025", "WTF03"]

# Saline control sessions will be grouped into a single "salineControl" session.
saline_control_sessions = ["saline", "saline2", "WTFS1", "WTMS1"]

# Labels used for the sessions in the figures.
session_labels = {
    "salineControl": "Saline",
    "WTM10": "Morphine (10 mg/kg)",
    "WTM20": "Morphine (20 mg/kg)",
    "WTM30": "Morphine (30 mg/kg)",
    "WTM70": "Morphine (70 mg/kg)",
    "WTF01": "Fentanyl (0.1 mg/kg)",
    "WTF02": "Fentanyl (0.2 mg/kg)",
    "WTF025": "Fentanyl (0.25 mg/kg)",
    "WTF03": "Fentanyl (0.3 mg/kg)",
}

# Duration of the window used to compute the relative CBV is 10 minutes (provided in
# seconds).
window_duration = 10 * 60

# Duration of the baseline period is 20 minutes (provided in seconds).
baseline_duration = 20 * 60

# Duration of the total fUSI acquisitions is 80 minutes (provided in seconds).
total_duration = 80 * 60

# Sampling frequency of the power Doppler is 2 Hz (400 ms power Doppler integration +
# 100 ms delay).
power_doppler_sampling_frequency = 2.0

sample_masks_path = opioids_results_root / "sample_masks.h5"
subject_level_path = opioids_results_root / "subject_level_pearson.h5"
group_level_path = opioids_results_root / "group_level_pearson.h5"

tracking_root = opioids_dataset_root / "derivatives" / "tracking"

registered_root = opioids_dataset_root / "derivatives" / "registration" / "rawdata"

########################################################################################
# Initializations
########################################################################################

output_path = opioids_results_root / "multimodal"
output_path.mkdir(exist_ok=True, parents=True)

figures_path = Path("figures")
figures_path.mkdir(exist_ok=True, parents=True)

########################################################################################
# Relative CBV
########################################################################################

with console.status("[bold cyan]Computing relative CBV..."):
    registered_layout = BIDSLayout(registered_root, validate=False)

    qform = np.diag([0.11, 0.4, 0.1, 1])

    template_name = params_path / "opioids_template.nii.gz"
    template_img = Nifti1Image.from_filename(template_name)
    template_img.set_qform(qform, code=1)
    template_img.set_sform(qform, code=0)

    # Ignore warning about casting to int32.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        brain_mask_img = math_img("(img != 0).astype(int)", img=template_img)

    seed_masks_img = Nifti1Image.from_filename(
        params_path / "Mask-autoROIs18-slim.nii.gz"
    )
    seed_labels = [
        "Ctx-Lat (L)",
        "Ctx (L)",
        "Ctx-Med (L)",
        "Ctx-Med (R)",
        "Ctx (R)",
        "Ctx-Lat (R)",
        "Hip-CA1 (L)",
        "HIP-CA3 (L)",
        "HIP-CA3 (R)",
        "Hip-CA1 (R)",
        "DThal (L)",
        "LThal (L)",
        "VThal (L)",
        "VThal (R)",
        "LThal (R)",
        "DThal (R)",
        "HY (L)",
        "HY (R)",
    ]
    seed_masks_img.set_qform(qform, code=1)
    seed_masks_img.set_sform(qform, code=0)

    def get_subject_relative_cbv(paths, sample_mask):
        cbv = np.concatenate(
            MultiNiftiLabelsMasker(seed_masks_img).fit_transform(paths)
        )

        cbv[~sample_mask] = np.nan

        baseline_cbv = np.nanmin(
            cbv[: int(baseline_duration * power_doppler_sampling_frequency)], axis=0
        )

        # Ignore division by zero warnings when normalizing the cbv time series.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cbv = 100 * (cbv - baseline_cbv) / baseline_cbv

        cbv[np.isinf(cbv)] = 0

        return cbv

    with Parallel(n_jobs=-1) as parallel:
        rcbv_dfs = []
        # We store the number of subjects per session to use it as weights when
        # re-averaging the rCBV time series to create the salineControl session.
        n_subjects = {}
        for session in (
            saline_control_sessions
            + morphine_dose_response_sessions
            + fentanyl_dose_response_sessions
        ):
            subjects = registered_layout.get_subjects(session=session)
            n_subjects[session] = len(subjects)

            session_sample_masks = sample_mask = read_session_sample_masks(
                sample_masks_path, session
            )

            subjects_rcbvs = parallel(
                delayed(get_subject_relative_cbv)(
                    paths=sorted(
                        registered_layout.get(
                            subject=subject,
                            session=session,
                            extension=".nii",
                            return_type="file",
                        )
                    ),
                    sample_mask=session_sample_masks[subject].reshape((-1,)),
                )
                for subject in subjects
            )

            # Ingore mean of empty slices warnings when averaging the rCBV time series.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                seeds_rcbvs = np.nanmean(np.array(subjects_rcbvs), axis=0)
            seed_rcbv_df = pd.DataFrame(
                seeds_rcbvs,
                index=pd.to_timedelta(
                    np.arange(seeds_rcbvs.shape[0]) / power_doppler_sampling_frequency,
                    unit="s",
                ),
                columns=seed_labels,  # type: ignore
            ).interpolate()
            seed_rcbv_df["session"] = session
            rcbv_dfs.append(seed_rcbv_df)

    rcbv_df = pd.concat(rcbv_dfs)

    # Re-average the saline sessions into a single salineControl session, weighting by
    # the number of subjects in each session.
    n_subjects["salineControl"] = sum(n_subjects[s] for s in saline_control_sessions)
    rcbv_df = (
        rcbv_df.groupby("session")
        .apply(lambda x: x.mul(n_subjects[x.name], axis=0), include_groups=False)
        .reset_index(level="session")
    )
    rcbv_df["session"] = rcbv_df["session"].replace(
        saline_control_sessions, "salineControl"
    )
    rcbv_df = (
        rcbv_df.reset_index(names="time")
        .groupby(["time", "session"], as_index=False)
        .sum()
        .groupby("session")
        .apply(lambda x: x / n_subjects[x.name])
        .reset_index("session")
        .set_index("time")
    )

    rcbv_df.to_csv(output_path / "seed_cbv.csv")

########################################################################################
# Relative CBV
########################################################################################

for session in track(
    session_labels.keys(),
    description="[bold cyan]Plotting rCBV rasterplots",
    transient=True,
):
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        rcbv_df.query("session == @session").drop(columns="session").T,
        vmax=60,
        vmin=-60,
        cmap="coolwarm",
        cbar_kws={"label": "Relative CBV (%)"},
        ax=ax,
    )
    ax.axvline(
        baseline_duration * power_doppler_sampling_frequency,
        color="black",
        linestyle="--",
    )
    ax.set_xticks(
        np.arange(
            0,
            total_duration * power_doppler_sampling_frequency + 1,
            window_duration * power_doppler_sampling_frequency,
        )
    )
    ax.set_xlabel("")
    fig.savefig(
        figures_path / f"rcbv_rasterplot_{session}.tiff",
        bbox_inches="tight",
        dpi=300,
    )
