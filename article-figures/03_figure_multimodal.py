import itertools
import warnings
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bids import BIDSLayout
from joblib import Parallel, delayed
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from nibabel.nifti1 import Nifti1Image
from nilearn.image import math_img
from nilearn.masking import apply_mask
from rich.console import Console
from rich.progress import track
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from opioids_analysis.multimodal import (
    compute_max_velocities,
    compute_moving_time_percentage,
    load_instant_velocity,
)
from opioids_analysis.pearson import (
    _matrix2tril,
    _tril2matrix,
    read_session_group_level_pearson,
    read_session_sample_masks,
    read_session_subject_level_pearson,
)

console = Console()

########################################################################################
# Parameters
########################################################################################

params_path = Path("params/")

opioids_dataset_root = Path("/mnt/feanor/datasets/opioids/")
opioids_results_root = Path("/mnt/feanor/home/sdiebolt/opioids-paper-results/")

# The morphine dose response sessions are used for Fig. 6 A.
morphine_dose_response_sessions = ["WTM10", "WTM20", "WTM30", "WTM70"]

# The first tracking session must be the saline control session.
tracking_sessions = ["WTSC11"] + morphine_dose_response_sessions

# Threshold for the velocity to consider the animal is moving, in cm/s.
velocity_threshold = 5.0

# Saline control sessions will be grouped into a single "salineControl" session.
saline_control_sessions = ["saline", "saline2", "WTFS1", "WTMS1"]

# Labels used for the sessions in the figures.
session_labels = {
    "salineControl": "Saline",
    "WTM10": "Morphine (10 mg/kg)",
    "WTM20": "Morphine (20 mg/kg)",
    "WTM30": "Morphine (30 mg/kg)",
    "WTM70": "Morphine (70 mg/kg)",
    "BALBcM30": "BALB/cJ Morphine (30 mg/kg)",
}

# Labels used for the time series in the figures.
series_labels = {
    "fc_index": "FC index",
    "mobility": "Mobility",
    "rcbv": "rCBV",
    "analgesia": "Analgesia",
    "phosphorylation": "Phosphorylation",
    "respiratory_rate": "Respiratory rate",
}

# Reference session for the functional connectivity index.
fc_index_reference_session = "WTM70"

# Duration of the FC estimation phases is 10 minutes (provided in seconds).
fc_window_duration = 10 * 60

# Mobility window duration is 5 minutes (provided in seconds).
mobility_window_duration = 5 * 60

# Max velocity window duration is 20 minutes (provided in seconds).
max_velocity_window_duration = 20 * 60

# Resampling frequencies for the time series, provided as formulas compatible
# with pandas.resample.
fc_resampling_frequency = "5min"
rcbv_resampling_frequency = "5min"
analgesia_resampling_frequency = "5min"
phosphorylation_resampling_frequency = "5min"
respiratory_rate_resampling_frequency = "5min"

# Duration of the baseline period is 20 minutes (provided in seconds).
baseline_duration = 20 * 60

# Duration of the total fUSI acquisitions is 80 minutes (provided in seconds).
total_duration = 80 * 60

# Sampling frequency of the power Doppler is 2 Hz (400 ms power Doppler integration +
# 100 ms delay).
power_doppler_sampling_frequency = 2.0

# The sampling frequency of the respiratory data is 0.5 Hz, as indicated in the
# Excel file.
respiratory_sampling_frequency = 0.5

# Physiological time series colors for the plots.
series_colors = {
    "fc_index": "tab:blue",
    "mobility": "tab:orange",
    "rcbv": "tab:green",
    "analgesia": "tab:red",
    "phosphorylation": "tab:purple",
    "respiratory_rate": "tab:cyan",
}

# Colors for the treatments in the time series plots and correlation matrix.
treatments_colors = {
    "salineControl": "#696969",  # Saline (dark gray - control)
    "WTM10": "#26C6DA",  # Morphine 10mg/kg (cyan)
    "WTM20": "#66BB6A",  # Morphine 20mg/kg (light green)
    "WTM30": "#FFA726",  # Morphine 30mg/kg (amber)
    "WTM70": "#EF5350",  # Morphine 70mg/kg (coral red)
}

# We create a custom blue-white-red colormap for the correlation matrix using the
# blue/red colors from the "husl" palette, to be consistent with the treatment colors.
correlation_cmap = LinearSegmentedColormap.from_list(
    "div_cmap", [treatments_colors["WTM10"], "white", treatments_colors["WTM70"]], N=256
)

# Whether to overwrite existing CSV files.
overwrite_csvs = False

sample_masks_path = opioids_results_root / "sample_masks.h5"
subject_level_path = opioids_results_root / "subject_level_pearson.h5"
group_level_path = opioids_results_root / "group_level_pearson.h5"

tracking_root = opioids_dataset_root / "derivatives" / "tracking"

registered_root = opioids_dataset_root / "derivatives" / "registration" / "rawdata"

phosphorylation_xlsx_path = (
    opioids_dataset_root / "derivatives" / "phosphorylation_data.xlsx"
)

analgesia_xlsx_path = opioids_dataset_root / "derivatives" / "hot_plate_data.xlsx"

respiratory_xlsx_path = opioids_dataset_root / "derivatives" / "respiratory_data.xlsx"

########################################################################################
# Initializations
########################################################################################

output_path = opioids_results_root / "multimodal"
output_path.mkdir(exist_ok=True, parents=True)

figures_path = Path("figures")
figures_path.mkdir(exist_ok=True, parents=True)

########################################################################################
# Functional connectivity index
########################################################################################

fc_index_path = output_path / "fc_index.csv"
with console.status("[bold cyan]Computing functional connectivity index..."):
    if fc_index_path.exists() and not overwrite_csvs:
        console.log(f"[bold green]Loading FC index from {fc_index_path}...")
        fc_index_df = pd.read_csv(fc_index_path, index_col=0)
        fc_index_df.index = pd.to_timedelta(fc_index_df.index)
    else:
        group_level_results = read_session_group_level_pearson(
            group_level_path, fc_index_reference_session
        )

        # Compute the SVD of the group-level morphine 70 mg/kg matrices.
        group_level_matrices = group_level_results["correlation_matrices"][:, 0]
        group_level_lower_triangle = _matrix2tril(group_level_matrices)

        u, s, vh = np.linalg.svd(group_level_lower_triangle, full_matrices=False)

        # Compute FC index across all sessions.
        mode_ref = _tril2matrix(vh[1])
        fc_index_dfs = []
        for session in saline_control_sessions + morphine_dose_response_sessions:
            subject_level_results = read_session_subject_level_pearson(
                subject_level_path, session
            )
            subject_level_matrices = subject_level_results["correlation_matrices"]

            fc_index = np.array(
                [
                    [
                        np.corrcoef(_matrix2tril(phase_matrix), _matrix2tril(mode_ref))[
                            0, 1
                        ]
                        for phase_matrix in phase_matrices
                    ]
                    for phase_matrices in subject_level_matrices.values()
                ]
            ).T

            fc_index_df = pd.DataFrame(
                fc_index,
                index=pd.to_timedelta(
                    np.arange(1, fc_index.shape[0] + 1) * fc_window_duration, unit="s"
                ),
                columns=subject_level_matrices.keys(),  # type: ignore
            )
            fc_index_df -= fc_index_df.loc[
                : pd.Timedelta(baseline_duration, unit="s")
            ].mean(axis=0)
            fc_index_df = (
                fc_index_df.resample(fc_resampling_frequency).mean().interpolate()
            )
            fc_index_df = fc_index_df.melt(
                var_name="subject", value_name="fc_index", ignore_index=False
            )
            fc_index_df["session"] = session
            fc_index_dfs.append(fc_index_df)

        fc_index_df = pd.concat(fc_index_dfs)
        fc_index_df["session"] = fc_index_df["session"].replace(
            saline_control_sessions, "salineControl"
        )
        fc_index_df.to_csv(output_path / "fc_index.csv")

########################################################################################
# Animal mobility
########################################################################################

tracking_path = output_path / "mobility.csv"
max_velocity_path = output_path / "max_velocity.csv"
with console.status("[bold cyan]Computing animal mobility..."):
    if tracking_path.exists() and not overwrite_csvs:
        console.log(f"[bold green]Loading mobility data from {tracking_path}...")
        mobility_df = pd.read_csv(tracking_path, index_col=0)
        mobility_df.index = pd.to_timedelta(mobility_df.index)
        max_velocity_df = pd.read_csv(max_velocity_path, index_col=0)
    else:
        tracking_layout = BIDSLayout(tracking_root, validate=False)

        mobility_dfs = []
        max_velocity_dfs = []
        for session in tracking_sessions:
            subjects = tracking_layout.get_subjects(session=session)

            session_mobilities = []
            session_max_velocities = []
            valid_subjects = []
            for subject in subjects:
                tracking_paths = sorted(
                    tracking_layout.get(
                        subject=subject,
                        session=session,
                        extension=".h5",
                        return_type="file",
                    )
                )

                if len(tracking_paths) == 3:
                    speeds = []
                    for file in tracking_paths:
                        speeds.append(load_instant_velocity(file))
                    speeds = np.concatenate(speeds)

                    session_max_velocities.append(
                        compute_max_velocities(
                            speeds,
                            window_size=int(
                                max_velocity_window_duration
                                * power_doppler_sampling_frequency
                            ),
                        )
                    )
                    session_mobilities.append(
                        compute_moving_time_percentage(
                            speeds,
                            window_size=int(
                                mobility_window_duration
                                * power_doppler_sampling_frequency
                            ),
                            threshold=velocity_threshold,
                        )
                    )
                    valid_subjects.append(subject)

            session_mobilities = np.array(session_mobilities).T
            mobility_df = pd.DataFrame(
                session_mobilities,
                index=pd.to_timedelta(
                    np.arange(1, session_mobilities.shape[0] + 1)
                    * mobility_window_duration,
                    unit="s",
                ),
                columns=valid_subjects,  # type: ignore
            )
            mobility_df -= mobility_df.loc[
                : pd.Timedelta(baseline_duration, unit="s")
            ].mean(axis=0)
            mobility_df = mobility_df.melt(
                var_name="subject", value_name="mobility", ignore_index=False
            )
            mobility_df["session"] = session
            mobility_dfs.append(mobility_df)

            session_max_velocities = np.array(session_max_velocities).T
            max_velocity_df = pd.DataFrame(
                session_max_velocities,
                index=pd.to_timedelta(
                    np.arange(1, session_max_velocities.shape[0] + 1)
                    * max_velocity_window_duration,
                    unit="s",
                ),
                columns=valid_subjects,  # type: ignore
            )
            max_velocity_df = max_velocity_df.melt(
                var_name="subject",
                value_name="max_velocity",
                ignore_index=False,
            )
            max_velocity_df["session"] = session
            max_velocity_dfs.append(max_velocity_df)

        mobility_df = pd.concat(mobility_dfs)
        mobility_df["session"] = mobility_df["session"].replace(
            tracking_sessions[0], "salineControl"
        )
        mobility_df.to_csv(output_path / "mobility.csv")

        max_velocity_df = pd.concat(max_velocity_dfs)
        max_velocity_df["session"] = max_velocity_df["session"].replace(
            tracking_sessions[0], "salineControl"
        )
        max_velocity_df.to_csv(output_path / "max_velocity.csv")

########################################################################################
# Relative CBV
########################################################################################

cbv_path = output_path / "cbv.csv"
with console.status("[bold cyan]Computing relative CBV..."):
    if cbv_path.exists() and not overwrite_csvs:
        console.log(f"[bold green]Loading relative CBV data from {cbv_path}...")
        rcbv_df = pd.read_csv(cbv_path, index_col=0)
        rcbv_df.index = pd.to_timedelta(rcbv_df.index)
    else:
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
        seed_masks_img.set_qform(qform, code=1)
        seed_masks_img.set_sform(qform, code=0)

        # We binarize the seed masks image to use it in apply_mask.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            seed_masks_img = math_img("(img != 0).astype(int)", img=seed_masks_img)

        def get_subject_relative_cbv(paths, sample_mask):
            cbv = np.concatenate(
                [apply_mask(path, seed_masks_img) for path in paths], axis=0
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

            # Ignore "mean of empty slice" warnings when computing the mean.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return np.nanmean(cbv, axis=1)

        with Parallel(n_jobs=-1) as parallel:
            rcbv_dfs = []
            for session in saline_control_sessions + morphine_dose_response_sessions:
                subjects = registered_layout.get_subjects(session=session)

                session_sample_masks = sample_mask = read_session_sample_masks(
                    sample_masks_path, session
                )

                subjects_relative_cbvs = parallel(
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

                subjects_relative_cbvs = np.array(subjects_relative_cbvs).T
                rcbv_df = pd.DataFrame(
                    subjects_relative_cbvs,
                    index=pd.to_timedelta(
                        np.arange(subjects_relative_cbvs.shape[0])
                        / power_doppler_sampling_frequency,
                        unit="s",
                    ),
                    columns=subjects,
                ).interpolate()
                rcbv_df = (
                    rcbv_df.resample(
                        rcbv_resampling_frequency, offset=rcbv_resampling_frequency
                    )
                    .mean()
                    .melt(
                        var_name="subject",
                        value_name="rcbv",
                        ignore_index=False,
                    )
                )
                rcbv_df["session"] = session
                rcbv_dfs.append(rcbv_df)

        rcbv_df = pd.concat(rcbv_dfs)
        rcbv_df["session"] = rcbv_df["session"].replace(
            saline_control_sessions, "salineControl"
        )

        rcbv_df.to_csv(output_path / "cbv.csv")

########################################################################################
# Phosphorylation
########################################################################################

phosphorylation_csv_path = output_path / "phosphorylation.csv"
with console.status("[bold cyan]Loading phosphorylation data from Excel file..."):
    if phosphorylation_csv_path.exists() and not overwrite_csvs:
        console.log(
            "[bold green]Loading phosphorylation data from "
            f"{phosphorylation_csv_path}..."
        )
        phosphorylation_df = pd.read_csv(phosphorylation_csv_path, index_col=0)
        phosphorylation_df.index = pd.to_timedelta(phosphorylation_df.index)
    else:
        phosphorylation_df = pd.read_excel(
            phosphorylation_xlsx_path,
            skiprows=16,
            skipfooter=12,
            names=["time"]
            + [
                f"{s}_{i}" for s in morphine_dose_response_sessions for i in range(1, 3)
            ],
        ).drop(0)

        # Since the phosphorylation data is sampled only post-injection, we need to
        # assume the baseline phosphorylation level is 0 to be able to compare with the
        # fUSI data (rCBV, FC index, mobility).
        phosphorylation_df.loc[-2] = 0
        phosphorylation_df.loc[-2, "time"] = -20
        phosphorylation_df.loc[-1] = 0
        phosphorylation_df.loc[-1, "time"] = -10
        phosphorylation_df.index = phosphorylation_df.index + 2
        phosphorylation_df.sort_index(inplace=True)

        phosphorylation_df.set_index(
            pd.to_timedelta(phosphorylation_df.time * 60 + baseline_duration, unit="s"),
            inplace=True,
        )
        phosphorylation_df.drop(columns=["time"], inplace=True)

        phosphorylation_df["salineControl"] = 0
        phosphorylation_df = (
            phosphorylation_df.resample(
                phosphorylation_resampling_frequency,
                offset=phosphorylation_resampling_frequency,
            )
            .mean()
            .interpolate()
        )
        # We keep only the first 80 minutes of data, to match the duration of the
        # fUSI data.
        phosphorylation_df = phosphorylation_df.loc[:"80min"]

        phosphorylation_df = phosphorylation_df.melt(
            var_name="session",
            value_name="phosphorylation",
            ignore_index=False,
        )

        phosphorylation_df["session"] = phosphorylation_df["session"].str.replace(
            r"_(\d)", "", regex=True
        )

        phosphorylation_df.to_csv(output_path / "phosphorylation.csv")

########################################################################################
# Analgesia
########################################################################################

analgesia_csv_path = output_path / "analgesia.csv"
with console.status("[bold cyan]Loading analgesia data from Excel file..."):
    if analgesia_csv_path.exists() and not overwrite_csvs:
        console.log(f"[bold green]Loading analgesia data from {analgesia_csv_path}...")
        analgesia_df = pd.read_csv(analgesia_csv_path, index_col=0)
        analgesia_df.index = pd.to_timedelta(analgesia_df.index)
    else:
        analgesia_df = pd.read_excel(
            analgesia_xlsx_path,
            skiprows=22,
        )
        analgesia_df.rename(columns={"Time (min)": "time"}, inplace=True)

        analgesia_df.index = pd.to_timedelta(
            analgesia_df.time * 60 + baseline_duration, unit="s"
        )
        analgesia_df.drop(columns=["time"], inplace=True)

        # Renaming columns and dropping null columns
        group_num = 0
        for col_index, col in enumerate(analgesia_df.columns):
            if cast(pd.Series, analgesia_df[col].notnull()).any():
                if analgesia_df.iloc[:, col_index - 1].isnull().any():
                    group_num += 1
                analgesia_df.rename(
                    columns={col: morphine_dose_response_sessions[group_num]},
                    inplace=True,
                )

        # Since the analgesia data is sampled only post-injection, we need to
        # assume the baseline analgesia level is 0 to be able to compare with the
        # fUSI data (rCBV, FC index, mobility).
        analgesia_df.loc[pd.Timedelta("0min")] = 0
        analgesia_df.loc[pd.Timedelta("10min")] = 0
        analgesia_df.sort_index(inplace=True)
        analgesia_df = (
            analgesia_df.resample(analgesia_resampling_frequency).mean().interpolate()
        )

        # We keep only the first 80 minutes of data, to match the duration of the
        # fUSI data.
        analgesia_df = analgesia_df.loc[: pd.Timedelta(total_duration, unit="s")]

        analgesia_df = analgesia_df.dropna(axis=1)
        analgesia_df["salineControl"] = 0
        analgesia_df = analgesia_df.melt(
            var_name="session",
            value_name="analgesia",
            ignore_index=False,
        )

        analgesia_df.to_csv(output_path / "analgesia.csv")

########################################################################################
# Respiratory rate
########################################################################################

respiratory_csv_path = output_path / "respiratory.csv"
with console.status("[bold cyan]Loading respiratory rate data from Excel file..."):
    if respiratory_csv_path.exists() and not overwrite_csvs:
        console.log(
            f"[bold green]Loading respiratory data from {respiratory_csv_path}..."
        )
        respiratory_rate_df = pd.read_csv(respiratory_csv_path, index_col=0)
        respiratory_rate_df.index = pd.to_timedelta(respiratory_rate_df.index)
    else:
        respiratory_morphine = pd.read_excel(
            respiratory_xlsx_path,
            sheet_name=1,
            skiprows=12,
        ).dropna(axis=1)
        respiratory_morphine.columns = [
            f"WTM{dose:02d}" for dose in (30, 20, 10, 5, 70) for _ in range(6)
        ]

        respiratory_control = pd.read_excel(
            respiratory_xlsx_path,
            sheet_name=0,
            skiprows=11,
        ).dropna(axis=1)
        respiratory_control.columns = ["WT"] * 6

        respiratory_rate_df = pd.concat(
            (respiratory_control, respiratory_morphine), axis=1
        )
        respiratory_rate_df.set_index(
            pd.to_timedelta(
                respiratory_rate_df.index / respiratory_sampling_frequency, unit="s"
            ),
            inplace=True,
        )

        # Interpolate outliers.
        respiratory_rate_df[respiratory_rate_df < 40] = np.nan
        respiratory_rate_df[respiratory_rate_df > 270] = np.nan
        respiratory_rate_df.interpolate(inplace=True)

        # Transform to BPM relative to control.
        # There are 6 repeats for each session.
        wt_respiratory_rate = np.tile(
            cast(pd.Series, respiratory_rate_df["WT"].mean()).to_numpy(), 6
        )
        respiratory_rate_df = cast(
            pd.DataFrame,
            100 * (respiratory_rate_df - wt_respiratory_rate) / wt_respiratory_rate,
        )

        respiratory_rate_df = respiratory_rate_df.resample(
            respiratory_rate_resampling_frequency,
            offset=respiratory_rate_resampling_frequency,
        ).mean()
        respiratory_rate_df.index += pd.Timedelta(baseline_duration, unit="s")
        baseline_timepoints = pd.timedelta_range(
            respiratory_rate_resampling_frequency,
            pd.Timedelta(baseline_duration, unit="s"),
            freq=respiratory_rate_resampling_frequency,
        )
        for t in baseline_timepoints:
            respiratory_rate_df.loc[t] = 0
        respiratory_rate_df = respiratory_rate_df.sort_index().melt(
            var_name="session",
            value_name="respiratory_rate",
            ignore_index=False,
        )
        respiratory_rate_df.replace({"WT": "salineControl"}, inplace=True)

        respiratory_rate_df.to_csv(output_path / "respiratory.csv")

########################################################################################
# Merging all the physiological data into a single DataFrame.
########################################################################################

with console.status("[bold cyan]Merging physiological time series..."):
    mean_fc_index_df = cast(
        pd.DataFrame,
        fc_index_df.reset_index(names="time")
        .drop(columns="subject")
        .groupby(["time", "session"], as_index=False)
        .mean(),
    )

    mean_mobility_df = cast(
        pd.DataFrame,
        mobility_df.reset_index(names="time")
        .drop(columns="subject")
        .groupby(["time", "session"], as_index=False)
        .mean(),
    )

    mean_rcbv_df = cast(
        pd.DataFrame,
        rcbv_df.reset_index(names="time")
        .drop(columns="subject")
        .groupby(["time", "session"], as_index=False)
        .mean(),
    )

    mean_analgesia_df = cast(
        pd.DataFrame,
        analgesia_df.reset_index().groupby(["time", "session"], as_index=False).mean(),
    )

    mean_phosphorylation_df = cast(
        pd.DataFrame,
        phosphorylation_df.reset_index()
        .groupby(["time", "session"], as_index=False)
        .mean(),
    )

    mean_respiratory_rate_df = cast(
        pd.DataFrame,
        respiratory_rate_df.reset_index(names="time")
        .groupby(["time", "session"], as_index=False)
        .mean(),
    )

    # DataFrames computed from fUSI acquisitions can be merged directly since they all
    # have the same time index.
    merged_physio_df = mean_fc_index_df.merge(mean_mobility_df).merge(mean_rcbv_df)

    # The analgesia and phosphorylation DataFrames have a different time index, so we
    # merge them using an outer join to keep all time points, then interpolate missing
    # values.
    merged_physio_df = (
        merged_physio_df.merge(mean_analgesia_df, how="outer")
        .merge(mean_phosphorylation_df, how="outer")
        .merge(mean_respiratory_rate_df, how="outer")
        .set_index("time")
        .groupby("session")
        .apply(lambda g: g.interpolate("index"), include_groups=False)
        .dropna()
        .reset_index()
    )

    # Reorder the columns for visualization purposes.
    merged_physio_df = cast(
        pd.DataFrame,
        merged_physio_df[
            [
                "time",
                "session",
                "fc_index",
                "rcbv",
                "mobility",
                "analgesia",
                "phosphorylation",
                "respiratory_rate",
            ]
        ],
    )

# Uncomment this to limit the time series to the first 50 minutes of data (to match the
# respiratory rate data duration).
# merged_physio_df = merged_physio_df[merged_physio_df.time <= pd.Timedelta("50min")]

########################################################################################
# Create normalized time series relative to the baseline minimum and the global maximum
########################################################################################


def scale_series(s, min, max):
    return (s - min) / max


with console.status("[bold cyan]Normalizing time series..."):
    fc_index_normalized_df = fc_index_df.copy()
    min = merged_physio_df.loc[merged_physio_df.time < "20min", "fc_index"].min()
    max = (merged_physio_df["fc_index"] - min).abs().max()
    fc_index_normalized_df["fc_index"] = scale_series(
        fc_index_normalized_df["fc_index"], min, max
    )

    mobility_normalized_df = mobility_df.copy()
    min = merged_physio_df.loc[merged_physio_df.time < "20min", "mobility"].min()
    max = (merged_physio_df["mobility"] - min).abs().max()
    mobility_normalized_df["mobility"] = scale_series(
        mobility_normalized_df["mobility"], min, max
    )

    rcbv_normalized_df = rcbv_df.copy()
    min = merged_physio_df.loc[merged_physio_df.time < "20min", "rcbv"].min()
    max = (merged_physio_df["rcbv"] - min).abs().max()
    rcbv_normalized_df["rcbv"] = scale_series(rcbv_normalized_df["rcbv"], min, max)

    analgesia_normalized_df = analgesia_df.copy()
    min = merged_physio_df.loc[merged_physio_df.time < "20min", "analgesia"].min()
    max = (merged_physio_df["analgesia"] - min).abs().max()
    analgesia_normalized_df["analgesia"] = scale_series(
        analgesia_normalized_df["analgesia"], min, max
    )

    phosphorylation_normalized_df = phosphorylation_df.copy()
    min = merged_physio_df.loc[merged_physio_df.time < "20min", "phosphorylation"].min()
    max = (merged_physio_df["phosphorylation"] - min).abs().max()
    phosphorylation_normalized_df["phosphorylation"] = scale_series(
        phosphorylation_normalized_df["phosphorylation"], min, max
    )

    respiratory_rate_normalized_df = respiratory_rate_df.copy()
    min = merged_physio_df.loc[
        merged_physio_df.time < "20min", "respiratory_rate"
    ].min()
    max = (merged_physio_df["respiratory_rate"] - min).abs().max()
    respiratory_rate_normalized_df["respiratory_rate"] = scale_series(
        respiratory_rate_normalized_df["respiratory_rate"], min, max
    )

########################################################################################
# Plotting normalized multimodal time series
########################################################################################

with console.status("[bold cyan]Plotting normalized multimodal time series..."):
    fig, axes = plt.subplots(2, 2, figsize=(18, 6), sharex=True, sharey=True)
    for ax, session in zip(axes.ravel(), morphine_dose_response_sessions):
        for y in merged_physio_df.columns[2:]:
            df = eval(f"{y}_normalized_df")
            sns.lineplot(
                df.query("session == @session"),
                x=df.query("session == @session").index,
                y=y,
                style="session",
                markers=True,
                c=series_colors[y],
                ax=ax,
                legend=False,
            )

        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax.axvline(baseline_duration * 1e9, color="black", linestyle="--")
        ax.text(
            baseline_duration * 1e9 - 0.1e12,
            -1.1,
            "Injection",
            color="black",
            fontsize=14,
            ha="right",
            va="bottom",
            rotation=90,
        )

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xlim(0, total_duration * 1e9)
        ax.set_ylim(-1.2, 1.4)

        ax.set_xticks(
            np.arange(0, total_duration * 1e9 + 1, fc_window_duration * 1e9),
            labels=[
                f"{i * fc_window_duration // 60} min"
                for i in range(total_duration // fc_window_duration + 1)
            ],
            rotation=45,
            ha="right",
        )
        ax.set_yticks([-1, 0, 1], labels=["-100%", "0%", "100%"])

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        n_subjects = len(
            fc_index_normalized_df.query("session == @session").subject.unique()
        )
        ax.set_title(
            f"{session_labels[session]} (n={n_subjects})",
            fontsize=16,
            fontweight="bold",
        )

    axes[0, 1].legend(
        handles=[
            Line2D([], [], color=series_colors[series_name], label=series_label)
            for series_name, series_label in series_labels.items()
        ],
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
    )
    fig.savefig(
        figures_path / "multimodal_timeseries.tiff", bbox_inches="tight", dpi=300
    )
    plt.close(fig)

########################################################################################
# Plotting swarm plots of the different time series at specific time points
########################################################################################

specific_time_points = list(map(pd.Timedelta, ["20min", "30min", "40min", "50min"]))
time_point_labels = ["BP2", "SP1", "SP2", "SP3"]

specific_time_points_analgesia_df = analgesia_df.loc[specific_time_points]
specific_time_points_fc_index_df = fc_index_df.drop(columns="subject").loc[
    specific_time_points
]
specific_time_points_respiratory_rate_df = respiratory_rate_df.loc[specific_time_points]
specific_time_points_mobility_df = mobility_df.drop(columns="subject").loc[
    specific_time_points
]
specific_time_points_rcbv_df = rcbv_df.drop(columns="subject").loc[specific_time_points]

for session in track(
    specific_time_points_fc_index_df["session"].unique(),
    description="[bold cyan]Plotting swarm plots...",
    transient=True,
):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3), tight_layout=True)

    for series_col, ax in zip(
        ["analgesia", "fc_index", "respiratory_rate", "rcbv"], axes.ravel()
    ):
        df = globals()[f"specific_time_points_{series_col}_df"].query(
            "session == @session"
        )
        sns.swarmplot(
            df,
            x=df.index,
            y=series_col,
            c=series_colors[series_col],
            size=5,
            ax=ax,
        )

        mean_df = (
            df.reset_index(names="time")
            .groupby(["time", "session"], as_index=False)
            .mean()
        )
        ax.plot(
            np.arange(len(mean_df)),
            mean_df[series_col],
            c="k",
            marker="o",
            linestyle="--",
        )

        ax.set_ylabel(series_labels[series_col], c=series_colors[series_col])
        ax.set_xlabel("")
        ax.set_xticks(np.arange(len(time_point_labels)))
        ax.set_xticklabels(time_point_labels)
        ax.spines["left"].set_color(series_colors[series_col])
        ax.tick_params(axis="y", colors=series_colors[series_col])

    fig.savefig(
        figures_path / f"swarm_plots_{session}.tiff", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

########################################################################################
# Plotting regression plots
########################################################################################

merged_physio_without_saline_df = merged_physio_df.query("session != 'salineControl'")

column_combinations = list(itertools.combinations(series_labels.keys(), 2))
for col1, col2 in track(
    column_combinations,
    description="[bold cyan]Plotting regression plots...",
    transient=True,
):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for session in merged_physio_without_saline_df["session"].unique():
        session_df = merged_physio_without_saline_df.query("session == @session")
        ax.scatter(
            session_df[col1],
            session_df[col2],
            alpha=np.arange(1, len(session_df) + 1) / len(session_df),
            c=treatments_colors[session],
        )

    sns.regplot(
        merged_physio_without_saline_df,
        x=col1,
        y=col2,
        scatter=False,
        line_kws=dict(color="k"),
        ax=ax,
    )
    ax.set_xlabel(series_labels[col1])
    ax.set_ylabel(series_labels[col2])
    fig.savefig(
        figures_path / f"regression_{col1}_{col2}.tiff", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

########################################################################################
# Plotting multimodal correlation matrix
########################################################################################


def corrdot(*args, **kwargs):
    corr_r = args[0].corr(args[1], "pearson")
    corr_text = f"{corr_r:2.2f}".replace("0.", ".")
    ax = plt.gca()
    ax.set_axis_off()
    marker_size = abs(corr_r) * 10000
    ax.scatter(
        [0.5],
        [0.5],
        marker_size,
        [corr_r],
        alpha=0.6,
        cmap=correlation_cmap,
        edgecolor="none",
        vmin=-1,
        vmax=1,
        transform=ax.transAxes,
    )
    font_size = abs(corr_r) * 40 + 5
    ax.annotate(
        corr_text,
        (0.5, 0.5),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=font_size,
    )


with console.status("[bold cyan]Plotting multimodal correlation matrix..."):
    merged_physio_without_respiratory_rate_df = (
        merged_physio_df.query("session != 'salineControl'")
        .drop(columns=["respiratory_rate"])
        .rename(columns=series_labels)
    )

    sns.set_theme(style="white", font_scale=1.6)
    g = sns.PairGrid(
        merged_physio_without_respiratory_rate_df.drop(columns=["time"]),
        diag_sharey=False,
    )
    g.map_lower(sns.regplot, scatter=False, lowess=False, line_kws={"color": "black"})
    g.map_lower(
        sns.scatterplot,
        hue=merged_physio_without_respiratory_rate_df["session"],
        palette=treatments_colors,
        alpha=0.6,
        s=50,
    )
    g.map_diag(sns.kdeplot)
    g.map_upper(corrdot)

    # Add axes on the upper right corner to show a custom legend for the scatter plot in
    # the lower diagonal.
    legend_ax = g.figure.add_axes((0.95, 0.6, 0.05, 0.4))
    legend_ax.set_axis_off()
    legend_ax.legend(
        handles=[
            Line2D(
                [], [], marker="o", markersize=14, color="none", markerfacecolor=color
            )
            for color in treatments_colors.values()
        ],
        labels=[
            session_labels[session]
            for session in ["salineControl"] + morphine_dose_response_sessions
        ],
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        fontsize=12,
        title="Session",
        title_fontsize=14,
    )

    g.figure.savefig(
        figures_path / "multimodal_correlation_matrix.tiff",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(g.figure)


def calculate_pvalues(df):
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how="outer")
    for r in df.columns:
        for c in df.columns:
            tmp = df[df[r].notnull() & df[c].notnull()]
            pvalues.loc[r, c] = pearsonr(tmp[r], tmp[c])[1]
    return pvalues


with console.status("[bold cyan]Computing correlation matrix and p-values..."):
    correlation_matrix = merged_physio_without_respiratory_rate_df.drop(
        columns=["time", "session"]
    ).corr("pearson")
    pvalues = calculate_pvalues(
        merged_physio_without_respiratory_rate_df.drop(columns=["time", "session"])
    )

    correlation_matrix.to_csv(output_path / "multimodal_correlation_matrix.csv")
    pvalues.to_csv(output_path / "multimodal_correlation_matrix_pvalues.csv")


########################################################################################
# Fitting a PCA to the multimodal time series and plotting the projection on PC1/2
########################################################################################

with console.status("[bold cyan]Plotting PCA projection..."):
    pca = make_pipeline(StandardScaler(), PCA(n_components=3))

    session = "WTM70"
    reference_pca_data = (
        merged_physio_df.drop(columns="respiratory_rate")
        .query("session == @session")
        .drop(columns=["time", "session"])
    )

    pca.fit(reference_pca_data)

    transformed_data = pca.transform(
        merged_physio_df.drop(columns=["respiratory_rate", "time", "session"])
    )
    transformed_df = pd.DataFrame(
        transformed_data,
        columns=["PC1", "PC2", "PC3"],  # type: ignore
        index=merged_physio_df.index,
    )
    transformed_df = pd.concat(
        [merged_physio_df[["time", "session"]], transformed_df], axis=1
    )

    fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=300)
    for session in ["salineControl"] + morphine_dose_response_sessions:
        ax.scatter(
            transformed_df.query("session == @session")["PC1"],
            transformed_df.query("session == @session")["PC2"],
            alpha=np.arange(1, len(reference_pca_data) + 1) / len(reference_pca_data),
            label=session_labels[session],
            c=treatments_colors[session],
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    fig.savefig(figures_path / "projection.tiff", dpi=300, bbox_inches="tight")
    plt.close(fig)

########################################################################################
# Plot PC loadings
########################################################################################

with console.status("[bold cyan]Plotting PCA loadings..."):
    pca_model = cast(PCA, pca["pca"])
    loadings = pca_model.components_.T * np.sqrt(pca_model.explained_variance_)

    fig, ax = plt.subplots(1, 1, figsize=(7, 3), tight_layout=True)
    im = ax.imshow(loadings[:, :2], vmax=1, vmin=-1, cmap=correlation_cmap)
    ax.set_xticks(ticks=np.arange(2), labels=["PC1", "PC2"], rotation=90, ha="center")
    ax.set_yticks(
        ticks=np.arange(len(series_labels) - 1),
        labels=list(series_labels.values())[:-1],
    )
    fig.colorbar(im, ax=ax, label="Loading (a.u.)", ticks=[-1, 0, 1])

    fig.savefig(figures_path / "loadings.tiff", dpi=300, bbox_inches="tight")
    plt.close(fig)

########################################################################################
# Plot explained variance
########################################################################################

with console.status("[bold cyan]Plotting explained variance..."):
    rng = np.random.default_rng(seed=42)
    random_pca = make_pipeline(StandardScaler(), PCA(n_components=3))
    random_pca.fit(rng.normal(size=reference_pca_data.shape))
    random_variance_ratio = 100 * cast(PCA, random_pca["pca"]).explained_variance_ratio_

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    explained_variance_ratio = 100 * pca_model.explained_variance_ratio_

    ax.scatter(np.arange(3), explained_variance_ratio, label="Actual data")
    ax.plot(np.arange(3), explained_variance_ratio)
    ax.scatter(np.arange(3), random_variance_ratio, label="White noise")
    ax.plot(np.arange(3), random_variance_ratio)
    ax.set_xticks(ticks=np.arange(3), labels=["PC1", "PC2", "PC3"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Explained variance ratio (%)")
    ax.axvline(1.5, color="red")
    ax.text(1.53, 30, "Marchenko-Pastur limit", color="red", rotation=90)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_title("Explained variance")
    ax.legend()

    fig.savefig(figures_path / "explained_variance.tiff", dpi=300, bbox_inches="tight")
    plt.close(fig)
