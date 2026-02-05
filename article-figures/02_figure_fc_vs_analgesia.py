from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich.console import Console

from opioids_analysis.pearson import read_session_subject_level_pearson

console = Console()

########################################################################################
# Parameters
########################################################################################

params_path = Path("params/")

opioids_dataset_root = Path("/mnt/feanor/datasets/opioids/")
opioids_results_root = Path("/mnt/feanor/home/sdiebolt/opioids-paper-results/")

# Saline control sessions will be grouped into a single "salineControl" session.
saline_control_sessions = ["saline", "saline2", "WTFS1", "WTMS1"]

# Labels used for the sessions in the figures.
session_labels = {
    "salineControl": "Saline",
    "WTM30": "Morphine (30 mg/kg)",
    "WTF025": "Fentanyl (0.25 mg/kg)",
    "WTMet10": "Methadone (10 mg/kg)",
    "WTBue3": "Buprenorphine (3 mg/kg)",
}

# Colors used for the sessions in the figures.
session_colors = {
    "salineControl": "tab:gray",
    "WTM30": "tab:blue",
    "WTF025": "tab:orange",
    "WTMet10": "tab:green",
    "WTBue3": "tab:red",
}

# Phase to use for the plotting FC index and analgesia data (as a formula compatible
# with pandas.resample).
reference_phase = "70min"

# Indices of the ROIs used to get functional connectivity values.
fc_roi1, fc_roi2 = 1, 12

# Label for the y-axis of the plot.
y_label = "Ctx (L) - VThal (L) functional connectivity (a.u.)"

# Duration of the FC estimation phases is 10 minutes (provided in seconds).
fc_window_duration = 10 * 60

# Resampling frequencies for the time series, provided as formulas compatible
# with pandas.resample.
fc_resampling_frequency = "5min"
analgesia_resampling_frequency = "5min"

# Duration of the baseline period is 20 minutes (provided in seconds).
baseline_duration = 20 * 60

# Duration of the total fUSI acquisitions is 80 minutes (provided in seconds).
total_duration = 80 * 60

subject_level_path = opioids_results_root / "subject_level_pearson.h5"
group_level_path = opioids_results_root / "group_level_pearson.h5"

registered_root = opioids_dataset_root / "derivatives" / "registration" / "rawdata"

analgesia_xlsx_path = opioids_dataset_root / "derivatives" / "hot_plate_data_2.xlsx"

########################################################################################
# Initializations
########################################################################################

figures_path = Path("figures")
figures_path.mkdir(exist_ok=True, parents=True)

########################################################################################
# Functional connectivity index
########################################################################################

with console.status("[bold cyan]Extract functional connectivity values..."):
    fc_dfs = []
    for session in session_labels:
        subject_level_results = read_session_subject_level_pearson(
            subject_level_path, session
        )
        subject_level_matrices = subject_level_results["correlation_matrices"]

        fc = np.array(
            [
                phase_matrices[:, fc_roi1, fc_roi2]
                for phase_matrices in subject_level_matrices.values()
            ]
        ).T

        fc_df = pd.DataFrame(
            fc.mean(axis=1),
            index=pd.to_timedelta(
                np.arange(1, fc.shape[0] + 1) * fc_window_duration, unit="s"
            ),
            columns=["fc"],  # type: ignore
        )
        fc_df -= fc_df.loc[: pd.Timedelta(baseline_duration, unit="s")].mean(axis=0)
        fc_df = fc_df.resample(fc_resampling_frequency).mean().interpolate()
        fc_df["session"] = session
        fc_dfs.append(fc_df)

    fc_df = pd.concat(fc_dfs)
    fc_df["session"] = fc_df["session"].replace(
        saline_control_sessions, "salineControl"
    )

########################################################################################
# Analgesia
########################################################################################

time_index = pd.to_timedelta(
    np.array([0, 10, 20, 30, 40, 50, 60, 120, 180, 240]) + 20, unit="m"
)


def process_analgesia_df(df: pd.DataFrame) -> pd.DataFrame:
    """Resample and average the analgesia data."""
    # Since the analgesia data is sampled only post-injection, we need to
    # assume the baseline analgesia level is 0 to be able to compare with the
    # fUSI data.
    df.index = time_index
    df.loc[pd.Timedelta("0min")] = 0
    df.loc[pd.Timedelta("10min")] = 0
    df.sort_index(inplace=True)
    df = df.resample(analgesia_resampling_frequency).mean().interpolate()

    # We keep only the first 80 minutes of data, to match the duration of the
    # fUSI data.
    df = df.loc[: pd.Timedelta(total_duration, unit="s")]

    return cast(
        pd.DataFrame,
        df.melt(
            var_name="subject",
            value_name="analgesia",
            ignore_index=False,
        )
        .reset_index(names="time")
        .drop(columns="subject")
        .groupby("time")
        .mean(),
    )


with console.status("[bold cyan]Processing analgesia data..."):
    fentanyl_analgesia_df = pd.read_excel(
        analgesia_xlsx_path,
        skiprows=6,
        header=None,
    ).iloc[:10, 25:31]
    fentanyl_analgesia_df = process_analgesia_df(fentanyl_analgesia_df)

    morphine_analgesia_df = pd.read_excel(
        analgesia_xlsx_path,
        skiprows=23,
        header=None,
    ).iloc[:18, 22:39]
    morphine_analgesia_df = process_analgesia_df(morphine_analgesia_df)

    methadone_analgesia_df = pd.read_excel(
        analgesia_xlsx_path, skiprows=35, header=None, sheet_name=1
    ).iloc[:10, 2:8]
    methadone_analgesia_df = process_analgesia_df(methadone_analgesia_df)

    buprenorphine_analgesia_df = (
        pd.read_excel(analgesia_xlsx_path, skiprows=8, header=None, sheet_name=1)
        .iloc[:10, 2:8]
        .astype("double")
    )
    buprenorphine_analgesia_df = process_analgesia_df(buprenorphine_analgesia_df)

    analgesia_df = pd.concat(
        [
            fentanyl_analgesia_df.assign(session="WTF025"),
            morphine_analgesia_df.assign(session="WTM30"),
            methadone_analgesia_df.assign(session="WTMet10"),
            buprenorphine_analgesia_df.assign(session="WTBue3"),
        ]
    )

########################################################################################
# Merge and plot the functional connectivity index and analgesia data
########################################################################################

with console.status("[bold cyan]Plotting functional connectivity vs analgesia..."):
    merged_df = (
        fc_df.loc[pd.Timedelta(reference_phase)]
        .merge(analgesia_df.loc[pd.Timedelta(reference_phase)], how="outer")
        .fillna(0)
    )
    merged_df.replace({"session": session_labels}, inplace=True)
    merged_df.rename(columns={"session": "Treatment"}, inplace=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.scatterplot(merged_df, x="analgesia", y="fc", hue="Treatment", s=100, ax=ax)
    ax.set_xlabel("Analgesia (%)")
    ax.set_ylabel(y_label)
    fig.savefig(
        figures_path
        / f"fc_vs_analgesia_scatterplot_roi{fc_roi1}-roi{fc_roi2}_{reference_phase}.tiff",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)
