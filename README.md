# Opioid-Specific Brain Connectivity Dynamics Distinguish Analgesia from Secondary Effects

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10401165.svg)](https://doi.org/10.5281/zenodo.10401165)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10286698.svg)](https://doi.org/10.5281/zenodo.10286698)

This repository contains the Python package and scripts for analysis and visualization
of the Opioids dataset, from the article "Opioid-Specific Brain Connectivity Dynamics
Distinguish Analgesia from Secondary Effects".

## Abstract

> The µ-opioid receptor (MOP) is a critical pharmaceutical target that mediates both the
> therapeutic benefits and adverse effects of opioid drugs. However, the large-scale
> neural circuit dynamics underlying key opioid effects, such as analgesia and
> respiratory depression, remain poorly understood, hindering the development of safer
> analgesics. Here, we present a multimodal experimental framework that integrates
> functional ultrasound imaging (fUSI) through the intact skull with behavioral and
> molecular analyses to investigate opioid-induced large-scale functional responses and
> their physiological relevance in awake, behaving mice. Administration of major
> opioids—morphine, fentanyl, methadone, and buprenorphine—elicited robust, dose- and
> time-dependent reorganization of functional brain connectivity (FC) patterns, with
> magnitude scaling according to MOP agonist efficacy. This opioid-specific functional
> fingerprint is marked by decreased FC between the somatosensory cortex and
> hippocampal/thalamic regions and increased bilateral subcortical FC within the
> somatosensory cortex. Notably, this fingerprint was attenuated following tolerance
> induction and abolished by pharmacological or genetic MOP inactivation. Through power
> Doppler spectral analysis and lagged correlation measurements, we show that morphine
> perturbs temporal FC dynamics and the propagation of brain-wide oscillatory activity,
> disrupting critical-state dynamics. Importantly, we identify a dissociation between
> fast, transient processes—such as cerebral blood volume (CBV) changes, locomotion, and
> respiratory depression—and slower processes driving FC reorganization, analgesia, and
> sustained MOP activation. This study provides mechanistic insights into opioid-induced
> network reorganization, establishes FC alterations as a reliable biomarker of opioid
> efficacy, and offers a framework for advancing the development of analgesic compounds
> with improved therapeutic windows and reduced side effects.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for Python dependency management. You
may also find installing [just](https://github.com/casey/just) useful to run the
provided recipes.

## Repository Structure

- `src/opioids_analysis/`: Python package containing analysis modules
  - `pearson.py`: Pearson correlation analysis functions
  - `cross_correlation.py`: Cross-correlation analysis functions
  - `multimodal.py`: Multimodal data processing utilities
  - `plotting.py`: Plotting utilities for figures
- `article-figures/`: Scripts to generate publication figures
  - `01_figure_pearson.py`: Pearson correlation analysis and figures
  - `02_figure_fc_vs_analgesia.py`: Functional connectivity vs analgesia scatter plots
  - `03_figure_multimodal.py`: Multimodal time series and correlation matrices
  - `04_figure_rcbv_rasterplots.py`: Relative CBV rasterplots
  - `05_figure_xcorr.py`: Cross-correlation analysis and figures
- `params/`: Analysis parameters and ROI masks

## Generating Figures

The scripts in `article-figures/` generate the publication sub-figures. They must be run
in order due to dependencies:

```bash
# Run all figure generation scripts in order
just figures

# Or run individual scripts with uv
uv run article-figures/01_figure_pearson.py
uv run article-figures/02_figure_fc_vs_analgesia.py
# ... and so on
```

> [!NOTE]
> The analysis scripts require access to the preprocessed dataset, which is not included
> in this repository. Please download it from the [Zenodo
> archive](https://doi.org/10.5281/zenodo.10286698).
