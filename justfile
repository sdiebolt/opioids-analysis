# Print the help message.
@help:
    echo 'Usage: just [RECIPE]'
    echo
    just --list

# Generate the figures from the preprocessed dataset.
figures:
    uv run article-figures/01_figure_pearson.py
    uv run article-figures/02_figure_fc_vs_analgesia.py
    uv run article-figures/03_figure_multimodal.py
    uv run article-figures/04_figure_rcbv_rasterplots.py
    uv run article-figures/05_figure_xcorr.py

# Create the zip archive for Zenodo.
zip:
    git archive --format=zip --output=opioids-analysis.zip main
