# Machine Learning for PhDs: Wildfire Susceptibility

[![Build notebook](https://github.com/mirkodandrea/wildfire-susceptibility-ml-lesson/actions/workflows/build-notebook.yml/badge.svg)](https://github.com/mirkodandrea/wildfire-susceptibility-ml-lesson/actions/workflows/build-notebook.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mirkodandrea/wildfire-susceptibility-ml-lesson/blob/main/wildfire_susceptibility_rf_intro.ipynb)

Lesson material for a compact, end-to-end machine learning workflow using
environmental raster data. The main lesson builds a Random Forest baseline for
wildfire susceptibility from static landscape predictors and yearly burned-area
rasters.

The example is designed for teaching: it starts with raster inspection, turns
pixels into a tabular modelling dataset, trains and tunes a classifier, evaluates
on later hold-out years, inspects permutation importance, and maps the resulting
susceptibility score.

## Contents

- `wildfire_susceptibility_rf_intro.py` - the main lesson, written as a
  Jupytext/percent-format notebook.
- `wildfire_susceptibility_rf_intro.ipynb` - generated notebook for Google Colab
  and notebook viewers.
- `utils.py` - helper functions for combining yearly fire masks and converting
  raster pixels into tabular samples.
- `data/` - prepared predictor rasters and yearly burned-area rasters.
- `pyproject.toml` and `uv.lock` - Python environment definition.

## Lesson Outline

The main lesson covers:

1. Reading raster predictors and yearly burned-area rasters.
2. Building a valid analysis mask and checking predictor rasters.
3. Converting burned and unburned pixels into a modelling table.
4. Splitting pre-2016 samples into training and validation data.
5. Tuning and training a `RandomForestClassifier`.
6. Evaluating susceptibility scores on the 2016-2022 hold-out period.
7. Comparing balanced validation metrics with full-landscape hold-out metrics.
8. Exploring threshold choices for operational susceptibility classes.
9. Inspecting mean decrease in accuracy permutation importance.
10. Refitting the model and mapping the final susceptibility surface.
11. Working through short extension exercises.

The target is `1` for pixels burned during the sampled period and `0` for
sampled unburned pixels. The model output should be interpreted as a relative
susceptibility score, not as a calibrated annual fire probability.

## Requirements

This project uses `uv` for environment management. The lockfile currently targets
Python `>=3.14`.

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create or update the environment:

```bash
uv sync
```

## Running the Lesson

Start JupyterLab:

```bash
uv run jupyter lab
```

Then open `wildfire_susceptibility_rf_intro.py`. Jupytext will treat it as a
notebook with executable cells.

You can also convert it to an `.ipynb` notebook:

```bash
uv run jupytext --to ipynb wildfire_susceptibility_rf_intro.py
```

Or execute the script-style notebook from the command line:

```bash
uv run python wildfire_susceptibility_rf_intro.py
```

## Data

The prepared `data/` directory contains:

- static predictor rasters such as elevation, slope, aspect components,
  vegetation, and distances to roads and urban areas;
- yearly burned-area rasters from 1997 through 2022.

The lesson assumes these rasters are already aligned on the same 100 m grid.

## Teaching Notes

The workflow intentionally keeps the baseline simple. It uses balanced
presence/pseudo-absence samples for model fitting, then evaluates on the full
valid landscape for the hold-out period. This contrast is part of the lesson:
validation design, class imbalance, spatial dependence, and threshold choice
matter as much as the classifier itself.
