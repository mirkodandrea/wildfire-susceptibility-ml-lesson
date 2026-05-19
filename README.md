# Machine Learning for PhDs: Wildfire Susceptibility

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mirkodandrea/wildfire-susceptibility-ml-lesson/blob/main/wildfire_susceptibility_rf_intro.ipynb)

Lesson material for a compact, end-to-end machine learning workflow using
environmental raster data. The main lesson builds a tuned Random Forest baseline
for wildfire susceptibility from static landscape predictors and yearly
burned-area rasters.

The example is designed for teaching: it starts with raster inspection, turns
valid 100 m pixels into a tabular modelling dataset, engineers circular and
categorical predictors, trains and tunes a classifier, evaluates on later
hold-out years, inspects permutation importance, and maps the resulting
susceptibility score.

## Contents

- `wildfire_susceptibility_rf_intro.py` - the main lesson, written as a
  Jupytext/percent-format notebook.
- `wildfire_susceptibility_rf_intro.ipynb` - generated notebook for Google Colab
  and notebook viewers.
- `wildfire_susceptibility_rf_intro.html` - generated HTML version with executed
  outputs.
- `lesson_setup.py` - shared imports and Colab runtime setup for the lesson.
- `utils.py` - helper functions for combining yearly fire masks and converting
  raster pixels into tabular samples.
- `scripts/render_ipynb_html.sh` - local rebuild script for the executed
  notebook and HTML export.
- `scripts/deploy_lesson.sh` - local publish script that rebuilds, executes,
  exports, commits, and pushes the lesson outputs.
- `data/` - prepared predictor rasters and yearly burned-area rasters.
- `pyproject.toml` and `uv.lock` - Python environment definition.

## Lesson Outline

The main lesson follows the script sections:

1. Setup: imports, constants, paths, and runtime configuration.
2. Data loading and analysis mask: aligned raster ingestion and valid-pixel mask.
3. Burned-area target and temporal split: yearly fire rasters, pre-2016 training
   pool, and 2016-2022 hold-out period.
4. Feature engineering: aspect eastness/northness and vegetation one-hot
   encoding.
5. Training set construction: balanced pseudo-absence sampling for training and
   full-landscape hold-out labels.
6. Exploratory class comparison: burned vs sampled-unburned predictor contrasts.
7. Model tuning: `GridSearchCV` with a fixed validation fold and ROC-AUC
   scoring.
8. Model fitting: separate validation and final pre-2016 Random Forest models.
9. Hold-out evaluation: ROC-AUC, recall@top%, score distributions, and decile
   calibration checks.
10. Threshold decisions: percentile thresholds, susceptibility classes, and
    balanced confusion-matrix diagnostics.
11. Model explanation: permutation importance as mean decrease in hold-out
    ROC-AUC.
12. Susceptibility map: full-grid prediction with 2016-2022 burned-area overlay.

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

In Google Colab, use the badge at the top of this README. The first setup cell
downloads `lesson_setup.py` if needed; in Colab the setup script installs the
lesson dependencies, clones this repository, and switches into the cloned
working directory.

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

To rebuild the executed notebook and HTML export locally:

```bash
scripts/render_ipynb_html.sh
```

## Publishing

To rebuild the notebook, execute it so outputs are stored, export HTML, commit,
and push:

```bash
scripts/deploy_lesson.sh
```

You can pass a custom commit message:

```bash
scripts/deploy_lesson.sh "Update executed lesson outputs"
```

## Data

The prepared `data/` directory contains:

- default static predictor rasters: elevation, slope, aspect, vegetation,
  distance to roads, and distance to urban areas;
- optional climate rasters used in the extension exercise: summer/winter
  temperature and summer/winter precipitation;
- yearly burned-area rasters from 1997 through 2022.

The lesson assumes these rasters are already aligned on the same 100 m grid.

## Teaching Notes

The workflow intentionally keeps the baseline simple. It uses balanced
presence/pseudo-absence samples for model fitting, then evaluates on the full
valid landscape for the hold-out period. This contrast is part of the lesson:
validation design, class imbalance, spatial dependence, and threshold choice
matter as much as the classifier itself.

Important caveats are called out in the notebook: predictors are treated as
time-invariant, nearby pixels are spatially autocorrelated, pseudo-absence
sampling changes class prevalence, Random Forest vote fractions are not
calibrated probabilities, and permutation importance is model reliance rather
than causal attribution.

## Exercises

The lesson ends with short extensions: adding climate predictors, trying another
classifier, designing a spatial cross-validation split, varying the
pseudo-absence ratio, and bootstrapping a confidence interval for hold-out AUC.
