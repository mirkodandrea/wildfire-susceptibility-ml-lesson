# %% [markdown]
# # Wildfire susceptibility: a Random Forest baseline
#
# This notebook is a compact end-to-end example for environmental modelling.
# It shows the complete baseline workflow:
#
# 1. read raster predictors and yearly burned-area rasters
# 2. convert geospatial data into a tabular modelling dataset
# 3. sample burned and unburned pixels
# 4. split pre-2016 samples into training and validation sets
# 5. tune and train a Random Forest
# 6. evaluate on the 2016-2022 hold-out years
# 7. inspect MDA variable importance and map susceptibility
#
# The row unit is one valid 100 m pixel.
# The target is `1` for pixels burned in the period being sampled and `0` for
# sampled pixels unburned in that period.
# The output is a relative susceptibility score, not a calibrated annual fire probability.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, PredefinedSplit, train_test_split

from utils import (
    combined_fire_mask,
    full_period_table,
    pixel_frame,
    print_section,
    sampled_period_table,
)

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 140)

RANDOM_STATE = 42
HOLDOUT_START_YEAR = 2016

DATA_DIR = Path("data")
FIRE_RASTER_DIR = DATA_DIR / "fires"

RASTER_PATHS = {
    "elevation": DATA_DIR / "elevation.tif",
    "slope": DATA_DIR / "slope.tif",
    "aspect_eastness": DATA_DIR / "aspect_eastness.tif",
    "aspect_northness": DATA_DIR / "aspect_northness.tif",
    "vegetation": DATA_DIR / "vegetation.tif",
    "urban_distance": DATA_DIR / "urban_distance.tif",
    "roads_distance": DATA_DIR / "roads_distance.tif",
}

NUMERIC_FEATURES = ["elevation", "slope", "aspect_eastness", "aspect_northness", "urban_distance", "roads_distance"]
FEATURES = [*NUMERIC_FEATURES, "vegetation"]

VEGETATION_NAMES = {
    3111: "Evergreen xeric forest",
    3112: "Thermophilous mixed forest",
    3113: "Mesophilous mixed forest",
    3114: "Beech forest",
    3115: "Chestnut forest",
    3116: "Chestnut orchards",
    3117: "Riparian forest",
    312: "Conifer forest",
    313: "Mixed forest",
    322: "Shrubland and scrub",
    323: "Sclerophyll vegetation",
    324: "Woodland-shrub transition",
}


# %% [markdown]
# ## 1. Prediction problem
#
# A wildfire susceptibility baseline asks whether pixels that burn in future
# years have distinguishable environmental signatures. We train on pre-2016
# burned and unburned pixels, then evaluate on a 2016-2022 hold-out period.
# The model output is a ranking score: higher means "more similar to pixels that
# burned in the training period".

# %%
problem_df = pd.DataFrame(
    {
        "choice": ["unit", "positive label", "negative label", "model output", "main caveat"],
        "definition": [
            "one valid 100 m pixel",
            f"pixel centre in a burned raster cell for the evaluated period",
            f"sampled valid pixel outside burned pixels for the evaluated period",
            "relative susceptibility score",
            "nearby pixels are not independent and predictors are static",
        ],
    }
)
problem_df
print_section("Prediction problem", problem_df)


# %% [markdown]
# ## 2. Raster predictors
#
# Raster values become feature columns in the modelling table.
# We assume all predictors are on the same grid.

# %% [markdown]
# ### Choose the reference grid
#
# Raster data are stored as regular grids. The `slope` raster is used as the
# reference grid for the rest of the workflow:
#
# - `TEMPLATE_SHAPE` tells us how many rows and columns the analysis grid has
# - `TEMPLATE_TRANSFORM` converts between raster row/column indices and map
#   coordinates
#
# The yearly burned-area rasters use this same grid, so target labels and
# predictor values refer to the same pixel locations.

# %%
with rasterio.open(RASTER_PATHS["slope"]) as src:
    TEMPLATE_TRANSFORM = src.transform
    TEMPLATE_SHAPE = (src.height, src.width)

extent = (
    TEMPLATE_TRANSFORM.c,
    TEMPLATE_TRANSFORM.c + TEMPLATE_TRANSFORM.a * TEMPLATE_SHAPE[1],
    TEMPLATE_TRANSFORM.f + TEMPLATE_TRANSFORM.e * TEMPLATE_SHAPE[0],
    TEMPLATE_TRANSFORM.f,
)

raster_arrays = {}

for name, path in RASTER_PATHS.items():
    with rasterio.open(path) as src:
        raster_arrays[name] = src.read(1, masked=True).astype("float32").filled(np.nan)


# %% [markdown]
# ### Create the analysis mask
#
# The model should only learn from pixels where the predictors are meaningful.
# The vegetation raster defines the broad analysis domain: pixels with missing
# vegetation or code `0` are excluded. Numeric predictors are then masked to this
# same domain, and `valid_mask` keeps only pixels where every numeric predictor
# is available.
#
# This mask is reused throughout the notebook. It limits sampling, evaluation,
# and final mapping to the same set of valid landscape pixels.

# %%
vegetation = raster_arrays["vegetation"]
analysis_mask = np.isfinite(vegetation) & (vegetation != 0)

valid_mask = analysis_mask.copy()
for feature in NUMERIC_FEATURES:
    raster_arrays[feature] = np.where(analysis_mask, raster_arrays[feature], np.nan)
    valid_mask &= np.isfinite(raster_arrays[feature])

valid_pixel_count = int(valid_mask.sum())
valid_pixel_count
print_section("Valid analysis pixels", valid_pixel_count)

# %%
raster_summary_df = pd.DataFrame(
    [
        {
            "feature": feature,
            "valid_pixels": int(np.isfinite(raster_arrays[feature]).sum()),
            "median": float(np.nanmedian(raster_arrays[feature])),
        }
        for feature in NUMERIC_FEATURES
    ]
)
raster_summary_df
print_section("Raster predictor summary", raster_summary_df)

# %% [markdown]
# ### Map all predictor rasters
#
# Mapping each feature is a quick visual check before modelling. It helps reveal
# spatial structure, missing areas, and variables that may be strongly tied to
# geography. The vegetation layer is categorical; the other predictors are
# continuous rasters.

# %%
fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
axes = axes.ravel()

for ax, feature in zip(axes, FEATURES):
    feature_map = np.where(analysis_mask, raster_arrays[feature], np.nan)
    cmap = "tab20" if feature == "vegetation" else "viridis"
    image = ax.imshow(feature_map, extent=extent, origin="upper", cmap=cmap)
    ax.set_title(feature.replace("_", " "), loc="left", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)

for ax in axes[len(FEATURES) :]:
    ax.axis("off")

fig.suptitle("Predictor rasters", x=0.01, ha="left", fontweight="bold")
fig


# %% [markdown]
# ## 3. Burned-area rasters
#
# Yearly burned-area rasters define the positive label.
# Each file is a binary raster on the predictor grid:
#
# - `1` means the pixel burned in that year
# - `0` means the pixel did not burn in that year
#
# We use the years only for validation design:
#
# - training period: years before 2016
# - hold-out period: 2016-2022

# %%
fire_raster_paths = {
    int(path.stem.split("_")[1]): path
    for path in sorted(FIRE_RASTER_DIR.glob("fire_*.tiff"))
}
fire_years = sorted(fire_raster_paths)

if not fire_years:
    raise FileNotFoundError(f"No fire rasters found in {FIRE_RASTER_DIR}")

fire_masks = {}
for year, path in fire_raster_paths.items():
    with rasterio.open(path) as src:
        fire_masks[year] = src.read(1).astype(bool)

pixel_area_km2 = abs(TEMPLATE_TRANSFORM.a * TEMPLATE_TRANSFORM.e) / 1_000_000
fire_summary_df = pd.DataFrame(
    [
        {
            "fire_year": year,
            "burned_pixels": int(mask.sum()),
            "burned_area_km2": int(mask.sum()) * pixel_area_km2,
        }
        for year, mask in fire_masks.items()
    ]
)
fire_summary_df.head()
print_section("Burned-area summary by year", fire_summary_df)

# %%
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(fire_summary_df["fire_year"].astype(str), fire_summary_df["burned_area_km2"], color="#b45309")
ax.set_title("Mapped burned area by year", loc="left", fontweight="bold")
ax.set_xlabel("Fire year")
ax.set_ylabel("Burned area (km2)")
ax.tick_params(axis="x", rotation=45)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()


# %% [markdown]
# ## 4. Build the modelling table
#
# The geospatial problem becomes standard supervised learning after sampling:
# each row is a pixel, feature columns come from rasters, and the target comes
# from the burned-area rasters.
#
# We use a temporal split:
#
# - pre-2016 pixels are the training pool
# - a random split of that pool creates a validation set for tuning
# - 2016-2022 pixels are the final hold-out test set
#
# The pre-2016 training pool is balanced: keep all valid burned pixels and
# sample the same number of valid unburned pixels.
#
# The 2016-2022 hold-out test uses the full valid landscape: every valid pixel is
# labelled burned or unburned for the hold-out period. This makes test metrics
# reflect the real class imbalance in the evaluation period.

# %% [markdown]
# ### Convert selected pixels into rows
#
# `pixel_frame` is the bridge between raster data and tabular machine learning.
# It receives raster row/column indices, converts them to map coordinates, and
# extracts each predictor value at those pixel locations.
#
# The returned table has one row per pixel. Keeping `row`, `col`, `x`, and `y`
# makes it possible to move back and forth between model tables and maps.

# %% [markdown]
# ### Combine yearly fire rasters by period
#
# The yearly fire rasters are already on the predictor grid. For a period such
# as 1997-2015 or 2016-2022, we combine the corresponding yearly rasters with a
# logical OR. A `True` cell means that pixel burned at least once during that
# period.
#
# The training helper keeps all valid burned pixels and randomly samples the
# same number of valid unburned pixels. Those unburned samples are
# pseudo-absences: they are valid landscape pixels with no mapped fire in the
# training period, used as the comparison class for this baseline.
#
# The hold-out helper keeps every valid pixel so the test set reflects the real
# imbalance between burned and unburned landscape.

# %% [markdown]
# ### Build temporal datasets
#
# Fires before 2016 define the training pool. Fires from 2016 onward define the
# final hold-out test period. The validation split is carved out of the
# pre-2016 training pool, so model selection never sees the later fires.
#
# This keeps two evaluation questions separate:
#
# - validation: which Random Forest settings work best on pre-2016 samples?
# - hold-out: do those settings rank later burned pixels higher than later
#   unburned pixels?

# %%
train_years = [year for year in fire_years if year < HOLDOUT_START_YEAR]
test_years = [year for year in fire_years if year >= HOLDOUT_START_YEAR]

train_burned_mask = combined_fire_mask(fire_masks, train_years)
test_burned_mask = combined_fire_mask(fire_masks, test_years)

# %% [markdown]
# ### Map the labels used by each period
#
# The training label map marks pixels that burned at least once before 2016.
# The hold-out label map marks pixels that burned at least once from 2016 to
# 2022. These maps show the spatial target pattern before any model is fitted.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

for ax, period_name, label_mask in [
    (axes[0], f"{min(train_years)}-{max(train_years)} training labels", train_burned_mask),
    (axes[1], f"{min(test_years)}-{max(test_years)} hold-out labels", test_burned_mask),
]:
    label_map = np.where(valid_mask, label_mask.astype(float), np.nan)
    image = ax.imshow(label_map, extent=extent, origin="upper", cmap="Reds", vmin=0, vmax=1)
    ax.set_title(period_name, loc="left", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02, label="Burned label")
fig.suptitle("Burned labels by modelling period", x=0.01, ha="left", fontweight="bold")
fig

# %%
# Training uses earlier fires; testing uses later fires. The training table
# keeps all pre-2016 burned pixels and samples an equal number of pre-2016
# pseudo-absence pixels from valid, unburned locations.
train_pool_df = sampled_period_table(
    train_burned_mask,
    f"{min(train_years)}-{max(train_years)}",
    RANDOM_STATE,
    valid_mask,
    TEMPLATE_TRANSFORM,
    raster_arrays,
    FEATURES,
)
test_df = full_period_table(
    test_burned_mask,
    f"{min(test_years)}-{max(test_years)}",
    valid_mask,
    TEMPLATE_TRANSFORM,
    raster_arrays,
    FEATURES,
)

# The validation split is internal to the balanced pre-2016 training pool.
# Stratification keeps the 50/50 presence/pseudo-absence ratio in both the
# model-fitting split and the validation split. The validation split is used for
# tuning, not for the final scientific evaluation.
train_model_df, validation_df = train_test_split(
    train_pool_df,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=train_pool_df["target"],
)

train_pool_df.head()
print_section("Training-pool sample", train_pool_df.head())

# %%
dataset_summary_df = pd.DataFrame(
    [
        {
            "dataset": "training pool",
            "period": train_pool_df["period"].iloc[0],
            "rows": len(train_pool_df),
            "burned_pixels": int(train_pool_df["target"].sum()),
            "sampled_unburned_pixels": int((train_pool_df["target"] == 0).sum()),
            "burned_share": float(train_pool_df["target"].mean()),
        },
        {
            "dataset": "model training split",
            "period": train_model_df["period"].iloc[0],
            "rows": len(train_model_df),
            "burned_pixels": int(train_model_df["target"].sum()),
            "sampled_unburned_pixels": int((train_model_df["target"] == 0).sum()),
            "burned_share": float(train_model_df["target"].mean()),
        },
        {
            "dataset": "validation split",
            "period": validation_df["period"].iloc[0],
            "rows": len(validation_df),
            "burned_pixels": int(validation_df["target"].sum()),
            "sampled_unburned_pixels": int((validation_df["target"] == 0).sum()),
            "burned_share": float(validation_df["target"].mean()),
        },
        {
            "dataset": "hold-out test",
            "period": test_df["period"].iloc[0],
            "rows": len(test_df),
            "burned_pixels": int(test_df["target"].sum()),
            "unburned_pixels": int((test_df["target"] == 0).sum()),
            "burned_share": float(test_df["target"].mean()),
        },
    ]
)
dataset_summary_df
print_section("Dataset summary", dataset_summary_df)

# %%
sample_summary_df = (
    pd.concat([train_pool_df, test_df], ignore_index=True)
    .groupby(["period", "target_name"], observed=True)
    .size()
    .unstack(fill_value=0)
    .rename_axis(columns=None)
    .reset_index()
)
sample_summary_df
print_section("Pixel samples by period", sample_summary_df)


# %% [markdown]
# ## 5. Quick exploratory analysis
#
# Before fitting a model, ask whether burned and unburned sampled pixels differ
# in the pre-2016 training pool.

# %%
class_summary_df = (
    train_pool_df.groupby("target_name", observed=True)[NUMERIC_FEATURES]
    .median()
    .T.rename_axis("feature")
    .reset_index()
)
class_summary_df
print_section("Class-wise median numeric predictors", class_summary_df)

# %%
vegetation_summary_df = (
    pd.crosstab(
        train_pool_df["vegetation"].map(VEGETATION_NAMES).fillna(train_pool_df["vegetation"].astype(str)),
        train_pool_df["target_name"],
        normalize="columns",
    )
    .mul(100)
    .sort_values("burned", ascending=False)
    .head(8)
)
vegetation_summary_df
print_section("Top vegetation classes by sampled burned share", vegetation_summary_df)

# %% [markdown]
# ### Compare all numeric features by sampled class
#
# These histograms use the balanced pre-2016 training pool. Blue is the sampled
# pseudo-absence class; red is the burned class. Strong separation suggests a
# feature may help rank susceptibility, while heavy overlap means the model will
# need combinations of variables rather than a single threshold.

# %%
fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
axes = axes.ravel()

for ax, feature in zip(axes, NUMERIC_FEATURES):
    for target, label, color in [(0, "unburned", "#2563eb"), (1, "burned", "#dc2626")]:
        ax.hist(
            train_pool_df.loc[train_pool_df["target"] == target, feature],
            bins=35,
            density=True,
            alpha=0.45,
            color=color,
            label=label,
        )
    ax.set_title(feature.replace("_", " "), loc="left", fontweight="bold")
    ax.set_ylabel("Density")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].legend(frameon=False)
fig.suptitle("Sampled numeric predictor distributions", x=0.01, ha="left", fontweight="bold")
fig


# %% [markdown]
# ## 6. Train and tune the baseline model
#
# The model needs only numeric columns.
# Numeric raster features are already numeric. The vegetation code is
# categorical, so we convert it to dummy variables with `pd.get_dummies`.
#
# Hyperparameters are selected on the validation split only.
# The 2016-2022 hold-out test set is not used until final evaluation.

# %%
def make_features(frame, columns=None):
    # RandomForestClassifier needs numeric columns.
    # Numeric raster predictors are already ready to use.
    numeric = frame[NUMERIC_FEATURES].reset_index(drop=True)

    # Vegetation is categorical: code 3111 is not "smaller" than 324 in a
    # meaningful numeric sense, so we expand it into one binary column per class.
    vegetation = pd.get_dummies(frame["vegetation"].astype(int), prefix="vegetation", dtype=int)
    X = pd.concat([numeric, vegetation.reset_index(drop=True)], axis=1)

    # Validation, test, and map data must have exactly the same columns as train.
    # Missing vegetation classes get a zero-filled dummy column.
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X


X_train = make_features(train_model_df)
FEATURE_COLUMNS = X_train.columns

# Reuse the training columns everywhere to avoid train/test column mismatch.
X_valid = make_features(validation_df, FEATURE_COLUMNS)
X_test = make_features(test_df, FEATURE_COLUMNS)

y_train = train_model_df["target"]
y_valid = validation_df["target"]
y_test = test_df["target"]

param_grid = {
    "n_estimators": [150, 300],
    "max_features": ["sqrt", 0.5],
    "min_samples_leaf": [1, 3, 5],
    "max_depth": [None, 15],
}

X_tuning = pd.concat([X_train, X_valid], ignore_index=True)
y_tuning = pd.concat([y_train, y_valid], ignore_index=True)

# PredefinedSplit tells GridSearchCV: fit on rows marked -1 and validate on
# rows marked 0. This preserves our explicit train/validation split.
validation_fold = PredefinedSplit(
    test_fold=np.r_[
        np.full(len(X_train), -1),
        np.zeros(len(X_valid), dtype=int),
    ]
)

tuning_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    param_grid=param_grid,
    scoring=["roc_auc", "average_precision"],
    refit="roc_auc",
    cv=validation_fold,
    n_jobs=1,
)
tuning_search.fit(X_tuning, y_tuning)

tuning_results_df = (
    pd.DataFrame(tuning_search.cv_results_)
    .sort_values("rank_test_roc_auc")
    [
        [
            "rank_test_roc_auc",
            "mean_test_roc_auc",
            "mean_test_average_precision",
            "param_n_estimators",
            "param_max_features",
            "param_min_samples_leaf",
            "param_max_depth",
        ]
    ]
    .reset_index(drop=True)
)
tuning_results_df.head()
print_section("Top tuning results", tuning_results_df.head(10))

# %%
best_params = tuning_search.best_params_
best_params
print_section("Selected hyperparameters", best_params)

# %%
validation_model = tuning_search.best_estimator_

# After choosing hyperparameters, refit on all pre-2016 training-pool samples.
# The 2016-2022 hold-out set remains untouched until evaluation.
X_train_pool = make_features(train_pool_df, FEATURE_COLUMNS)
y_train_pool = train_pool_df["target"]
X_test = make_features(test_df, FEATURE_COLUMNS)

rf_model = RandomForestClassifier(
    **best_params,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf_model.fit(X_train_pool, y_train_pool)

rf_model
print_section("Fitted Random Forest", rf_model)


# %% [markdown]
# ## 7. Evaluate probability scores
#
# Susceptibility is usually used as a ranking.
# The key evaluation question is:
#
# **Do 2016-2022 burned pixels receive higher scores than sampled 2016-2022
# unburned pixels?**
#
# ROC AUC and average precision evaluate ranking.
# Brier loss checks the numeric quality of the score as a probability-like value.

# %%
def fire_score(model, X):
    # predict_proba returns one column per class; select the probability of class 1.
    return model.predict_proba(X)[:, list(model.classes_).index(1)]


def score_metrics(split, name, model, X, y):
    score = fire_score(model, X)

    # The 0.5 label is only a reporting threshold.
    # Ranking metrics such as ROC AUC use the full continuous score.
    label = score >= 0.5
    return {
        "split": split,
        "model": name,
        "roc_auc": roc_auc_score(y, score),
        "average_precision": average_precision_score(y, score),
        "brier_loss": brier_score_loss(y, score),
        "balanced_accuracy_at_0_50": balanced_accuracy_score(y, label),
    }


metrics_df = pd.DataFrame(
    [
        score_metrics("validation", "tuned random forest", validation_model, X_valid, y_valid),
        score_metrics("2016-2022 hold-out", "tuned random forest", rf_model, X_test, y_test),
    ]
).set_index(["split", "model"])
metrics_df
print_section("Validation and hold-out metrics", metrics_df)

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

RocCurveDisplay.from_estimator(rf_model, X_test, y_test, name="random forest", ax=axes[0])
axes[0].plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", linewidth=1)
axes[0].set_title("ROC curve", loc="left", fontweight="bold")

PrecisionRecallDisplay.from_estimator(rf_model, X_test, y_test, name="random forest", ax=axes[1])
axes[1].set_title("Precision-recall curve", loc="left", fontweight="bold")

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig

# %%
test_scores = test_df[["target", "target_name", "row", "col", "x", "y"]].copy()
test_scores["score"] = fire_score(rf_model, X_test)

fig, ax = plt.subplots(figsize=(7, 4.5))
for target, label, color in [(0, "unburned", "#2563eb"), (1, "burned", "#dc2626")]:
    ax.hist(test_scores.loc[test_scores["target"] == target, "score"], bins=30, density=True, alpha=0.5, color=color, label=label)
ax.set_title("Random Forest scores on held-out pixels", loc="left", fontweight="bold")
ax.set_xlabel("Susceptibility score")
ax.set_ylabel("Density")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False)
fig.tight_layout()
fig

# %%
score_bin = pd.qcut(test_scores["score"], q=10, duplicates="drop")
score_bin_df = (
    test_scores.assign(score_bin=score_bin)
    .groupby("score_bin", observed=True)
    .agg(
        rows=("target", "size"),
        mean_score=("score", "mean"),
        observed_presence_share=("target", "mean"),
    )
    .reset_index()
)
score_bin_df
print_section("Observed burned share by score bin", score_bin_df)


# %% [markdown]
# ## 8. Operational percentile thresholds
#
# For operational use, the score can be converted into susceptibility classes by
# selecting score percentiles across the territory.
#
# This is often easier to explain to stakeholders than a raw model probability:
# for example, "high susceptibility" can mean the top 25% of valid pixels by
# Random Forest score.
#
# The percentile is a policy choice. A smaller high-susceptibility area is easier
# to inspect or prioritize, but it will miss more burned pixels. The hold-out test
# below quantifies this trade-off using 2016-2022 fires.

# %%
baseline_burned_share = test_scores["target"].mean()

operational_threshold_df = pd.DataFrame(
    [
        {
            "high_risk_territory_share": high_risk_share,
            "score_percentile_threshold": 1 - high_risk_share,
            "score_threshold": test_scores["score"].quantile(1 - high_risk_share),
            "mapped_high_risk_share": float((test_scores["score"] >= test_scores["score"].quantile(1 - high_risk_share)).mean()),
            "precision": precision_score(
                y_test,
                test_scores["score"] >= test_scores["score"].quantile(1 - high_risk_share),
                zero_division=0,
            ),
            "recall": recall_score(
                y_test,
                test_scores["score"] >= test_scores["score"].quantile(1 - high_risk_share),
                zero_division=0,
            ),
            "lift_vs_landscape": precision_score(
                y_test,
                test_scores["score"] >= test_scores["score"].quantile(1 - high_risk_share),
                zero_division=0,
            )
            / baseline_burned_share,
        }
        for high_risk_share in [0.10, 0.25, 0.50]
    ]
)
operational_threshold_df
print_section("Operational percentile thresholds on hold-out period", operational_threshold_df)

# %%
medium_threshold = test_scores["score"].quantile(0.50)
high_threshold = test_scores["score"].quantile(0.75)

test_scores["susceptibility_class"] = pd.cut(
    test_scores["score"],
    bins=[-np.inf, medium_threshold, high_threshold, np.inf],
    labels=["low", "medium", "high"],
    include_lowest=True,
)

susceptibility_class_df = (
    test_scores.groupby("susceptibility_class", observed=True)
    .agg(
        pixels=("target", "size"),
        territory_share=("target", lambda values: len(values) / len(test_scores)),
        mean_score=("score", "mean"),
        burned_pixels=("target", "sum"),
        observed_burned_share=("target", "mean"),
    )
    .reset_index()
)
susceptibility_class_df["captured_burned_share"] = susceptibility_class_df["burned_pixels"] / test_scores["target"].sum()
susceptibility_class_df["lift_vs_landscape"] = susceptibility_class_df["observed_burned_share"] / baseline_burned_share
susceptibility_class_df
print_section("Low, medium, high susceptibility classes on hold-out period", susceptibility_class_df)

# %%
fig, ax = plt.subplots(figsize=(7, 4))
class_plot_df = susceptibility_class_df.set_index("susceptibility_class").loc[["low", "medium", "high"]]
ax.bar(class_plot_df.index, class_plot_df["captured_burned_share"], color=["#2563eb", "#f59e0b", "#dc2626"])
ax.set_title("Burned pixels captured by susceptibility class", loc="left", fontweight="bold")
ax.set_xlabel("Susceptibility class")
ax.set_ylabel("Share of 2016-2022 burned pixels")
ax.set_ylim(0, max(0.05, class_plot_df["captured_burned_share"].max() * 1.15))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig


# %% [markdown]
# ## 9. Thresholds are decisions
#
# The Random Forest gives a score.
# A threshold turns that score into a class label.
#
# There is no universal wildfire threshold: it depends on the cost of missing
# susceptible areas versus over-flagging lower-susceptibility areas.

# %%
threshold_df = pd.DataFrame(
    [
        {
            "threshold": threshold,
            "mapped_positive_share": float((test_scores["score"] >= threshold).mean()),
            "precision": precision_score(y_test, test_scores["score"] >= threshold, zero_division=0),
            "recall": recall_score(y_test, test_scores["score"] >= threshold, zero_division=0),
        }
        for threshold in [0.25, 0.50, 0.75]
    ]
)
threshold_df
print_section("Threshold sensitivity", threshold_df)

# %%
threshold = 0.50

fig, ax = plt.subplots(figsize=(5, 4.5))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    test_scores["score"] >= threshold,
    display_labels=["unburned", "burned"],
    cmap="Blues",
    colorbar=False,
    ax=ax,
)
ax.set_title(f"Threshold = {threshold:.2f}", loc="left", fontweight="bold")
fig.tight_layout()
fig


# %% [markdown]
# ## 10. Validation design matters
#
# The validation split is random within the pre-2016 training pool, so it is
# useful for tuning but still optimistic for spatial data.
#
# The 2016-2022 hold-out is a harder test because it evaluates later fires that
# were not used for fitting or tuning. It is still not perfect: nearby pixels can
# share terrain, vegetation, accessibility, and fire history.


# %% [markdown]
# ## 11. MDA variable importance
#
# MDA means Mean Decrease in Accuracy.
# Here "accuracy" is the validation metric we care about: ROC AUC.
#
# The idea is simple: shuffle one original predictor, score the model again,
# and measure how much ROC AUC decreases.
#
# We ask scikit-learn to permute the original pre-encoded columns. For
# `vegetation`, this means the single vegetation code column is shuffled as one
# variable.
#
# This is model interpretation, not causal attribution.
# For example, coordinates can be predictive because they summarize location,
# but location is not itself a transferable fire mechanism.

# %%
def raw_feature_auc(model, raw_X, y):
    # permutation_importance passes the raw pre-encoded frame here.
    # Encode it with the same columns used to train the model, then score it.
    encoded_X = make_features(raw_X, FEATURE_COLUMNS)
    return roc_auc_score(y, fire_score(model, encoded_X))


importance = permutation_importance(
    rf_model,
    test_df[FEATURES],
    y_test,
    scoring=raw_feature_auc,
    n_repeats=5,
    random_state=RANDOM_STATE,
    n_jobs=1,
)

importance_df = (
    pd.DataFrame(
        {
            "feature": FEATURES,
            "mda_roc_auc_drop": importance.importances_mean,
            "std": importance.importances_std,
        }
    )
    .sort_values("mda_roc_auc_drop", ascending=False)
    .reset_index(drop=True)
)
importance_df
print_section("Permutation importance on hold-out period", importance_df)

# %%
fig, ax = plt.subplots(figsize=(7, 4))
plot_df = importance_df.sort_values("mda_roc_auc_drop")
ax.barh(plot_df["feature"], plot_df["mda_roc_auc_drop"], color="#1d4ed8")
ax.set_title("MDA variable importance", loc="left", fontweight="bold")
ax.set_xlabel("Mean decrease in ROC AUC after permutation")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig


# %% [markdown]
# ## 12. Refit and map susceptibility
#
# For a strict hold-out evaluation, the mapped model below is still fitted only
# on the pre-2016 training pool with the selected hyperparameters.
#
# The map is a model product. It is not independent evidence of accuracy.

# %%
final_model = RandomForestClassifier(
    **best_params,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
final_model.fit(X_train_pool, y_train_pool)

rows, cols = np.where(valid_mask)
grid_pixels = pixel_frame(
    rows,
    cols,
    target=0,
    sample_type="map",
    template_transform=TEMPLATE_TRANSFORM,
    raster_arrays=raster_arrays,
    features=FEATURES,
)

# Predict the fitted model over every valid pixel to create the susceptibility map.
grid_features = make_features(grid_pixels, FEATURE_COLUMNS)
grid_scores = fire_score(final_model, grid_features)

susceptibility_grid = np.full(TEMPLATE_SHAPE, np.nan, dtype=float)
susceptibility_grid[rows, cols] = grid_scores

grid_score_summary = pd.Series(grid_scores).describe(percentiles=[0.1, 0.5, 0.9])
grid_score_summary
print_section("Mapped susceptibility score summary", grid_score_summary)

fig, ax = plt.subplots(figsize=(12, 5.5))
image = ax.imshow(susceptibility_grid, extent=extent, origin="upper", cmap="viridis", vmin=0, vmax=1)

# Red contours show the true burned pixels from the 2016-2022 hold-out period.
# This overlay is visual evaluation only; the numeric test metrics above are the formal evaluation.
holdout_burned_mask = test_burned_mask & valid_mask
ax.contour(
    holdout_burned_mask.astype(int),
    levels=[0.5],
    extent=extent,
    origin="upper",
    colors="red",
    linewidths=0.8,
)

ax.set_title("Predicted wildfire susceptibility", loc="left", fontweight="bold")
ax.set_xlabel("Easting")
ax.set_ylabel("Northing")
ax.set_aspect("equal", adjustable="box")
fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="Model score")
fig.tight_layout()
fig


# %% [markdown]
# ## 13. Take-home messages
#
# - Raster predictors can be converted into a tabular supervised-learning problem.
# - Yearly burned-area rasters provide positives; sampled unburned pixels define the comparison group.
# - Elevation, slope, aspect, vegetation, and accessibility distances are used as static predictors.
# - Categorical raster classes can be made numeric with simple dummy variables.
# - Use a random validation split inside the training period to tune hyperparameters.
# - Keep the 2016-2022 hold-out years untouched until final evaluation.
# - Percentile thresholds translate model scores into operational territory shares.
# - Evaluate scores before choosing thresholds.
# - MDA variable importance explains model behavior, not ecological causality.
# - The susceptibility map is a model output and must be interpreted with the sampling and validation design in mind.


# %% [markdown]
# ## 14. Exercises
#
# 1. **Add temperature and rain predictors.**
#    Extend `RASTER_PATHS`, `NUMERIC_FEATURES`, and `FEATURES` with the climate
#    rasters in `data/`:
#
#    - `temperature_summer.tif`
#    - `temperature_winter.tif`
#    - `precipitation_summer.tif`
#    - `precipitation_winter.tif`
#
#    Re-run the workflow and compare the baseline Random Forest with the
#    climate-augmented Random Forest. Report at least ROC AUC, average precision,
#    and Brier loss on the 2016-2022 hold-out period. Also inspect whether the
#    climate variables appear important in the MDA table.
#
# 2. **Try another learning algorithm.**
#    Replace or complement `RandomForestClassifier` with another classifier,
#    such as `HistGradientBoostingClassifier`, `ExtraTreesClassifier`,
#    `LogisticRegression`, or `KNeighborsClassifier`. Keep the same train,
#    validation, and 2016-2022 hold-out splits so the comparison is fair.
#    Compare validation metrics, hold-out metrics, and the shape of the score
#    distributions.
#
# 3. **Use a geographical split.**
#    Design a spatial validation strategy instead of the temporal hold-out. For
#    example, split pixels by easting or northing, train on one part of Liguria,
#    and test on the held-out part. Compare the result with the temporal split.
#    Does the model still generalize when evaluated on places it did not see
#    during fitting? Explain how the geographical split changes the scientific
#    question being tested.
