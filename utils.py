import numpy as np
import pandas as pd


def print_section(title, value):
    """Print notebook results when this file is run as a script."""
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")
    if isinstance(value, pd.DataFrame):
        print(value.to_string())
    elif isinstance(value, pd.Series):
        print(value.to_string())
    else:
        print(value)


def combined_fire_mask(fire_masks, years):
    """Combine yearly binary fire rasters into one period mask."""
    selected_masks = [fire_masks[year] for year in years]
    if not selected_masks:
        raise ValueError("At least one fire year is required.")
    return np.logical_or.reduce(selected_masks)


def pixel_frame(
    rows,
    cols,
    target,
    sample_type,
    template_transform,
    raster_arrays,
    features,
):
    """Convert raster row/column indices into a tabular pixel dataset."""
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)

    frame = pd.DataFrame(
        {
            "target": int(target),
            "sample_type": sample_type,
            "row": rows,
            "col": cols,
            "x": template_transform.c + (cols + 0.5) * template_transform.a,
            "y": template_transform.f + (rows + 0.5) * template_transform.e,
        }
    )

    for feature in features:
        values = raster_arrays[feature][rows, cols]
        frame[feature] = values.astype(int) if feature == "vegetation" else values

    return frame


def sampled_period_table(
    period_burned_mask,
    period_name,
    seed,
    valid_mask,
    template_transform,
    raster_arrays,
    features,
):
    """Build a balanced burned/unburned pixel table for a fire period."""
    # Presence samples: every valid pixel that burned at least once in the
    # training period is kept. The period mask is already a logical OR across
    # the yearly fire rasters selected by the notebook.
    burned_rows, burned_cols = np.where(period_burned_mask & valid_mask)

    # Pseudo-absence candidates: valid pixels with no mapped fire in the same
    # period. These are not guaranteed true absences in an ecological sense;
    # they are sampled background/unburned comparison pixels.
    unburned_rows, unburned_cols = np.where(~period_burned_mask & valid_mask)

    # Balance presences and pseudo-absences for a simple baseline classifier.
    # This makes fitting and interpretation easier, but it deliberately changes
    # class prevalence: the training table is 50/50 even though fire is rare on
    # the real landscape.
    rng = np.random.default_rng(seed)
    unburned_sample = rng.choice(unburned_rows.size, size=burned_rows.size, replace=False)

    burned_df = pixel_frame(
        burned_rows,
        burned_cols,
        target=1,
        sample_type="burned",
        template_transform=template_transform,
        raster_arrays=raster_arrays,
        features=features,
    )
    unburned_df = pixel_frame(
        unburned_rows[unburned_sample],
        unburned_cols[unburned_sample],
        target=0,
        sample_type="unburned",
        template_transform=template_transform,
        raster_arrays=raster_arrays,
        features=features,
    )

    period_df = pd.concat([burned_df, unburned_df], ignore_index=True)
    period_df["period"] = period_name
    period_df["target_name"] = period_df["target"].map({0: "unburned", 1: "burned"})
    return period_df


def full_period_table(
    period_burned_mask,
    period_name,
    valid_mask,
    template_transform,
    raster_arrays,
    features,
):
    """Build a full valid-landscape pixel table for a fire period."""
    # Hold-out evaluation is not balanced. It keeps every valid pixel so metrics
    # reflect the real class imbalance in the later evaluation period.
    rows, cols = np.where(valid_mask)
    targets = period_burned_mask[rows, cols].astype(int)

    period_df = pixel_frame(
        rows,
        cols,
        target=0,
        sample_type="holdout_full",
        template_transform=template_transform,
        raster_arrays=raster_arrays,
        features=features,
    )
    period_df["target"] = targets
    period_df["sample_type"] = np.where(period_df["target"] == 1, "burned", "unburned")
    period_df["period"] = period_name
    period_df["target_name"] = period_df["target"].map({0: "unburned", 1: "burned"})
    return period_df
