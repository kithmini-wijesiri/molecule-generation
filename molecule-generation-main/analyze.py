#!/usr/bin/env python3

import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


############################################################
# Command-line arguments
############################################################

parser = argparse.ArgumentParser(
    description=(
        "Analyze Transformer VAE training and "
        "conditional molecular generation."
    )
)

parser.add_argument(
    "train_log",
    help="Training log, e.g. train_lat1_dec0.log"
)

parser.add_argument(
    "generation_results",
    help=(
        "Generation results CSV, e.g. "
        "results/generation_results_lat1_dec0_1.csv"
    )
)

parser.add_argument(
    "--output-dir",
    default="analysis",
    help="Parent output directory (default: analysis)"
)

args = parser.parse_args()

TRAIN_LOG = args.train_log
GEN_RESULTS = args.generation_results


############################################################
# Check input files
############################################################

if not os.path.isfile(TRAIN_LOG):
    raise FileNotFoundError(
        f"Training log not found: {TRAIN_LOG}"
    )

if not os.path.isfile(GEN_RESULTS):
    raise FileNotFoundError(
        f"Generation results not found: {GEN_RESULTS}"
    )


############################################################
# Determine experiment name
#
# train_lat1_dec0.log -> lat1_dec0
############################################################

experiment = os.path.splitext(
    os.path.basename(TRAIN_LOG)
)[0]

if experiment.startswith("train_"):
    experiment = experiment[len("train_"):]


############################################################
# Output directory
############################################################

OUTPUT_DIR = os.path.join(
    args.output_dir,
    experiment
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


print()
print("=" * 60)
print("Transformer VAE Experiment Analysis")
print("=" * 60)
print(f"Experiment         : {experiment}")
print(f"Training log       : {TRAIN_LOG}")
print(f"Generation results : {GEN_RESULTS}")
print(f"Output directory   : {OUTPUT_DIR}")
print()


############################################################
# Parse training log
#
# Expected lines:
#
# train 0 | total_loss rce property_loss kl_loss
# valid 0 | total_loss rce property_loss kl_loss
############################################################

train_pattern = re.compile(
    r"train\s+(\d+)\s+\|\s+"
    r"([\d.eE+-]+)\s+"
    r"([\d.eE+-]+)\s+"
    r"([\d.eE+-]+)\s+"
    r"([\d.eE+-]+)"
)

valid_pattern = re.compile(
    r"valid\s+(\d+)\s+\|\s+"
    r"([\d.eE+-]+)\s+"
    r"([\d.eE+-]+)\s+"
    r"([\d.eE+-]+)\s+"
    r"([\d.eE+-]+)"
)


train_records = []
valid_records = []


with open(
    TRAIN_LOG,
    "r",
    encoding="utf-8",
    errors="ignore"
) as f:

    for line in f:

        ####################################################
        # Training
        ####################################################

        match = train_pattern.search(line)

        if match:

            train_records.append(
                {
                    "epoch": int(match.group(1)),
                    "train_loss": float(match.group(2)),
                    "train_rce": float(match.group(3)),
                    "train_property": float(match.group(4)),
                    "train_kl": float(match.group(5)),
                }
            )

        ####################################################
        # Validation
        ####################################################

        match = valid_pattern.search(line)

        if match:

            valid_records.append(
                {
                    "epoch": int(match.group(1)),
                    "valid_loss": float(match.group(2)),
                    "valid_rce": float(match.group(3)),
                    "valid_property": float(match.group(4)),
                    "valid_kl": float(match.group(5)),
                }
            )


############################################################
# Check parsing
############################################################

if len(train_records) == 0:

    raise ValueError(
        f"No training records found in {TRAIN_LOG}"
    )


if len(valid_records) == 0:

    raise ValueError(
        f"No validation records found in {TRAIN_LOG}"
    )


train_df = pd.DataFrame(
    train_records
)

valid_df = pd.DataFrame(
    valid_records
)


############################################################
# Merge train and validation data by epoch
############################################################

metrics = pd.merge(
    train_df,
    valid_df,
    on="epoch",
    how="outer"
)

metrics = metrics.sort_values(
    "epoch"
).reset_index(
    drop=True
)


############################################################
# Save training metrics
############################################################

training_metrics_file = os.path.join(
    OUTPUT_DIR,
    "training_metrics.csv"
)

metrics.to_csv(
    training_metrics_file,
    index=False
)


############################################################
# Best validation epoch
############################################################

best_idx = valid_df[
    "valid_loss"
].idxmin()


best_epoch = int(
    valid_df.loc[
        best_idx,
        "epoch"
    ]
)


best_valid_loss = float(
    valid_df.loc[
        best_idx,
        "valid_loss"
    ]
)


############################################################
# Training curve plotting function
############################################################

def plot_training_curve(
    train_x,
    train_y,
    valid_x,
    valid_y,
    ylabel,
    outfile,
    log_scale=False
):

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        train_x,
        train_y,
        marker="o",
        linewidth=2,
        label="Train"
    )

    plt.plot(
        valid_x,
        valid_y,
        marker="o",
        linewidth=2,
        label="Validation"
    )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)

    if log_scale:
        plt.yscale("log")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


############################################################
# Total loss curve
############################################################

plot_training_curve(
    train_df["epoch"],
    train_df["train_loss"],
    valid_df["epoch"],
    valid_df["valid_loss"],
    "Loss",
    os.path.join(
        OUTPUT_DIR,
        "loss_curve.png"
    )
)


############################################################
# Reconstruction loss curve
############################################################

plot_training_curve(
    train_df["epoch"],
    train_df["train_rce"],
    valid_df["epoch"],
    valid_df["valid_rce"],
    "SMILES Reconstruction Loss",
    os.path.join(
        OUTPUT_DIR,
        "rce_curve.png"
    )
)


############################################################
# KL loss curve
#
# Log scale is useful because your epoch-0 training KL
# can be much larger than later epochs.
############################################################

plot_training_curve(
    train_df["epoch"],
    train_df["train_kl"],
    valid_df["epoch"],
    valid_df["valid_kl"],
    "KL Loss",
    os.path.join(
        OUTPUT_DIR,
        "kl_curve.png"
    ),
    log_scale=True
)


############################################################
# Read generation results
############################################################

gen = pd.read_csv(
    GEN_RESULTS
)


############################################################
# Required columns
############################################################

required_columns = [

    "validity",

    "condition(weight)",
    "rdkit(weight)",

    "condition(logP)",
    "rdkit(logP)",

    "condition(TPSA)",
    "rdkit(TPSA)",
]


missing_columns = [

    col
    for col in required_columns
    if col not in gen.columns
]


if missing_columns:

    raise ValueError(
        "Generation results are missing columns:\n"
        + "\n".join(missing_columns)
    )


############################################################
# Number generated and validity
############################################################

n_generated = len(gen)


if n_generated == 0:

    raise ValueError(
        "Generation results CSV contains no molecules."
    )


valid = gen[
    gen["validity"] == 1
].copy()


n_valid = len(valid)


validity = (
    n_valid / n_generated
)


if n_valid == 0:

    raise ValueError(
        "No valid molecules found."
    )


############################################################
# Regression statistics
############################################################

def regression_stats(
    dataframe,
    target_column,
    generated_column
):

    data = dataframe[
        [
            target_column,
            generated_column
        ]
    ].copy()


    ########################################################
    # Convert values to numeric
    ########################################################

    data[target_column] = pd.to_numeric(
        data[target_column],
        errors="coerce"
    )

    data[generated_column] = pd.to_numeric(
        data[generated_column],
        errors="coerce"
    )


    ########################################################
    # Remove invalid values
    ########################################################

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    data = data.dropna()


    if len(data) == 0:

        return {
            "target": np.array([]),
            "generated": np.array([]),
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "n": 0,
        }


    target = data[
        target_column
    ].to_numpy()

    generated = data[
        generated_column
    ].to_numpy()


    ########################################################
    # RMSE
    ########################################################

    rmse = np.sqrt(
        mean_squared_error(
            target,
            generated
        )
    )


    ########################################################
    # MAE
    ########################################################

    mae = mean_absolute_error(
        target,
        generated
    )


    ########################################################
    # R squared
    ########################################################

    if len(data) >= 2:

        r2 = r2_score(
            target,
            generated
        )

    else:

        r2 = np.nan


    return {
        "target": target,
        "generated": generated,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "n": len(data),
    }


############################################################
# Calculate property statistics
############################################################

mw_stats = regression_stats(
    valid,
    "condition(weight)",
    "rdkit(weight)"
)


logp_stats = regression_stats(
    valid,
    "condition(logP)",
    "rdkit(logP)"
)


tpsa_stats = regression_stats(
    valid,
    "condition(TPSA)",
    "rdkit(TPSA)"
)


############################################################
# Scatter plotting function
############################################################

def scatter_plot(
    stats,
    xlabel,
    ylabel,
    outfile
):

    target = stats["target"]
    generated = stats["generated"]


    if len(target) == 0:

        print(
            f"Warning: no data available for {outfile}"
        )

        return


    plt.figure(
        figsize=(6, 6)
    )


    ########################################################
    # Scatter points
    ########################################################

    plt.scatter(
        target,
        generated,
        s=35,
        alpha=0.65
    )


    ########################################################
    # Identity line: y = x
    ########################################################

    mn = min(
        np.min(target),
        np.min(generated)
    )

    mx = max(
        np.max(target),
        np.max(generated)
    )


    padding = (
        (mx - mn) * 0.05
        if mx > mn
        else 1.0
    )


    lower = mn - padding
    upper = mx + padding


    plt.plot(
        [lower, upper],
        [lower, upper],
        "r--",
        linewidth=2,
        label="Ideal: y = x"
    )


    ########################################################
    # Same limits for X and Y
    ########################################################

    plt.xlim(
        lower,
        upper
    )

    plt.ylim(
        lower,
        upper
    )


    ########################################################
    # Labels
    ########################################################

    plt.xlabel(
        xlabel
    )

    plt.ylabel(
        ylabel
    )


    ########################################################
    # Statistics annotation
    ########################################################

    stats_text = (
        f"R² = {stats['r2']:.3f}\n"
        f"RMSE = {stats['rmse']:.3f}\n"
        f"MAE = {stats['mae']:.3f}\n"
        f"n = {stats['n']}"
    )


    plt.text(
        0.05,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            alpha=0.8
        )
    )


    plt.legend(
        loc="lower right"
    )


    plt.tight_layout()


    plt.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


############################################################
# MW scatter
############################################################

scatter_plot(
    mw_stats,
    "Target MW",
    "Generated MW",
    os.path.join(
        OUTPUT_DIR,
        "mw_scatter.png"
    )
)


############################################################
# LogP scatter
############################################################

scatter_plot(
    logp_stats,
    "Target LogP",
    "Generated LogP",
    os.path.join(
        OUTPUT_DIR,
        "logp_scatter.png"
    )
)


############################################################
# TPSA scatter
############################################################

scatter_plot(
    tpsa_stats,
    "Target TPSA",
    "Generated TPSA",
    os.path.join(
        OUTPUT_DIR,
        "tpsa_scatter.png"
    )
)


############################################################
# Create summary
############################################################

summary = {

    "Experiment": experiment,

    "Generated": n_generated,
    "Valid": n_valid,
    "Validity": validity,

    "MW RMSE": mw_stats["rmse"],
    "MW MAE": mw_stats["mae"],
    "MW R2": mw_stats["r2"],

    "LogP RMSE": logp_stats["rmse"],
    "LogP MAE": logp_stats["mae"],
    "LogP R2": logp_stats["r2"],

    "TPSA RMSE": tpsa_stats["rmse"],
    "TPSA MAE": tpsa_stats["mae"],
    "TPSA R2": tpsa_stats["r2"],

    "Best valid loss": best_valid_loss,
    "Best epoch": best_epoch,
}


############################################################
# Save summary CSV
############################################################

summary_file = os.path.join(
    OUTPUT_DIR,
    "summary.csv"
)


summary_df = pd.DataFrame(
    [summary]
)


summary_df.to_csv(
    summary_file,
    index=False
)


############################################################
# Print summary
############################################################

print()
print("=" * 60)
print(f"Experiment: {experiment}")
print("=" * 60)

print(
    f"{'Generated molecules':<25}"
    f"{n_generated}"
)

print(
    f"{'Valid molecules':<25}"
    f"{n_valid}"
)

print(
    f"{'Validity':<25}"
    f"{validity:.3f}"
)

print()


############################################################
# MW
############################################################

print(
    f"{'MW RMSE':<25}"
    f"{mw_stats['rmse']:.3f}"
)

print(
    f"{'MW MAE':<25}"
    f"{mw_stats['mae']:.3f}"
)

print(
    f"{'MW R2':<25}"
    f"{mw_stats['r2']:.3f}"
)

print()


############################################################
# LogP
############################################################

print(
    f"{'LogP RMSE':<25}"
    f"{logp_stats['rmse']:.3f}"
)

print(
    f"{'LogP MAE':<25}"
    f"{logp_stats['mae']:.3f}"
)

print(
    f"{'LogP R2':<25}"
    f"{logp_stats['r2']:.3f}"
)

print()


############################################################
# TPSA
############################################################

print(
    f"{'TPSA RMSE':<25}"
    f"{tpsa_stats['rmse']:.3f}"
)

print(
    f"{'TPSA MAE':<25}"
    f"{tpsa_stats['mae']:.3f}"
)

print(
    f"{'TPSA R2':<25}"
    f"{tpsa_stats['r2']:.3f}"
)

print()


############################################################
# Best validation result
############################################################

print(
    f"{'Best valid loss':<25}"
    f"{best_valid_loss:.3f}"
)

print(
    f"{'Best epoch':<25}"
    f"{best_epoch}"
)


print("=" * 60)


############################################################
# Output files
############################################################

print()
print("Created:")
print(f"  {summary_file}")
print(f"  {training_metrics_file}")

print(
    f"  {os.path.join(OUTPUT_DIR, 'loss_curve.png')}"
)

print(
    f"  {os.path.join(OUTPUT_DIR, 'rce_curve.png')}"
)

print(
    f"  {os.path.join(OUTPUT_DIR, 'kl_curve.png')}"
)

print(
    f"  {os.path.join(OUTPUT_DIR, 'mw_scatter.png')}"
)

print(
    f"  {os.path.join(OUTPUT_DIR, 'logp_scatter.png')}"
)

print(
    f"  {os.path.join(OUTPUT_DIR, 'tpsa_scatter.png')}"
)

print()
