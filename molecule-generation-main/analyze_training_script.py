#!/usr/bin/env python3

import re
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)


############################################################
# Command-line arguments
############################################################

parser = argparse.ArgumentParser(
    description="Analyze Transformer VAE training and generation results."
)

parser.add_argument(
    "train_log",
    help="Training log file, e.g. train_lat1_dec0.log"
)

parser.add_argument(
    "--gen-results",
    default="results/generation_results_best_1.csv",
    help=(
        "Generation results CSV "
        "(default: results/generation_results_best_1.csv)"
    )
)

args = parser.parse_args()


############################################################
# INPUT FILES
############################################################

TRAIN_LOG = args.train_log
GEN_RESULTS = args.gen_results


############################################################
# Determine experiment name and output directory
############################################################

# Example:
# train_lat1_dec0.log -> lat1_dec0

experiment = os.path.splitext(
    os.path.basename(TRAIN_LOG)
)[0]

if experiment.startswith("train_"):
    experiment = experiment[len("train_"):]

OUTPUT_DIR = os.path.join("analysis", experiment)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=========================================")
print("Training analysis")
print("=========================================")
print(f"Training log       : {TRAIN_LOG}")
print(f"Generation results : {GEN_RESULTS}")
print(f"Experiment         : {experiment}")
print(f"Output directory   : {OUTPUT_DIR}")
print()


############################################################
# Check training log
############################################################

if not os.path.isfile(TRAIN_LOG):
    raise FileNotFoundError(
        f"Training log does not exist: {TRAIN_LOG}"
    )


############################################################
# Parse train.log
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


epochs = []

train_loss = []
train_rce = []
train_prop = []
train_kl = []

valid_epochs = []

valid_loss = []
valid_rce = []
valid_prop = []
valid_kl = []


has_train_log = (
    os.path.isfile(TRAIN_LOG)
    and os.path.getsize(TRAIN_LOG) > 0
)


if has_train_log:

    with open(TRAIN_LOG) as f:

        for line in f:

            m = train_pattern.search(line)

            if m:

                epochs.append(int(m.group(1)))

                train_loss.append(float(m.group(2)))
                train_rce.append(float(m.group(3)))
                train_prop.append(float(m.group(4)))
                train_kl.append(float(m.group(5)))

            m = valid_pattern.search(line)

            if m:

                valid_epochs.append(int(m.group(1)))

                valid_loss.append(float(m.group(2)))
                valid_rce.append(float(m.group(3)))
                valid_prop.append(float(m.group(4)))
                valid_kl.append(float(m.group(5)))

else:

    print(
        f"Warning: {TRAIN_LOG} is empty — "
        "skipping training curve plots."
    )


############################################################
# Check parsed training data
############################################################

print(f"Parsed training epochs   : {len(epochs)}")
print(f"Parsed validation epochs : {len(valid_epochs)}")
print()


if len(epochs) != len(valid_epochs):

    print(
        "Warning: number of training and validation epochs "
        "does not match."
    )


############################################################
# Save metrics
############################################################

# Only pair epochs that have both training and validation results.

n_metrics = min(
    len(epochs),
    len(valid_epochs),
    len(train_loss),
    len(valid_loss)
)


if n_metrics > 0:

    metrics = pd.DataFrame({

        "epoch": epochs[:n_metrics],

        "train_loss": train_loss[:n_metrics],
        "train_rce": train_rce[:n_metrics],
        "train_prop": train_prop[:n_metrics],
        "train_kl": train_kl[:n_metrics],

        "valid_loss": valid_loss[:n_metrics],
        "valid_rce": valid_rce[:n_metrics],
        "valid_prop": valid_prop[:n_metrics],
        "valid_kl": valid_kl[:n_metrics]
    })

else:

    metrics = pd.DataFrame()


if len(metrics) > 0:

    metrics_file = os.path.join(
        OUTPUT_DIR,
        "training_metrics.csv"
    )

    metrics.to_csv(
        metrics_file,
        index=False
    )


############################################################
# Plot curves
############################################################

def plot_curve(
    x1,
    y1,
    x2,
    y2,
    ylabel,
    outfile
):

    plt.figure(figsize=(7, 5))

    plt.plot(
        x1,
        y1,
        lw=2,
        label="Train"
    )

    plt.plot(
        x2,
        y2,
        lw=2,
        label="Validation"
    )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        outfile,
        dpi=300
    )

    plt.close()


if len(metrics) > 0:

    plot_curve(
        epochs,
        train_loss,
        valid_epochs,
        valid_loss,
        "Loss",
        os.path.join(
            OUTPUT_DIR,
            "loss_curve.png"
        )
    )

    plot_curve(
        epochs,
        train_rce,
        valid_epochs,
        valid_rce,
        "SMILES Reconstruction Loss",
        os.path.join(
            OUTPUT_DIR,
            "rce_curve.png"
        )
    )

    plot_curve(
        epochs,
        train_kl,
        valid_epochs,
        valid_kl,
        "KL Loss",
        os.path.join(
            OUTPUT_DIR,
            "kl_curve.png"
        )
    )


############################################################
# Analyze generated molecules
############################################################

if not os.path.isfile(GEN_RESULTS):

    raise FileNotFoundError(
        f"Generation results file does not exist: "
        f"{GEN_RESULTS}"
    )


gen = pd.read_csv(GEN_RESULTS)


############################################################
# Check required columns
############################################################

required_columns = [

    "validity",

    "condition(weight)",
    "rdkit(weight)",

    "condition(logP)",
    "rdkit(logP)",

    "condition(TPSA)",
    "rdkit(TPSA)"
]


missing_columns = [

    col
    for col in required_columns
    if col not in gen.columns
]


if missing_columns:

    raise ValueError(
        "Generation results file is missing columns: "
        + ", ".join(missing_columns)
    )


############################################################
# Valid molecules
############################################################

valid = gen[
    gen["validity"] == 1
].copy()


if len(gen) > 0:

    validity = len(valid) / len(gen)

else:

    validity = 0.0


if len(valid) == 0:

    raise ValueError(
        "No valid molecules were found in "
        f"{GEN_RESULTS}"
    )


############################################################
# Property statistics
############################################################

def regression_stats(target, pred):

    # Remove rows containing NaN values.

    mask = (
        target.notna()
        & pred.notna()
    )

    target_clean = target[mask]
    pred_clean = pred[mask]

    if len(target_clean) == 0:

        return np.nan, np.nan

    rmse = np.sqrt(
        mean_squared_error(
            target_clean,
            pred_clean
        )
    )

    mae = mean_absolute_error(
        target_clean,
        pred_clean
    )

    return rmse, mae


mw_rmse, mw_mae = regression_stats(
    valid["condition(weight)"],
    valid["rdkit(weight)"]
)


logp_rmse, logp_mae = regression_stats(
    valid["condition(logP)"],
    valid["rdkit(logP)"]
)


tpsa_rmse, tpsa_mae = regression_stats(
    valid["condition(TPSA)"],
    valid["rdkit(TPSA)"]
)


############################################################
# Scatter plots
############################################################

def scatter(
    target,
    pred,
    xlabel,
    ylabel,
    outfile
):

    # Remove NaN values.

    mask = (
        target.notna()
        & pred.notna()
    )

    target = target[mask]
    pred = pred[mask]

    if len(target) == 0:
        print(
            f"Warning: no valid data for {outfile}"
        )
        return

    plt.figure(
        figsize=(5, 5)
    )

    plt.scatter(
        target,
        pred,
        s=20,
        alpha=0.7
    )

    mn = min(
        target.min(),
        pred.min()
    )

    mx = max(
        target.max(),
        pred.max()
    )

    plt.plot(
        [mn, mx],
        [mn, mx],
        "r--"
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.tight_layout()

    plt.savefig(
        outfile,
        dpi=300
    )

    plt.close()


scatter(
    valid["condition(weight)"],
    valid["rdkit(weight)"],
    "Target MW",
    "Generated MW",
    os.path.join(
        OUTPUT_DIR,
        "mw_scatter.png"
    )
)


scatter(
    valid["condition(logP)"],
    valid["rdkit(logP)"],
    "Target LogP",
    "Generated LogP",
    os.path.join(
        OUTPUT_DIR,
        "logp_scatter.png"
    )
)


scatter(
    valid["condition(TPSA)"],
    valid["rdkit(TPSA)"],
    "Target TPSA",
    "Generated TPSA",
    os.path.join(
        OUTPUT_DIR,
        "tpsa_scatter.png"
    )
)


############################################################
# Determine best epoch
############################################################

has_training_metrics = (
    len(metrics) > 0
    and len(valid_loss) > 0
)


if has_training_metrics:

    best_idx = metrics["valid_loss"].idxmin()

    best_epoch = int(
        metrics.loc[
            best_idx,
            "epoch"
        ]
    )

    best_loss = metrics.loc[
        best_idx,
        "valid_loss"
    ]

else:

    best_epoch = None
    best_loss = None


############################################################
# Text report
############################################################

report_file = os.path.join(
    OUTPUT_DIR,
    "training_report.txt"
)


with open(
    report_file,
    "w"
) as f:

    f.write(
        "=========================================\n"
    )

    f.write(
        "Transformer VAE Training Report\n"
    )

    f.write(
        "=========================================\n\n"
    )

    f.write(
        f"Experiment               : "
        f"{experiment}\n"
    )

    f.write(
        f"Training log             : "
        f"{TRAIN_LOG}\n"
    )

    f.write(
        f"Generation results       : "
        f"{GEN_RESULTS}\n\n"
    )


    ########################################################
    # Training metrics
    ########################################################

    if has_training_metrics:

        f.write(
            "-----------------------------------------\n"
        )

        f.write(
            "Training Summary\n"
        )

        f.write(
            "-----------------------------------------\n"
        )

        f.write(
            f"Epochs trained           : "
            f"{len(metrics)}\n"
        )

        f.write(
            f"Best epoch               : "
            f"{best_epoch}\n"
        )

        f.write(
            f"Best validation loss     : "
            f"{best_loss:.4f}\n\n"
        )


        f.write(
            "-----------------------------------------\n"
        )

        f.write(
            "Final Training Metrics\n"
        )

        f.write(
            "-----------------------------------------\n"
        )

        f.write(
            f"Train loss               : "
            f"{train_loss[-1]:.4f}\n"
        )

        f.write(
            f"Train RCE                : "
            f"{train_rce[-1]:.4f}\n"
        )

        f.write(
            f"Train property loss      : "
            f"{train_prop[-1]:.4f}\n"
        )

        f.write(
            f"Train KL                 : "
            f"{train_kl[-1]:.4f}\n\n"
        )


        f.write(
            f"Validation loss          : "
            f"{valid_loss[-1]:.4f}\n"
        )

        f.write(
            f"Validation RCE           : "
            f"{valid_rce[-1]:.4f}\n"
        )

        f.write(
            f"Validation property loss : "
            f"{valid_prop[-1]:.4f}\n"
        )

        f.write(
            f"Validation KL            : "
            f"{valid_kl[-1]:.4f}\n\n"
        )

    else:

        f.write(
            "Training metrics were not available.\n\n"
        )


    ########################################################
    # Generation metrics
    ########################################################

    f.write(
        "-----------------------------------------\n"
    )

    f.write(
        "Generation Metrics\n"
    )

    f.write(
        "-----------------------------------------\n"
    )

    f.write(
        f"Generated molecules      : "
        f"{len(gen)}\n"
    )

    f.write(
        f"Valid molecules          : "
        f"{len(valid)}\n"
    )

    f.write(
        f"Validity                 : "
        f"{validity:.3f}\n\n"
    )


    f.write(
        f"MW RMSE                  : "
        f"{mw_rmse:.3f}\n"
    )

    f.write(
        f"MW MAE                   : "
        f"{mw_mae:.3f}\n\n"
    )


    f.write(
        f"LogP RMSE                : "
        f"{logp_rmse:.3f}\n"
    )

    f.write(
        f"LogP MAE                 : "
        f"{logp_mae:.3f}\n\n"
    )


    f.write(
        f"TPSA RMSE                : "
        f"{tpsa_rmse:.3f}\n"
    )

    f.write(
        f"TPSA MAE                 : "
        f"{tpsa_mae:.3f}\n"
    )


############################################################
# Final output
############################################################

print("=========================================")
print("Analysis complete.")
print("=========================================")

print(f"Experiment : {experiment}")
print(f"Output     : {OUTPUT_DIR}")
print()

print("Created:")

if has_training_metrics:

    print(
        f"  {os.path.join(OUTPUT_DIR, 'training_metrics.csv')}"
    )

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
    f"  {os.path.join(OUTPUT_DIR, 'training_report.txt')}"
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

