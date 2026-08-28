#!/usr/bin/env python3

import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)

############################################################
# INPUT FILES
############################################################

TRAIN_LOG = "train.log"
GEN_RESULTS = "results/generation_results_best_1.csv"

############################################################
# Parse train.log
############################################################

train_pattern = re.compile(
    r"train\s+(\d+)\s+\|\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)"
)

valid_pattern = re.compile(
    r"valid\s+(\d+)\s+\|\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)"
)

epochs = []
train_loss = []
train_rce = []
train_prop = []
train_kl = []

valid_loss = []
valid_rce = []
valid_prop = []
valid_kl = []

has_train_log = os.path.isfile(TRAIN_LOG) and os.path.getsize(TRAIN_LOG) > 0

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
                valid_loss.append(float(m.group(2)))
                valid_rce.append(float(m.group(3)))
                valid_prop.append(float(m.group(4)))
                valid_kl.append(float(m.group(5)))
else:
    print(f"Warning: {TRAIN_LOG} is missing or empty — skipping training curve plots.")

############################################################
# Save metrics
############################################################

metrics = pd.DataFrame({
    "epoch": epochs,
    "train_loss": train_loss,
    "train_rce": train_rce,
    "train_kl": train_kl,
    "valid_loss": valid_loss,
    "valid_rce": valid_rce,
    "valid_kl": valid_kl
})

if len(metrics) > 0:
    metrics.to_csv("training_metrics.csv", index=False)

############################################################
# Plot curves
############################################################

def plot_curve(y1, y2, ylabel, outfile):

    plt.figure(figsize=(7,5))

    plt.plot(epochs, y1, lw=2, label="Train")
    plt.plot(epochs, y2, lw=2, label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)

    plt.legend()

    plt.tight_layout()

    plt.savefig(outfile, dpi=300)

    plt.close()


if len(metrics) > 0:
    plot_curve(train_loss,
               valid_loss,
               "Loss",
               "loss_curve.png")

    plot_curve(train_rce,
               valid_rce,
               "SMILES Reconstruction Loss",
               "rce_curve.png")

    plot_curve(train_kl,
               valid_kl,
               "KL Loss",
               "kl_curve.png")

############################################################
# Analyze generated molecules
############################################################

gen = pd.read_csv(GEN_RESULTS)

valid = gen[gen.validity == 1].copy()

validity = len(valid) / len(gen)

############################################################
# Property statistics
############################################################

def regression_stats(target, pred):

    rmse = np.sqrt(mean_squared_error(target, pred))
    mae = mean_absolute_error(target, pred)

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

def scatter(target,
            pred,
            xlabel,
            ylabel,
            outfile):

    plt.figure(figsize=(5,5))

    plt.scatter(target,
                pred,
                s=20,
                alpha=0.7)

    mn = min(target.min(), pred.min())
    mx = max(target.max(), pred.max())

    plt.plot([mn,mx],[mn,mx],'r--')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.tight_layout()

    plt.savefig(outfile,dpi=300)

    plt.close()


scatter(valid["condition(weight)"],
        valid["rdkit(weight)"],
        "Target MW",
        "Generated MW",
        "mw_scatter.png")

scatter(valid["condition(logP)"],
        valid["rdkit(logP)"],
        "Target LogP",
        "Generated LogP",
        "logp_scatter.png")

scatter(valid["condition(TPSA)"],
        valid["rdkit(TPSA)"],
        "Target TPSA",
        "Generated TPSA",
        "tpsa_scatter.png")

############################################################
# Text report
############################################################

has_training_metrics = len(metrics) > 0 and len(valid_loss) > 0

if has_training_metrics:
    best_epoch = metrics.loc[metrics.valid_loss.idxmin(), "epoch"]
    best_loss = metrics.valid_loss.min()
else:
    best_epoch = None
    best_loss = None

with open("training_report.txt","w") as f:

    f.write("=========================================\n")
    f.write("Transformer VAE Training Report\n")
    f.write("=========================================\n\n")

    if has_training_metrics:
        f.write(f"Epochs trained          : {len(metrics)}\n")
        f.write(f"Best epoch              : {best_epoch}\n")
        f.write(f"Best validation loss    : {best_loss:.4f}\n\n")

        f.write("-----------------------------------------\n")
        f.write("Final Training Metrics\n")
        f.write("-----------------------------------------\n")

        f.write(f"Train loss              : {train_loss[-1]:.4f}\n")
        f.write(f"Train RCE               : {train_rce[-1]:.4f}\n")
        f.write(f"Train KL                : {train_kl[-1]:.4f}\n\n")

        f.write(f"Validation loss         : {valid_loss[-1]:.4f}\n")
        f.write(f"Validation RCE          : {valid_rce[-1]:.4f}\n")
        f.write(f"Validation KL           : {valid_kl[-1]:.4f}\n\n")
    else:
        f.write("Training log not available (train.log missing or empty).\n")
        f.write("Re-run training with:  python train.py ... 2>&1 | tee train.log\n\n")

    f.write("-----------------------------------------\n")
    f.write("Generation Metrics\n")
    f.write("-----------------------------------------\n")

    f.write(f"Generated molecules     : {len(gen)}\n")
    f.write(f"Valid molecules         : {len(valid)}\n")
    f.write(f"Validity               : {validity:.3f}\n\n")

    f.write(f"MW RMSE                : {mw_rmse:.3f}\n")
    f.write(f"MW MAE                 : {mw_mae:.3f}\n\n")

    f.write(f"LogP RMSE              : {logp_rmse:.3f}\n")
    f.write(f"LogP MAE               : {logp_mae:.3f}\n\n")

    f.write(f"TPSA RMSE              : {tpsa_rmse:.3f}\n")
    f.write(f"TPSA MAE               : {tpsa_mae:.3f}\n")

print("Analysis complete.")
print("Created:")
if has_training_metrics:
    print("  training_metrics.csv")
    print("  loss_curve.png")
    print("  rce_curve.png")
    print("  kl_curve.png")
print("  training_report.txt")
print("  mw_scatter.png")
print("  logp_scatter.png")
print("  tpsa_scatter.png")
