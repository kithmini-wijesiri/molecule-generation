# Conditional Transformer VAE for Molecular Generation

## Introduction

This repository implements a **conditional Transformer Variational Autoencoder (VAE)** for de novo molecular generation.

Molecules are represented as **SMILES strings**, and the model is trained to reconstruct molecular structures while learning a continuous latent representation of chemical space.

In addition to molecular structure, the model can use physicochemical properties as conditioning variables. The current implementation uses three molecular properties:

- Molecular weight (MW)
- LogP
- Topological polar surface area (TPSA)

The goal is to generate valid molecules whose calculated properties are close to user-specified target values.

For example, the model can be provided with a target property vector:

```text
MW   = 400
LogP = 3.0
TPSA = 70
```

and generate a molecular structure that approximately satisfies these conditions.

---

## Model Architecture

The model combines a **Transformer encoder-decoder architecture** with a **variational latent space**.

A simplified representation of the model is:

```text
                    Molecular properties
                    [MW, LogP, TPSA]
                           |
                           |
                      Conditioning
                           |
                           v
SMILES ---> Transformer Encoder ---> Latent Space (z)
                                      |
                                      |
                                      v
                              Transformer Decoder
                                      |
                                      v
                               Generated SMILES
```

The encoder converts an input SMILES sequence into a molecular representation.

The VAE component maps this representation into a continuous latent space defined by:

```text
mu
log_variance
```

A latent vector is sampled using the VAE reparameterization procedure:

```text
z = mu + sigma * epsilon
```

where `epsilon` is sampled from a standard normal distribution.

The Transformer decoder then uses the latent representation to reconstruct or generate a SMILES sequence.

---

## Property Conditioning

Property information can be introduced at different locations in the architecture.

Two options are currently available:

```text
-use_cond2lat
-use_cond2dec
```

### Latent conditioning

```text
-use_cond2lat True
```

provides the molecular property information to the latent representation.

Conceptually:

```text
SMILES
  |
  v
Transformer Encoder
  |
  +-------- Molecular Properties
  |          [MW, LogP, TPSA]
  |
  v
Latent Representation
  |
  v
Transformer Decoder
  |
  v
Generated SMILES
```

### Decoder conditioning

```text
-use_cond2dec True
```

provides the property information to the decoder during molecular reconstruction/generation.

Conceptually:

```text
SMILES
  |
  v
Transformer Encoder
  |
  v
Latent Representation
  |
  +-------- Molecular Properties
  |          [MW, LogP, TPSA]
  |
  v
Transformer Decoder
  |
  v
Generated SMILES
```

These options allow different conditioning strategies to be compared.

For example:

| Experiment | Latent conditioning | Decoder conditioning |
|---|---|---|
| `lat0_dec0` | No | No |
| `lat1_dec0` | Yes | No |
| `lat0_dec1` | No | Yes |
| `lat1_dec1` | Yes | Yes |

---

## Data

The model is trained using molecular SMILES and associated molecular properties.

The input dataset contains:

```text
smiles
weight
logp
TPSA
```

The data are divided into training and validation sets.

The molecular properties are normalized before training using a robust scaler.

The fitted scaler is saved as:

```text
weights/robust_scaler.pkl
```

and is used during molecular generation to ensure that conditioning values are transformed consistently with the training data.

---

## Training

An example training command is:

```bash
/sb/bpbrownlab/shared/apps/miniconda3/envs/torch_24/bin/torchrun \
    --nproc_per_node=1 \
    train.py \
    -batch_size 64 \
    -epochs 10 \
    -use_cond2lat True \
    -use_cond2dec False \
    2>&1 | tee train_lat1_dec0.log
```

This configuration corresponds to:

```text
Latent conditioning  = True
Decoder conditioning = False
```

and is referred to as:

```text
lat1_dec0
```

The training log is saved as:

```text
train_lat1_dec0.log
```

and checkpoints are stored in:

```text
weights/lat1_dec0/
```

The checkpoint with the lowest validation loss is saved as:

```text
weights/lat1_dec0/best.pt
```

---

## Training Objective

The model is optimized using a combination of:

1. **SMILES reconstruction loss**

   Measures how accurately the decoder reconstructs the target molecular SMILES.

2. **Property reconstruction loss**

   Used when the corresponding property-prediction component is enabled.

3. **KL divergence**

   Regularizes the latent distribution toward a standard normal distribution.

The general VAE objective is:

```text
Total Loss =
    Reconstruction Loss
    + Property Loss
    + beta * KL Divergence
```

KL annealing is used during training so that the contribution of the KL term gradually increases.

For example:

```text
Epoch 0 : beta = 0.02
Epoch 1 : beta = 0.04
Epoch 2 : beta = 0.06
...
Epoch 9 : beta = 0.20
```

This allows the model to initially focus on learning molecular reconstruction before stronger latent-space regularization is applied.

---

## Molecular Generation

After training, molecules can be generated using the trained model.

For conditional generation, target molecular properties are supplied to the model.

Generation results are stored in files such as:

```text
results/generation_results_lat1_dec0_1.csv
```

The output contains both the requested properties and properties calculated from the generated molecule using RDKit.

Example columns:

```text
mol
validity

condition(weight)
condition(logP)
condition(TPSA)

rdkit(weight)
rdkit(logP)
rdkit(TPSA)

rdkit(QED)
```

This makes it possible to directly compare:

```text
Target property
       vs
Generated-molecule property
```

---

## Model Evaluation

Training and generation results can be analyzed using:

```bash
python analyze_experiment.py \
    train_lat1_dec0.log \
    results/generation_results_lat1_dec0_1.csv
```

Results are written to:

```text
analysis/lat1_dec0/
```

including:

```text
summary.csv
training_metrics.csv

loss_curve.png
rce_curve.png
kl_curve.png

mw_scatter.png
logp_scatter.png
tpsa_scatter.png
```

---

## Evaluation Metrics

Generated molecules are evaluated using:

- Molecular validity
- RMSE
- MAE
- R²

Property statistics are calculated using valid generated molecules.

The target property values are compared against properties calculated from the generated structures using RDKit.

---

## Example: `lat1_dec0`

For the `lat1_dec0` model:

```text
Generated molecules      100
Valid molecules           84
Validity                0.840

MW RMSE                 53.413
MW MAE                  23.055
MW R²                    0.818

LogP RMSE                0.701
LogP MAE                 0.477
LogP R²                  0.782

TPSA RMSE               10.306
TPSA MAE                 6.537
TPSA R²                  0.950

Best validation loss    31.527
Best epoch                   8
```

For this experiment, TPSA shows the strongest agreement between requested and generated properties, followed by molecular weight and LogP.

---

## Project Structure

```text
molecule-generation-main/
│
├── train.py
├── inference.py
├── analyze_experiment.py
├── args.py
├── losses.py
├── data_loader.py
├── tokenizer.py
├── utils.py
│
├── models/
│   └── transformer.py
│
├── weights/
│   ├── lat1_dec0/
│   │   └── best.pt
│   └── ...
│
├── results/
│   ├── generation_results_lat1_dec0_1.csv
│   └── ...
│
└── analysis/
    ├── lat1_dec0/
    │   ├── summary.csv
    │   ├── training_metrics.csv
    │   ├── loss_curve.png
    │   ├── rce_curve.png
    │   ├── kl_curve.png
    │   ├── mw_scatter.png
    │   ├── logp_scatter.png
    │   └── tpsa_scatter.png
    └── ...
```

---

## Summary

This framework provides a way to investigate how molecular property information can be incorporated into a Transformer VAE for conditional molecular generation.

The main workflow is:

```text
SMILES + Molecular Properties
            |
            v
       Model Training
            |
            v
      Latent Chemical Space
            |
            v
   Conditional Generation
            |
            v
       Generated SMILES
            |
            v
      RDKit Evaluation
            |
            v
 Validity + Property Accuracy
```

Different latent and decoder conditioning strategies can be trained and compared to determine how the location of property conditioning affects molecular validity and control over generated physicochemical properties.
