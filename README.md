# TGLR

Implementation of **Temporally Guided Latent Refinement (TGLR)** for the manuscript
*Latent Distribution Modeling for Source Localization under Temporal Observation Mismatch*.

TGLR addresses source localization when temporal propagation sequences are
available during offline training, while inference relies on a single observed
snapshot. Temporal dynamics guide residual refinement of latent distributions
during training. Latent diffusion regularization learns distributional structure,
and test-time adaptation uses the trained diffusion model to refine snapshot-based
representations during inference.

## Repository structure

| File or directory | Purpose |
| --- | --- |
| `main.py` | Training, checkpoint evaluation, and dataset statistics |
| `args.py` | Default model and experiment configuration |
| `model.py` | TGLR, ablation variants, GLAD, and GLCFGD |
| `train.py` | Training loops and representation collection |
| `eval.py` | Graph-level evaluation metrics |
| `data.py` | NetworkX-to-PyTorch-Geometric conversion and node features |
| `create_graphs.py` | Propagation graph loading and preprocessing |
| `propagation_model.py` | SI, SIR, and IC simulation |
| `node_feature.py` | Structural and optional user-profile feature extraction |
| `visual.py` | Latent representation visualization |
| `static.py` | Structural statistics for the real-world datasets |
| `numpy_data_conv.py` | Optional legacy pickle export for external SIDSL experiments |
| `data/saved_graphs/` | Preprocessed graph caches and dataset availability information |
| `model_saves/checkpoints/` | Provided best-validation-F1 checkpoints |

## Installation

Use Python 3.10 or later. The release checks use Python 3.12 and the package
versions in `requirements.txt`; these are not a record of the original training
environment.

Create and activate an environment, for example:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
```

For CPU execution:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For GPU execution, install a PyTorch 2.6.0 CUDA build compatible with your system
using the [official PyTorch instructions](https://pytorch.org/get-started/previous-versions/),
then install `requirements.txt`. PyTorch Geometric's basic installation is sufficient
for the GAT operations used here; see its
[installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

Run the commands below from the repository root.

## Datasets

The small Karate and Jazz caches are included in this repository. The remaining
eight preprocessed graph caches are available in **[TGLR Datasets v1.0](https://github.com/DancinPuppet/TGLR/releases/tag/datasets-v1.0)**:

**[Download graphs.zip](https://github.com/DancinPuppet/TGLR/releases/download/datasets-v1.0/graphs.zip)** (approximately 315 MiB compressed; 2.4 GB extracted).

The archive contains Cora-ML and Facebook caches under SI and SIR, plus Twitter15,
Twitter16, Twitter25, and Weibo caches. To install them:

1. Download `graphs.zip` from the link above or the release's **Assets** section.
   GitHub's automatically generated **Source code** archives do not contain these datasets.
2. Extract the eight `.pkl` files directly into `data/saved_graphs/`.
   Do not leave them inside an additional `graphs/` subdirectory.
3. Keep the original filenames. For example, Twitter25 must be located at
   `data/saved_graphs/twitter25_graph.pkl`.

If you save `graphs.zip` in the repository root, extract it with:

```bash
python -m zipfile -e graphs.zip data/saved_graphs/
```

See [data/saved_graphs/README.md](data/saved_graphs/README.md) for the full filename
list and archive checksum. The large caches remain excluded from Git and are
provided as a Release attachment.

The main entry point loads preprocessed caches. It does not automatically download
data or regenerate missing datasets. Preprocessing functions in `create_graphs.py`
require the corresponding raw data for datasets other than the built-in Karate
graph.

## Quick start

Display the available arguments:

```bash
python main.py --help
```

Run a small CPU smoke test, including one training epoch, checkpoint saving,
validation, and evaluation:

```bash
python main.py --dataset karate_SI --model TGLR --gpu -1 --epochs 1 --limit-graphs 20 --ttt-steps 2 --no-visual
```

This reduced run checks that the pipeline works; its metrics are **not** the
manuscript results. New checkpoints are saved under `runs/checkpoints/` and logs
under `output/`, so supplied checkpoints are not overwritten.

## Training

For example, train TGLR on Karate-SI:

```bash
python main.py --dataset karate_SI --model TGLR --obs-len 20 --diff-weight 0.1 --ttt-steps 100 --gpu 0
```

Examples for other dataset configurations already noted in `args.py`:

```bash
python main.py --dataset jazz_SIR --model TGLR --obs-len 5 --diff-weight 0.1 --ttt-steps 100 --no-sampling
python main.py --dataset facebook_SI --model TGLR --obs-len 5 --diff-weight 1.0 --ttt-steps 30 --sampling
python main.py --dataset twitter25 --model TGLR --obs-len 5 --diff-weight 0.1 --no-sampling
python main.py --dataset twitter15 --model TGLR --obs-len 2 --diff-weight 0.01 --no-sampling
python main.py --dataset twitter16 --model TGLR --obs-len 3 --diff-weight 0.01 --no-sampling
python main.py --dataset weibo --model TGLR --obs-len 2 --diff-weight 0.01 --no-sampling
```

The CLI does not silently switch hyperparameters when `--dataset` changes.
Set the observation length and other dataset-specific options explicitly.
The Cora-ML notes specify `--obs-len 5 --diff-weight 1.0 --ttt-steps 30`;
training-time sampling should be set to match the experiment being reproduced.
`--sampling` controls latent sampling for training predictions; it does not
remove sampling used by the diffusion training objective.

Defaults include 300 epochs, batch size 1, learning rate 0.001, KL weight 0.01,
training seed 127, and graph-shuffle seed 123. Graphs are shuffled once and split
80%/10%/10% into training/validation/test sets. Model selection uses validation
F1, with the existing early-stopping logic in `train.py`.

Available `--model` values:

```text
TGLR, GLAD, GLCFGD, TGLR_wo_G, TGLR_wo_LA, TGLR_wo_T,
TGLR_wo_D, TGLR_w_C, TGLR_w_CA, TGLR_w_Dy
```

The full TGLR model uses `args.hidden_dim` for its latent width (32 by default).
The separate legacy `args.latent_dim` field does not set TGLR's effective latent
width. Keep architecture settings compatible with any checkpoint being loaded.

## Evaluation

Evaluate a provided checkpoint using the same dataset and observation configuration:

```bash
python main.py --mode test --dataset karate_SI --model TGLR --obs-len 20 --ttt-steps 100 --checkpoint model_saves/checkpoints/TGLR_karate_SI_best_by_f1.pt --gpu -1
```

Evaluate a newly trained checkpoint:

```bash
python main.py --mode test --dataset karate_SI --model TGLR --obs-len 20 --checkpoint runs/checkpoints/TGLR_karate_SI_best_by_f1.pt
```

The evaluator reports accuracy, precision, recall, F1, and AUC averaged across
graphs. It excludes undefined graph-level AUC values from the AUC average.
Inference uses the model's `inference()` path; the temporal context encoder is
not used by full TGLR in that path. The cached-data evaluation pipeline still
constructs the input tensor from stored graph data.

Checkpoint filenames identify the model and dataset but do not encode every
hyperparameter or random seed. Use the matching configuration and full dataset
for scientific comparisons; do not use `--limit-graphs` for that purpose.

## Additional commands

Resume from a checkpoint containing optimizer and scheduler state:

```bash
python main.py --dataset karate_SI --model TGLR --resume runs/checkpoints/TGLR_karate_SI_checkpoint_epoch_100.pt --epochs 300
```

Inspect a dataset without training:

```bash
python main.py --mode stats --dataset karate_SI
```

Enable manifold plots during training with `--visual`. Plotting and density
estimation add runtime. `static.py` computes additional real-world structural
statistics after all four real-world graph caches have been installed.

`numpy_data_conv.py` is an optional export script for an external SIDSL working
copy. SIDSL source code is not bundled, and this utility is not imported by the
TGLR training pipeline.

## Manuscript

Manuscript title: *Latent Distribution Modeling for Source Localization under
Temporal Observation Mismatch*.

Publication metadata will be added when available. No publication DOI is assigned
in this repository at present.
