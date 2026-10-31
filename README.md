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
eight preprocessed graph caches are available in **[TGLR Datasets on OSF](https://osf.io/v4mp5/overview?view_only=5792705d28d54a19885b470123dd4598)**:

**[Download graphs.zip](https://osf.io/download/6ac0c1f683fe948c5b496b14/?view_only=5792705d28d54a19885b470123dd4598)** (approximately 310 MiB compressed; 2.4 GB extracted).

The archive contains Cora-ML and Facebook caches under SI and SIR, plus Twitter15,
Twitter16, Twitter25, and Weibo caches. To install them:

1. Download `graphs.zip` from the link above. The OSF view-only link supports
   anonymous access without signing in.
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
provided through OSF and as a GitHub Release attachment.

The public Twitter25 cache uses node indices local to each cascade and reindexed
cascade identifiers; original identifier mappings are not released. The update
preserves graph structure, all numerical features, labels, snapshots, and the
ordering used by the data loader. If you downloaded the archive before the
October 1, 2026 update, download it again to obtain this public copy.

The main entry point loads preprocessed caches. It does not automatically download
data or regenerate missing datasets. Preprocessing functions in `create_graphs.py`
require the corresponding raw data for datasets other than the built-in Karate
graph.

### Twitter25-NQ

[Download Twitter25-NQ](https://github.com/DancinPuppet/TGLR/releases/download/twitter25-nq-v1.0/twitter25_noquote_graph.zip)
(977 cascades). Quote posts and their incident edges are removed, retaining the
connected component containing the original source. Extract the cache and evaluate
the supplied checkpoint:

```bash
python -m zipfile -e twitter25_noquote_graph.zip data/saved_graphs/
python main.py --mode test --dataset twitter25_noquote --model TGLR --obs-len 5 --obs-start 0 --diff-weight 0.1 --no-sampling --ttt-steps 80 --beta-start 1e-4 --beta-end 0.01 --checkpoint model_saves/checkpoints/TGLR_twitter25_noquote_best_by_f1.pt --no-visual
```

The checkpoint was selected by validation F1. Recorded test F1/AUC: **0.3800 / 0.8184**.
Standalone reevaluation can vary because test-time adaptation is stochastic.
The fixed-seed split contains 781 training, 98 validation, and 98 test cascades.

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
python main.py --dataset karate_SI --model TGLR --obs-len 20 --diff-weight 0.1 --ttt-steps 50 --no-sampling --beta-start 1e-5 --beta-end 1e-4 --gpu 0
```

The supplied full-model checkpoints correspond to Tables 3 and 4. The five
structural features are degree centrality, average neighbor degree, clustering,
core number, and **maximum neighbor degree**. The last feature replaces PageRank
and is divided by the maximum degree of the graph (zero for isolated nodes).
It is computed from the cached topology during tensor conversion; the released
cache files do not need to be regenerated.

The CLI does not switch settings automatically when `--dataset` changes. Set
`--obs-len`, `--diff-weight`, `--ttt-steps`, `--sampling`/`--no-sampling`,
`--beta-start`, and `--beta-end` according to the checkpoint configuration below.
All use `--obs-start 0` and `--kl-weight 0.01`.

| Dataset | Obs. length | Diff. weight | Sampling | TTA steps | Beta start / end | Recorded F1 / AUC |
| --- | ---: | ---: | :---: | ---: | --- | --- |
| `karate_SI` | 20 | 0.1 | no | 50 | 1e-05 / 0.0001 | 0.5425 / 0.9267 |
| `karate_SIR` | 20 | 0.1 | no | 50 | 0.0001 / 0.01 | 0.4047 / 0.7642 |
| `jazz_SI` | 5 | 0.1 | no | 100 | 0.0001 / 0.01 | 0.4976 / 0.9101 |
| `jazz_SIR` | 5 | 0.1 | no | 50 | 0.0001 / 0.01 | 0.4398 / 0.8619 |
| `cora_ml_SI` | 5 | 1 | yes | 30 | 0.0001 / 0.01 | 0.5683 / 0.9386 |
| `cora_ml_SIR` | 5 | 1 | yes | 50 | 1e-05 / 0.0001 | 0.4866 / 0.8707 |
| `facebook_SI` | 5 | 1 | yes | 30 | 0.0001 / 0.01 | 0.5971 / 0.9476 |
| `facebook_SIR` | 5 | 1 | yes | 30 | 0.0001 / 0.01 | 0.4713 / 0.8362 |
| `twitter25` | 5 | 0.1 | no | 100 | 0.0001 / 0.01 | 0.3420 / 0.8071 |
| `twitter25_noquote` | 5 | 0.1 | no | 80 | 0.0001 / 0.01 | 0.3800 / 0.8184 |
| `twitter15` | 2 | 0.01 | no | 100 | 0.0001 / 0.01 | 0.3140 / 0.7807 |
| `twitter16` | 3 | 0.01 | no | 100 | 0.0001 / 0.01 | 0.3186 / 0.7894 |
| `weibo` | 2 | 0.01 | no | 80 | 0.0001 / 0.01 | 0.3113 / 0.7893 |

For example, train Facebook-SI with its matching settings:

```bash
python main.py --dataset facebook_SI --model TGLR --obs-len 5 --diff-weight 1.0 --ttt-steps 30 --sampling --beta-start 1e-4 --beta-end 0.01 --no-visual
```

`--sampling` controls latent sampling for training predictions; it does not
remove sampling used by the diffusion training objective.

Defaults include 300 epochs, batch size 1, learning rate 0.001, KL weight 0.01,
training seed 127, and graph-shuffle seed 123. Graphs are shuffled once and split
80%/10%/10% into training/validation/test sets. Model selection uses validation
F1, with the existing early-stopping logic in `train.py`. Training and validation
loaders use the original shuffled ordering; standalone test evaluation uses a
fixed order. Recorded post-training metrics can differ from a standalone run
because test-time adaptation also depends on the random-number state.

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
python main.py --mode test --dataset karate_SI --model TGLR --obs-len 20 --diff-weight 0.1 --ttt-steps 50 --no-sampling --beta-start 1e-5 --beta-end 1e-4 --checkpoint model_saves/checkpoints/TGLR_karate_SI_best_by_f1.pt --gpu -1
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

### Scope of the supplied checkpoints

The existing `TGLR_*` files match the main comparison tables. The four-scene
`TGLR_wo_LA`, `TGLR_wo_G`, `TGLR_wo_T`, and `TGLR_wo_D` files match the selected
Table 5 ablations (without distribution modeling, temporal refinement, TTA,
and diffusion plus its dependent TTA, respectively). `TGLR_wo_D` uses the same
training prediction-sampling setting as the corresponding full model.

Use the dataset settings above for these variants, with two exceptions:
`TGLR_wo_LA` and `TGLR_wo_G` on Karate-SIR use `--ttt-steps 100`;
`GLAD` uses `--ttt-steps 0` and predicts directly from the student distribution.
The supplied `TGLR_w_C` checkpoints match the Karate/Cora-ML concatenation results
in Figure 4, and `GLAD`/`GLCFGD` match their Figure 5 results.

Some figures and the Karate-SIR full-model row of Table 5 use separate trained
runs from Tables 3/4. Their full-model checkpoints are not substituted for the
main-table files. No additional checkpoints are included in this update.

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
