# CALM-Stroke

Code and results for **"CALM-Stroke: Clinically Anchored, Lesion-Aware Deep Learning on Diffusion MRI for Predicting Functional Outcome after Acute Ischemic Stroke"** (L. R. Mercado-Diaz, L. S. Donado Pinto, C. Barrera-Causil, H. F. Posada-Quintero; submitted to the IEEE Journal of Biomedical and Health Informatics).

CALM-Stroke reads the admission DWI trace image and the relative ADC map with a pretrained 2.5D slice encoder, predicts the acute lesion, summarizes it inside 32 arterial territories as tokens that a transformer combines with clinical tokens, and adds the result as a correction to a clinical logistic model fitted inside each training fold. Everything here runs on the public [Stroke Outcome Optimization Project (SOOP)](https://openneuro.org/datasets/ds004889/versions/1.1.2) release, so every number in the paper can be regenerated from scratch.

## What is in the repository

```
reproduce.py               runs the whole study, stage by stage
scripts/download_soop.py   downloads SOOP from OpenNeuro and checks every file against data/soop_manifest.csv
scripts/fetch_atlas.py     downloads the arterial territory atlas (Liu et al., 2023)
src/preprocess_dwi.py      brain mask, relative ADC, affine registration to MNI152 2 mm, lesion features, QC images
src/cohort_splits.py       cohort table and the frozen folds of the three validation schemes
src/atlas_features.py      lesion load per arterial territory and 4 mm lesion maps
src/baselines.py           reference models R0, R1, R1b, R2a, R2b, I1 and I2
src/build_cache.py         memory-mapped image arrays for training
src/calm_stroke_model.py   tokenizers, losses and the 3D variant (CALM-3D)
src/train.py               all networks: CALM-Stroke, 2.5D MIL, CALM-3D, 3D ResNet-18 and the ablations
src/evaluate.py            AUC with bootstrap CIs, paired differences, calibration, ordinal metrics, subgroups, decision curves
src/paper_outputs.py       figures, table bodies and number macros of the manuscript
src/explain.py             lesion-map agreement, territory-token occlusion and the late-fusion comparator
results/                   out-of-fold predictions of every model in the paper, cohort table, logs and manuscript outputs
```

## Installation

Python 3.11 and a CUDA GPU with at least 16 GB of memory are needed for training; evaluation and figures run on a CPU.
The paper used PyTorch 2.11 (CUDA 12.8), MONAI 1.6.1, scikit-learn 1.9.1 and nibabel 5.4.2 on one NVIDIA RTX 4000 Ada (20 GB).

```bash
git clone https://github.com/jouninLRMD/CALM-Stroke.git
cd CALM-Stroke
conda env create -f environment.yml
conda activate calm-stroke
# install the PyTorch build that matches your CUDA version, for example:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Reproducing the paper

From the repository root:

```bash
python reproduce.py            # everything, from the download to the manuscript outputs
python reproduce.py --dry-run  # list the commands without running them
```

`reproduce.py` runs the following stages in order. Each one can be run alone with `--only <stage>`, or the pipeline can be resumed with `--from <stage>`.

| Stage | What it does | Time (approximate) |
|---|---|---|
| `download` | SOOP metadata, lesion masks and DWI (2.8 GB) | depends on the connection |
| `atlas` | arterial territory atlas | seconds |
| `preprocess` | preprocessing and registration of 1,715 sessions | 1 to 2 h with 6 CPU workers |
| `cohort` | cohort table, 718 labeled patients, frozen folds | 1 min |
| `features` | territory lesion loads, 4 mm lesion maps | 5 min |
| `baselines` | reference models in all validation schemes | 10 min |
| `cache` | image arrays for training (about 7 GB) | 15 min |
| `train` | 54 training runs (models, seeds, ablations) | about 20 min per five-fold run of a 2.5D model; the 3D models take longer |
| `evaluate` | all metrics and `results/report.md` | 10 to 20 min |
| `paper` | `results/paper`: Figs. 5 to 8, Tables I to III and the number macros | 10 min |
| `explain` | `results/explain`: Fig. 9 data, late fusion and the image-only network (Table II) | 15 min on the GPU |

Finished training runs are skipped, so the command can be interrupted and started again. `--models calm25d` limits training to one model family and `--force` retrains existing runs.

### Without a GPU

The out-of-fold predictions of every model in the paper are included in `results/oof`. To recompute all tables, figures and numbers from them:

```bash
python reproduce.py --only evaluate paper
```

## Models and runs

| Name in the paper | `src/train.py` arguments | Seeds (cv5 / loso / field) |
|---|---|---|
| CALM-Stroke | `--model calm25d` | 5 / 3 / 3 |
| CALM-Stroke, no lesion head | `--model calm25d --no-seg` | 3 / - / - |
| CALM-Stroke, no auxiliary tasks | `--model calm25d --no-aux` | 3 / - / - |
| CALM-Stroke, no anchor | `--model calm25d --no-anchor` | 3 / - / - |
| 2.5D MIL + anchor | `--model slice25d` | 5 / 3 / 3 |
| 2.5D MIL + lesion anchor | `--model slice25d --anchor-vars clin_lesion` | 5 / 3 / 3 |
| 2.5D network, image only (late fusion input) | `--model slice25d --no-anchor` | 3 / - / - |
| CALM-3D (from scratch) | `--model calm` | 3 / 1 / 1 |
| 3D ResNet-18 + anchor | `--model resnet3d` | 1 / 1 / 1 |
| 3D ResNet-18, no anchor | `--model resnet3d --no-anchor` | 1 / - / - |

Validation schemes: `cv5` is five-fold cross-validation with folds stratified by outcome and field strength and frozen before training; `loso` holds out each Philips scanner model with at least 30 labeled patients in turn; `field` trains on all 1.5 T patients and tests on the 178 patients scanned at 3 T. Predictions of the seeds of one configuration are averaged.

## Data

SOOP (OpenNeuro ds004889, version 1.1.2) is released under a CC0 license by Absher et al., *Scientific Data* 11, 839 (2024), doi:[10.1038/s41597-024-03667-5](https://doi.org/10.1038/s41597-024-03667-5). The arterial territory atlas is from Liu et al., *Scientific Data* 10, 74 (2023), doi:[10.1038/s41597-022-01923-0](https://doi.org/10.1038/s41597-022-01923-0), as distributed in the [SOOP demonstration repository](https://github.com/neurolabusc/StrokeOutcomeOptimizationProjectDemo) under a BSD 2-Clause license; it is downloaded, not redistributed here.

By default the data go to `data/` inside the repository. Set the environment variables `CALM_DATA` (download, preprocessed volumes, atlas), `CALM_WORK` (image cache and model weights) and `CALM_RESULTS` (tables and predictions) to use other folders.

## Citation

If you use this code, please cite the paper (the reference will be updated after publication):

```
L. R. Mercado-Diaz, L. S. Donado Pinto, C. Barrera-Causil and H. F. Posada-Quintero,
"CALM-Stroke: Clinically anchored, lesion-aware deep learning on diffusion MRI for predicting functional
outcome after acute ischemic stroke," submitted to IEEE J. Biomed. Health Inform., 2026.
```

## License

The code is released under the GNU General Public License v3.0 (see `LICENSE`).

## Contact

Luis R. Mercado-Diaz, Department of Biomedical Engineering, University of Connecticut.
Questions and issues are welcome through the GitHub issue tracker.
