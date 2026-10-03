# FasterDAN for 17th-Century Maltese Latin Notarial Manuscripts

Adaptation of [Faster DAN](https://github.com/FactoDeepLearning/FasterDAN) (Coquenet et al., ICDAR 2023) for page-level handwritten text recognition of the Notarial Archives Foundation registers **R352** (Notary Giovanni Battista Micallef) and **R335** (Notary Giovanni Luca Mamo). The code was developed for the MSc dissertation *Automated Transcription and Deed-Level Segmentation of Seventeenth-Century Maltese Latin Notarial Manuscripts* (University of Malta) as part of the Notarypedia initiative.

> **Not included in this repository:** manuscript images, ground-truth transcriptions, formatted datasets and trained weights. These are held by the Notarial Archives Foundation / the project team. Ask the maintainer for access.

The upstream README is kept as [`UPSTREAM_README.md`](UPSTREAM_README.md). The code is distributed under the CeCILL-C licence (see `LICENSE_CECILL-C.md`).

---

## Contents

1. [Repository layout](#1-repository-layout)
2. [Environment setup](#2-environment-setup)
3. [Data preparation](#3-data-preparation)
4. [Stage 1: synthetic line pre-training (CTC)](#4-stage-1-synthetic-line-pre-training-ctc)
5. [Stage 2: page-level FasterDAN training](#5-stage-2-page-level-fasterdan-training)
6. [Monitoring with TensorBoard](#6-monitoring-with-tensorboard)
7. [Evaluation and prediction](#7-evaluation-and-prediction)
8. [Known issues and gotchas](#8-known-issues-and-gotchas)
9. [Citation](#9-citation)

---

## 1. Repository layout

```
fasterdan/
├── setup.py
├── README.md                        ← this file
├── UPSTREAM_README.md               ← original Faster DAN README
├── LICENSE_CECILL-C.md
├── CITATION.cff
└── faster_dan/
    ├── basic/                       ← generic managers, models, transforms, schedulers, utils
    ├── Fonts/                       ← .ttf fonts used for synthetic document generation
    ├── Datasets/
    │   ├── raw/                     ← raw images + transcriptions (NOT in git)
    │   ├── formatted/               ← formatted datasets (NOT in git)
    │   └── dataset_formatters/      ← scripts that build Datasets/formatted/*
    ├── outputs/                     ← line pre-training checkpoints (NOT in git)
    ├── OCR/
    │   ├── ocr_dataset_manager.py   ← dataset, synthetic data, collate function
    │   ├── ocr_utils.py
    │   ├── line_OCR/ctc/            ← Stage 1: synthetic line generation + CTC pre-training
    │   └── document_OCR/faster_dan/
    │       ├── main_faster_dan.py            ← Stage 2: single register (R352)
    │       ├── main_faster_dan_combined.py   ← Stage 2: combined R352 + R335
    │       ├── trainer_faster_dan.py         ← FasterDAN training/inference logic
    │       ├── models_faster_dan.py          ← FasterDAN decoder
    │       ├── trainer_dan.py                ← base DAN trainer (required by trainer_faster_dan.py)
    │       ├── models_dan.py                 ← base DAN modules (required by models_faster_dan.py)
    │       └── outputs/                      ← page-model checkpoints + logs (NOT in git)
    ├── predict_combined_model.py    ← inference of the combined model on R352 / R335 / combined
    ├── predict_r335_with_r352.py    ← cross-scribal test: R352 model on R335
    ├── read_predictions.py          ← inspect saved predictions
    ├── inference.py                 ← transcribe raw page images with a trained checkpoint
    ├── run_transcription.py         ← command-line entry point for inference
    └── export_inference_assets.py   ← export checkpoint + params needed for inference
```

## 2. Environment setup

Tested on a Linux server with an NVIDIA GPU (CUDA). CPU works for inference only.

```bash
git clone https://github.com/charlieabela/notarypedia-placements.git
cd notarypedia-placements/fasterdan

# Option A: conda (as used on the training server)
conda create --name fdan_env python=3.10
conda activate fdan_env

# Option B: venv
python3 -m venv .venv
source .venv/bin/activate

pip install -e .
```

`setup.py` pins `torch==1.12.1` / `torchvision==0.13.1` (upstream). On newer GPUs or Python versions, install a matching PyTorch build first (https://pytorch.org/get-started/locally/), then run `pip install -e . --no-deps` and install the remaining dependencies manually:

```bash
pip install tensorboard scikit-learn opencv-python tqdm pillow networkx editdistance pyunpack fonttools
```

Before long runs, raise the open-file limit (the data loader opens many files):

```bash
ulimit -n 65536
```

## 3. Data preparation

### 3.1 Raw data

Place the raw register data under:

```
faster_dan/Datasets/raw/R352/
faster_dan/Datasets/raw/R335/
```

### 3.2 Format the datasets

Run the formatter scripts in `faster_dan/Datasets/dataset_formatters/` (list them with `ls faster_dan/Datasets/dataset_formatters/`). Each produces a folder in the standard DAN layout:

```
faster_dan/Datasets/formatted/<NAME>_page_sem/
├── train/        ← page images
├── valid/
├── test/
└── labels.pkl    ← ground truth per image + charset
```

Expected dataset folders:

| Folder | Used by |
|---|---|
| `R352_page_sem` | `main_faster_dan.py`, prediction scripts |
| `R335_page_sem` | prediction scripts |
| `R352_R335_page_sem` | `main_faster_dan_combined.py`, `predict_combined_model.py` |

The `_sem` suffix means the transcriptions include layout (semantic) tokens: `ⓑ … Ⓑ` for body text and `ⓜ … Ⓜ` for marginalia.

> **Filename collision:** R352 and R335 contain identically named files. When building `R352_R335_page_sem`, prefix R335 files (or key on `(register, filename)`); otherwise pages are silently overwritten.

### 3.3 Fonts

Synthetic page generation needs `.ttf` fonts in `faster_dan/Fonts/`. At least one font must support every character in the charset.

## 4. Stage 1: synthetic line pre-training (CTC)

The page model's encoder is initialised from a line-level CTC model trained on synthetic printed lines built from the register text.

```bash
cd faster_dan/OCR/line_OCR/ctc/
python3 main_syn_line.py        # generate the synthetic line dataset
python3 main_line_ctc_syn.py    # CTC pre-training
```

In both scripts, set the dataset name to the register (e.g. `R352`); the upstream defaults refer to `READ_2016`.

**Output:** `FCN_R352_line_syn/checkpoints/best_<epoch>.pt`. The Stage 2 scripts expect it at:

```
faster_dan/outputs/FCN_R352_line_syn/checkpoints/best_1040.pt
```

If your best epoch differs, update the `transfer_learning` paths in the Stage 2 script.

## 5. Stage 2: page-level FasterDAN training

All Stage 2 scripts must be run from their own directory, because dataset and checkpoint paths are relative:

```bash
cd faster_dan/OCR/document_OCR/faster_dan/
```

### 5.1 Single register (R352)

```bash
python3 main_faster_dan.py
```

Output folder: `outputs/fdan_R352_page_sem/`

To train an **R335-only** model, copy the script and change `dataset_name = "R335"` and `output_folder = "fdan_R335_page_sem"` (and the `transfer_learning` checkpoint if you pre-trained a separate R335 line model).

### 5.2 Combined register (R352 + R335)

```bash
python3 main_faster_dan_combined.py
```

Output folder: `outputs/fdan_R352_R335_page_sem/`. At the end of training this script also writes per-page predictions and a CER/WER summary to `outputs/fdan_R352_R335_page_sem/predictions/`.

### 5.3 Running on a remote server

Training runs for days. Launch it inside `tmux` (or `nohup`) so it survives SSH disconnects:

```bash
tmux new -s fdan
conda activate fdan_env
ulimit -n 65536
cd faster_dan/OCR/document_OCR/faster_dan/
python3 main_faster_dan_combined.py 2>&1 | tee train_combined.log
# Detach: Ctrl-b then d      Reattach: tmux attach -t fdan
```

### 5.4 Key hyperparameters (in the `params` dict of each script)

| Parameter | Value | Meaning |
|---|---|---|
| `max_training_time` | `3600 * 24 * 1.9` (combined) | Wall-clock limit in **seconds**. Training stops when this or `max_nb_epochs` is reached. |
| `max_nb_epochs` | `50000` | Epoch limit. |
| `load_epoch` | `"last"` | Resumes from the last checkpoint if one exists in the output folder. |
| `max_nb_lines` | `62` | Maximum lines per synthetic page; also drives the curriculum. |
| `curr_step` | `10000` | Training steps per added synthetic line in the curriculum. |
| `init_proba` → `end_proba` | `0.9` → `0.2` | Probability of a synthetic (vs. real) page, decayed linearly over `num_steps_proba`. |
| `batch_size` | `1` | Pages per training step. |
| `lr` | `1e-4` | Adam learning rate. |
| `use_amp` | `True` | Mixed-precision training. |
| `eval_on_valid_interval` | `5` | Validate every 5 epochs (once validation is enabled). |
| `focus_metric` | `cer` | Metric used to select the `best` checkpoint. |

**Validation is gated by the curriculum.** `start_valid_from_steps` is computed as `curr_start + curr_step × (max_nb_lines − min_nb_lines)` = `0 + 10000 × 61` = **610,000 steps**. No validation (and no `best` checkpoint) is produced before that point, so a shorter run will have no best model.

### 5.5 Resuming

Re-run the same script. With `load_epoch = "last"`, training resumes from `outputs/<output_folder>/checkpoints/last_<epoch>.pt`. To restart from scratch, move or delete that output folder.

### 5.6 Outputs

```
outputs/<output_folder>/
├── checkpoints/
│   ├── last_<epoch>.pt     ← most recent weights
│   └── best_<epoch>.pt     ← lowest validation CER
└── results/
    ├── events.out.tfevents.*   ← TensorBoard logs
    ├── params                  ← hyperparameters used (keep with the checkpoint)
    └── predict_*.txt           ← evaluation results after training
```

## 6. Monitoring with TensorBoard

On the server:

```bash
tensorboard --logdir faster_dan/OCR/document_OCR/faster_dan/outputs --port 6006
```

On your local machine, open an SSH tunnel and browse to http://localhost:6009:

```bash
ssh -N -L 6009:localhost:6006 <user>@<server>
```

Watch `syn_max_lines` climb to 62; validation CER appears only after that.

## 7. Evaluation and prediction

Evaluation on test / valid / train runs automatically at the end of each Stage 2 script using the `best` checkpoint.

To re-run inference separately (from the `faster_dan/` directory):

```bash
cd faster_dan/
python3 predict_combined_model.py      # combined model on R352, R335 and R352_R335
python3 predict_r335_with_r352.py      # R352-only model evaluated on R335 (cross-scribal)
python3 read_predictions.py            # inspect saved predictions
```

Predictions are saved as `<page>_pred.txt` / `<page>_gt.txt` pairs plus a `summary.txt` with per-page and overall CER/WER. The combined training script strips semantic tokens before computing its CER/WER summary.

Metrics: `cer`, `wer`, `map_cer` (layout-aware mAP-CER), `loer` (layout ordering error rate).

## 8. Known issues and gotchas

- **`max_training_time` is in seconds.** A test value such as `60` stops training after one minute. Check it before every long run.
- **Python ≥ 3.12/3.13 multiprocessing:** if data-loader workers crash with shared-memory errors, add near the top of the training script:
  ```python
  import torch.multiprocessing
  torch.multiprocessing.set_sharing_strategy("file_system")
  ```
- **Too many open files:** run `ulimit -n 65536` in the same shell before training.
- **Folio-side filenames** (e.g. `0001_R352_2v.jpg`): integer parsing of page indices in `OCRCollateFunction.__call__` (`ocr_dataset_manager.py`) is wrapped in `try/except ValueError` so `r`/`v` suffixes do not crash batching.
- **Relative paths:** Stage 2 scripts must be launched from `faster_dan/OCR/document_OCR/faster_dan/`; prediction scripts from `faster_dan/`.
- **Charset asymmetry:** R352 and R335 have different character inventories. A model trained on one register cannot emit characters it never saw.

## 9. Citation

Original method:

```bibtex
@inproceedings{Coquenet2023fasterdan,
  author    = {Coquenet, Denis and Chatelain, Clément and Paquet, Thierry},
  title     = {Faster DAN: Multi-target Queries with Document Positional Encoding for End-to-end Handwritten Document Recognition},
  booktitle = {International Conference on Document Analysis and Recognition (ICDAR)},
  year      = {2023},
  pages     = {182--199},
  series    = {Lecture Notes in Computer Science},
  volume    = {14190},
  doi       = {10.1007/978-3-031-41685-9_12}
}
```

Maltese notarial adaptation: Chukwuma, *Automated Transcription and Deed-Level Segmentation of Seventeenth-Century Maltese Latin Notarial Manuscripts*, MSc dissertation, Department of Artificial Intelligence, University of Malta.
