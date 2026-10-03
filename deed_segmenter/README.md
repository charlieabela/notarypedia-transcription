# Deed Segmenter for 17th-Century Maltese Latin Notarial Registers

Line-level deed-boundary classifier used as the second stage of the pipeline in the MSc dissertation *Automated Transcription and Deed-Level Segmentation of Seventeenth-Century Maltese Latin Notarial Manuscripts* (University of Malta, Notarypedia initiative). Stage 1 (page-level HTR) is in [`../fasterdan`](../fasterdan).

A fine-tuned `bert-base-multilingual-cased` model labels every transcribed line of a register, and the labels are then used to group lines, including across page breaks, into individual deeds.

> **Not included in git:** PAGE-XML ground truth, manuscript images and trained weights. The weights are distributed separately (see [§4](#4-trained-weights)). Ground truth is held by the Notarial Archives Foundation.

---

## 1. Method

| Item | Value |
|---|---|
| Base model | `bert-base-multilingual-cased` (Hugging Face) |
| Head | `BertForSequenceClassification`: `[CLS]` embedding → linear, 4 classes |
| Labels | `O`, `B-DEED`, `I-DEED`, `E-DEED` (`label_map.json`) |
| Input | ±3 lines of context around the target line, joined with ` [SEP] `; the target line is prefixed with `>>> ` |
| Max tokens | 256 |
| Loss | Cross-entropy with inverse-frequency class weights (up-weights the rare `B-DEED` / `E-DEED`) |
| Optimiser | AdamW, lr `2e-5`, weight decay `0.01`, linear warm-up over 10 % of steps, grad-clip 1.0 |
| Batch size | 16 (train), 32 (eval) |
| Split | By **page** (not line) to avoid leakage: 85 % train / 10 % val / 5 % test, seed 42 |
| Checkpoint selection | Best validation `deed_f1` (macro-F1 over `B-DEED`, `I-DEED`, `E-DEED`) |

### Label definitions

Labels come from the Transkribus structure tags in each `TextLine`'s `custom` attribute:

- `B-DEED`: line tagged `structure {type:SOD;}` (start of deed)
- `E-DEED`: line tagged `structure {type:EOD;}` (end of deed)
- `I-DEED`: any untagged line between an SOD and the next EOD
- `O`: any untagged line outside a deed (marginalia, foliation, etc.)

Pages are read in sorted filename order, so the SOD/EOD state carries across page breaks.

## 2. Setup

```bash
cd notarypedia-placements/deed_segmenter
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A GPU is used automatically if one is available. CPU works for inference.

## 3. Usage

### 3.1 Training

The input is a directory of PAGE-XML files (Transkribus export, 2013-07-15 schema) with SOD/EOD structure tags. `metadata.xml` is skipped.

```bash
python3 train_deed_segmenter.py \
    --page-dir   data/R352/page \
    --output-dir outputs \
    --epochs 10
```

Output:

```
outputs/
├── best_model/          ← model.safetensors, config.json, tokenizer.json, tokenizer_config.json
└── label_map.json
```

At the end of training, the script prints a per-class `classification_report` on the held-out test pages.

### 3.2 Inference

```bash
python3 train_deed_segmenter.py --infer \
    --model-dir outputs/best_model \
    --page-dir  <page-xml-dir> \
    --output    outputs/deeds.json
```

`--page-dir` can be any PAGE-XML directory with line text (e.g. a Transkribus or Kraken export). Structure tags are not needed at inference.

### 3.3 Deed reconstruction rules

Predicted labels are converted into deeds as follows:

- `B-DEED` opens a new deed (and closes any deed still open).
- `I-DEED` / `E-DEED` lines are appended to the open deed; `E-DEED` closes it.
- `O` lines are never dropped: inside an open deed they are appended to it; after a closed deed they are attached to that deed until the next `B-DEED`; before the first deed they go into a `preamble` entry (`deed_id = 0`).

Each record in `deeds.json`:

```json
{
  "deed_id": 1,
  "start_page": "0001_R352_1r",
  "end_page": "0002_R352_1v",
  "lines": ["...", "..."],
  "text": "...\n...",
  "line_count": 42
}
```

The script also prints the number of deeds found and how many span more than one page.

## 4. Trained weights

The released checkpoint (`deed_segmenter_best_model.zip`, ~700 MB) contains `best_model/` and `label_map.json`. It is attached to this repository's GitHub Releases rather than committed to git. To use it:

```bash
unzip deed_segmenter_best_model.zip -d outputs/
python3 train_deed_segmenter.py --infer \
    --model-dir outputs/best_model \
    --page-dir  <page-xml-dir> \
    --output    outputs/deeds.json
```

The checkpoint was saved with `transformers` 5.4.0.

## 5. Known issues

- **Page order matters.** Files are processed in sorted filename order, and deed state carries across pages. Use zero-padded, sequential filenames.
- **Filename collisions between registers.** R352 and R335 have identically named files. Prefix one register (e.g. `R335_`) before putting both in one directory.
- **Context windows span page breaks.** Context windows are built over the concatenated lines of each split, so a window can span two adjacent pages of the same split.
- **Rare classes.** `B-DEED` and `E-DEED` are a small fraction of lines, so per-class recall on a small test split varies a lot between runs.

## 6. Citation

Chukwuma, *Automated Transcription and Deed-Level Segmentation of Seventeenth-Century Maltese Latin Notarial Manuscripts*, MSc dissertation, Department of Artificial Intelligence, University of Malta.

```bibtex
@inproceedings{devlin2019bert,
  author    = {Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  title     = {{BERT}: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  booktitle = {Proceedings of NAACL-HLT},
  year      = {2019},
  pages     = {4171--4186}
}
```
