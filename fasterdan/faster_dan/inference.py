"""
inference.py
============
Trained checkpoint  ->  raw page images  ->  transcriptions (with layout tokens).

No dataset, no labels, no separate export step. Point it at the model's output
folder and a folder (or list) of raw page images.

    from notarypedia_transcription_framework.inference import TranscriptionEngine

    engine = TranscriptionEngine.from_pretrained("outputs/fdan_R352_R335_page_sem")
    results = engine.transcribe("path/to/raw_pages")     # {page_id: text}

What from_pretrained reads from the model folder:
  - the charset, from the checkpoint itself (checkpoints/best_*.pt);
  - the normalisation mean/std, from results/params.txt (the training hyperparameter
    dump). These are NOT metrics: the model was trained on normalised images, so the
    same normalisation must be applied at inference or the output is garbage.

If params.txt is missing, mean/std are computed from the input pages instead (a note
is printed), or you can pass mean=... and std=... explicitly.

NOTE on import paths: imports below use the current ``faster_dan.*`` layout. Adjust
the prefix if/when the modules are renamed under the new package.
"""

import io
import os
import re
import json

import numpy as np
import torch
from PIL import Image
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from faster_dan.basic.generic_dataset_manager import GenericDataset
from faster_dan.OCR.ocr_dataset_manager import (
    OCRDataset, OCRDatasetManager, OCRCollateFunction,
)
from faster_dan.OCR.ocr_utils import LM_ind_to_str
from faster_dan.OCR.document_OCR.faster_dan.trainer_faster_dan import Manager
from faster_dan.OCR.document_OCR.faster_dan.models_faster_dan import GlobalHTADecoder
from faster_dan.basic.models import FCN_Encoder
from faster_dan.Datasets.dataset_formatters.rimes_formatter import SEM_TOKENS as RIMES_SEM_TOKENS
from faster_dan.Datasets.dataset_formatters.read2016_formatter import SEM_TOKENS as READ_SEM_TOKENS
try:
    from faster_dan.Datasets.dataset_formatters.r352_formatter import SEM_TOKENS as R352_SEM_TOKENS
except ImportError:
    R352_SEM_TOKENS = {}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

# Architecture of the combined R352+R335 model (from the training build_params).
# Values found in params.txt override these; kept here only as a fallback.
_DEFAULT_MODEL_PARAMS = {
    "transfer_learning": None,
    "transfered_charset": True,
    "additional_tokens": 1,
    "input_channels": 3,
    "dropout": 0.5,
    "enc_dim": 256,
    "nb_layers": 5,
    "pe_h_max": 500,
    "pe_w_max": 1000,
    "l_max": 15000,
    "dec_num_layers": 8,
    "dec_num_heads": 4,
    "dec_res_dropout": 0.1,
    "dec_pred_dropout": 0.1,
    "dec_att_dropout": 0.1,
    "dec_dim_feedforward": 256,
    "attention_win": 100,
    "use_tokens_from_all_lines": True,
    "use_first_pass_tokens": True,
    "use_line_indices": True,
    "two_step_pos_enc_mode": "cat",
}
_DROP_MP_KEYS = {"models", "dropout_scheduler", "total_params", "device", "use_amp", "vocab_size"}


def _natural_key(name):
    """Sort key so 'p2' precedes 'p10' regardless of the folder's naming style."""
    return [int(tok) if tok.isdigit() else tok.lower()
            for tok in re.split(r"(\d+)", name)]


# --------------------------------------------------------------------------- #
#  Image ingestion                                                            #
# --------------------------------------------------------------------------- #
def _pil_to_array(pil_img):
    img = np.array(pil_img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return img


def _load_image_any(doc):
    """Load a single page image from a path, bytes, PIL image, or numpy array."""
    if isinstance(doc, np.ndarray):
        return doc if doc.ndim == 3 else np.expand_dims(doc, axis=2)
    if isinstance(doc, Image.Image):
        return _pil_to_array(doc)
    if isinstance(doc, (bytes, bytearray)):
        with Image.open(io.BytesIO(doc)) as pil_img:
            return _pil_to_array(pil_img)
    with Image.open(doc) as pil_img:        # path-like
        return _pil_to_array(pil_img)


def _to_samples(documents):
    """Normalise ``documents`` into [{"name", "path"?, "img"}].

    Accepts a directory path, a single page image, or a list of page images;
    each page image may be a file path, raw bytes, a PIL image, or a numpy array.
    """
    if isinstance(documents, (str, os.PathLike)) and os.path.isdir(documents):
        folder = os.fspath(documents)
        names = [f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS)]
        if not names:
            # No standard extensions (e.g. files named "R352_001_no_symbols"):
            # fall back to every file and let the image loader decide.
            names = [f for f in os.listdir(folder)
                     if os.path.isfile(os.path.join(folder, f))]
        names.sort(key=_natural_key)
        samples = []
        for f in names:
            full = os.path.join(folder, f)
            try:
                img = _load_image_any(full)
            except Exception:
                continue   # skip anything that isn't a readable image
            samples.append({"name": f, "path": full, "img": img})
        if not samples:
            raise FileNotFoundError("No readable images found in: {}".format(folder))
        return samples

    items = documents if isinstance(documents, (list, tuple)) else [documents]
    samples = []
    for i, doc in enumerate(items):
        if isinstance(doc, (str, os.PathLike)):
            path = os.fspath(doc)
            samples.append({"name": os.path.basename(path), "path": path,
                            "img": _load_image_any(path)})
        else:
            samples.append({"name": "page_{}".format(i), "img": _load_image_any(doc)})
    if not samples:
        raise ValueError("No documents to transcribe.")
    return samples


def _ensure_rgb(img):
    if img.ndim == 2:
        img = img[:, :, None]
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return img


def _compute_mean_std(samples):
    """Fallback only: per-channel mean/std over the input pages (after to_RGB)."""
    s = np.zeros(3, dtype=np.float64)
    s2 = np.zeros(3, dtype=np.float64)
    n = 0
    for sm in samples:
        img = _ensure_rgb(sm["img"]).astype(np.float64)
        s += img.sum(axis=(0, 1))
        s2 += (img ** 2).sum(axis=(0, 1))
        n += img.shape[0] * img.shape[1]
    mean = s / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-8))
    return mean.tolist(), std.tolist()


# --------------------------------------------------------------------------- #
#  Charset / stats recovery (no dataset needed)                               #
# --------------------------------------------------------------------------- #
def _derive_char_only_set(charset):
    """Reproduce OCRDatasetManager's split of the charset into character tokens
    vs. semantic/layout tokens, without touching the Fonts directory."""
    char_only_set = list(charset)
    sem_tokens = []
    for token_dict in (RIMES_SEM_TOKENS, READ_SEM_TOKENS, R352_SEM_TOKENS):
        for char in token_dict.values():
            if char in char_only_set:
                char_only_set.remove(char)
                sem_tokens.append(char)
            if char.upper() in char_only_set:
                char_only_set.remove(char.upper())
                sem_tokens.append(char.upper())
    for token in ("\n", "\t"):
        if token in char_only_set:
            char_only_set.remove(token)
    return char_only_set, sem_tokens


def _find_best_checkpoint(model_dir, checkpoint=None):
    if checkpoint:
        return checkpoint
    pts = []
    for d in (os.path.join(model_dir, "checkpoints"), model_dir):
        if os.path.isdir(d):
            pts += [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".pt")]
    if not pts and os.path.isdir(model_dir):
        for root, _, files in os.walk(model_dir):     # recursive fallback
            pts += [os.path.join(root, f) for f in files if f.endswith(".pt")]
    if not pts:
        raise FileNotFoundError(
            "No .pt checkpoint found under: {}\n"
            "Point model_dir at the folder containing checkpoints/ (and "
            "results/params.txt), or pass checkpoint=PATH explicitly.".format(model_dir))
    best = [p for p in pts if "best" in os.path.basename(p).lower()]   # prefer the best checkpoint
    return (best or pts)[0]


def _read_train_stats(model_dir):
    """Recover (mean, std, model_params) from the training params.txt dump.
    Returns (None, None, None) if no usable file is found."""
    candidates = []
    for d in (os.path.join(model_dir, "results"), model_dir):
        if os.path.isdir(d):
            candidates += [os.path.join(d, f) for f in os.listdir(d)
                           if f.startswith("params") and f.endswith(".txt")]
    for path in sorted(candidates):
        try:
            with open(path) as f:
                p = json.load(f)
            cfg = p["dataset_params"]["config"]
            mean, std = cfg.get("mean"), cfg.get("std")
            mp = {k: v for k, v in p.get("model_params", {}).items() if k not in _DROP_MP_KEYS}
            if mean is not None and std is not None:
                return mean, std, mp
        except Exception:
            continue
    return None, None, None


# --------------------------------------------------------------------------- #
#  Dataset: in-memory page images, no labels                                  #
# --------------------------------------------------------------------------- #
class InferenceImageDataset(OCRDataset):
    """Reuses OCRDataset.__getitem__ (image-only) but takes pre-loaded page images
    with empty labels. ``paths_and_sets`` here is the sample list from _to_samples."""

    def load_samples(self, paths_and_sets, load_in_memory=True):
        samples = []
        for entry in paths_and_sets:
            samples.append({
                "name": entry["name"],
                "label": "",
                "unchanged_label": "",
                "path": entry.get("path", entry["name"]),
                "nb_cols": 1,
                "img": entry["img"],
            })
        return samples


# --------------------------------------------------------------------------- #
#  Dataset manager: charset from checkpoint, no formatted data                 #
# --------------------------------------------------------------------------- #
class InferenceOCRDatasetManager(OCRDatasetManager):

    def __init__(self, params):
        params["config"]["synthetic_data"] = None     # skip the Fonts directory walk
        super().__init__(params)                        # derives tokens from the charset
        # The synthetic branch normally derives these; derive them directly instead.
        self.char_only_set, self.sem_tokens = _derive_char_only_set(self.charset)

    def generate_inference_loader(self, samples, custom_name="raw"):
        if custom_name in self.test_loaders:
            self.remove_test_dataset(custom_name)
        self.test_datasets[custom_name] = self.dataset_class(
            self.params, "test", custom_name, samples,
        )
        self.apply_specific_treatment_after_dataset_loading(self.test_datasets[custom_name])
        self.test_samplers[custom_name] = None
        self.test_loaders[custom_name] = DataLoader(
            self.test_datasets[custom_name],
            batch_size=1, shuffle=False, drop_last=False,
            num_workers=0, collate_fn=self.my_collate_function,
        )


# --------------------------------------------------------------------------- #
#  Manager: construct without training data; predict without ground truth      #
# --------------------------------------------------------------------------- #
class InferenceManager(Manager):

    def __init__(self, params, device=None):
        # Deliberately bypass Manager.__init__ (which loads training datasets).
        self.params = params
        self.models = {}
        self.dropout_scheduler = None

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Mixed precision is CUDA-only; off on CPU/MPS so the same script runs on a Mac.
        use_amp = (self.device.type == "cuda")
        self.params["training_params"]["use_amp"] = use_amp
        self.params["model_params"]["device"] = self.device.type
        self.params["model_params"]["use_amp"] = use_amp

        self.dataset = InferenceOCRDatasetManager(self.params["dataset_params"])
        self.dataset.my_collate_function = OCRCollateFunction(self.params["dataset_params"]["config"])

    def load_weights(self, checkpoint):
        """Build encoder/decoder and load weights only — no optimizers/schedulers."""
        mp = self.params["model_params"]
        mp["vocab_size"] = len(self.dataset.charset)     # decoder output dim
        mp["models"] = {"encoder": FCN_Encoder, "decoder": GlobalHTADecoder}
        for name in ("encoder", "decoder"):
            net = mp["models"][name](mp).to(self.device)
            net.load_state_dict(checkpoint["{}_state_dict".format(name)], strict=True)
            net.eval()
            self.models[name] = net

    def predict_batch(self, batch_data):
        """Forward pass returning transcriptions only (evaluate_batch minus all GT)."""
        x = batch_data["imgs"].to(self.device)
        reduced_size = [s[:2] for s in batch_data["imgs_reduced_shape"]]
        with autocast(enabled=self.params["training_params"]["use_amp"]):
            features = self.evaluate_encoder(x, batch_data)
            features_size = features.size()
            pos_features = self.models["decoder"].features_updater.get_pos_features(features)
            pos_features = torch.flatten(pos_features, start_dim=2, end_dim=3).permute(2, 0, 1)
            predicted_tokens, prediction_len, confidence_scores, cache = \
                self.eval_first_pass(pos_features, reduced_size, features_size)
            _, predicted_tokens = self.eval_second_pass(
                pos_features, reduced_size, features_size,
                predicted_tokens, prediction_len, confidence_scores, cache,
            )
            str_x = [LM_ind_to_str(self.dataset.charset, t, oov_symbol="")
                     for t in predicted_tokens]
        return {"names": batch_data["names"], "str_x": str_x}


# --------------------------------------------------------------------------- #
#  Params                                                                      #
# --------------------------------------------------------------------------- #
def build_inference_params(charset, mean, std, model_params):
    config = {
        "load_in_memory":  True,
        "worker_per_gpu":  1,
        "width_divisor":   8,
        "height_divisor":  32,
        "padding_value":   0,
        "padding_token":   None,
        "charset_mode":    "seq2seq",
        "constraints":     ["add_eot", "add_sot"],   # fdan_encoding dropped (label-only)
        "normalize":       True,
        "preprocessings":  [{"type": "to_RGB"}],
        "augmentation":    None,
        "synthetic_data":  None,
        "charset":         [],
    }
    if mean is not None:
        config["mean"] = mean
    if std is not None:
        config["std"] = std

    dataset_params = {
        "dataset_manager":  InferenceOCRDatasetManager,
        "dataset_class":    InferenceImageDataset,
        "datasets":         {},
        "charset":          charset,        # OCRDatasetManager reads this; no labels.pkl
        "config":           config,
        "batch_size":       1,
        "valid_batch_size": 1,
        "test_batch_size":  1,
        "num_gpu":          1,
    }
    training_params = {
        "use_amp":           True,
        "max_line_pred":     100,
        "max_pred_per_line": 150,
        "force_cpu":         False,
        "nb_gpu":            torch.cuda.device_count(),
    }
    return {
        "dataset_params":  dataset_params,
        "model_params":    dict(model_params),
        "training_params": training_params,
    }


# --------------------------------------------------------------------------- #
#  Public engine                                                              #
# --------------------------------------------------------------------------- #
class TranscriptionEngine:

    def __init__(self, manager):
        self.manager = manager

    @classmethod
    def from_pretrained(cls, model_dir, device=None, mean=None, std=None, checkpoint=None):
        """Load a trained model from its output folder (e.g. outputs/fdan_R352_R335_page_sem).
        Reads the charset from the checkpoint and the normalisation mean/std from
        results/params.txt. mean/std may be overridden explicitly."""
        ckpt_path = _find_best_checkpoint(model_dir, checkpoint)
        checkpoint_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        charset = checkpoint_data["charset"]

        file_mean, file_std, file_mp = _read_train_stats(model_dir)
        if mean is None:
            mean = file_mean
        if std is None:
            std = file_std

        model_params = dict(_DEFAULT_MODEL_PARAMS)
        if file_mp:
            model_params.update(file_mp)

        params = build_inference_params(charset, mean, std, model_params)
        manager = InferenceManager(params, device=device)
        manager.load_weights(checkpoint_data)
        return cls(manager)

    @torch.no_grad()
    def transcribe(self, documents, output_dir=None):
        """Transcribe one or more raw page images. Returns {page_id: transcription}
        (transcription includes the layout tokens). Writes <page_id>.txt into
        output_dir if given; otherwise writes nothing."""
        samples = _to_samples(documents)

        # If training mean/std were unavailable, derive them from these pages.
        cfg = self.manager.dataset.params["config"]
        if cfg.get("mean") is None or cfg.get("std") is None:
            mean, std = _compute_mean_std(samples)
            cfg["mean"], cfg["std"] = mean, std
            print("[inference] No training mean/std found; using stats computed from "
                  "the input pages (mean={}, std={}).".format(
                      [round(x, 2) for x in mean], [round(x, 2) for x in std]))

        self.manager.dataset.generate_inference_loader(samples, "raw")
        loader = self.manager.dataset.test_loaders["raw"]

        results = {}
        for batch in loader:
            out = self.manager.predict_batch(batch)
            for name, text in zip(out["names"], out["str_x"]):
                page_id = os.path.splitext(os.path.basename(name))[0]
                results[page_id] = text

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            for page_id, text in results.items():
                with open(os.path.join(output_dir, page_id + ".txt"), "w", encoding="utf-8") as f:
                    f.write(text)

        return results

    transcribe_folder = transcribe
