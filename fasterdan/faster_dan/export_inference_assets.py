"""
export_inference_assets.py
==========================
Run ONCE, on the machine where the trained combined model AND its formatted
datasets are still available (i.e. the server). It loads the combined model
exactly as ``predict_combined_model.py`` does, then writes a small, self-contained
asset bundle that the inference engine can consume WITHOUT any formatted datasets,
Fonts directory, or labels.pkl present.

Why this step exists
--------------------
The checkpoint already stores the ``charset`` and the model weights, but it does
NOT store the normalisation statistics (``mean``/``std``). Those are computed from
the training set at ``load_datasets`` time. If they are not carried over, inference
normalises images differently from training and accuracy degrades. This script
captures them once.

Run from the faster_dan/ root (same place you run predict_combined_model.py):

    conda activate fdan_env
    python export_inference_assets.py --out inference_assets

Produces inference_assets/:
    model.pt              copy of the best checkpoint (charset + weights)
    inference_config.pkl  charset, char_only_set, sem_tokens, tokens, mean, std, model_params
"""

import os
import shutil
import pickle
import argparse

import numpy as np

# build_params() is the exact training-time configuration for the combined model.
from predict_combined_model import build_params
from faster_dan.OCR.document_OCR.faster_dan.trainer_faster_dan import Manager


# Architecture hyperparameters are persisted, but live class/function references
# cannot (and should not) be pickled. They are re-attached at load time.
_NON_PICKLABLE_MODEL_KEYS = ("models", "dropout_scheduler")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="inference_assets",
                    help="Output directory for the asset bundle.")
    args = ap.parse_args()

    params = build_params()
    manager = Manager(params)      # builds dataset manager + computes mean/std
    manager.load_model()           # loads the best checkpoint

    ds = manager.dataset           # OCRDatasetManager
    cfg = params["dataset_params"]["config"]

    assets = {
        "charset":       ds.charset,
        "char_only_set": ds.char_only_set,
        "sem_tokens":    ds.sem_tokens,
        "tokens":        ds.tokens,
        "mean":          np.asarray(cfg["mean"]).tolist(),
        "std":           np.asarray(cfg["std"]).tolist(),
        "model_params":  {k: v for k, v in params["model_params"].items()
                          if k not in _NON_PICKLABLE_MODEL_KEYS},
    }

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "inference_config.pkl"), "wb") as f:
        pickle.dump(assets, f)

    # Copy the best checkpoint next to the config.
    ckpt_dir = manager.paths["checkpoints"]
    best = next(f for f in os.listdir(ckpt_dir) if "best" in f)
    shutil.copy(os.path.join(ckpt_dir, best), os.path.join(args.out, "model.pt"))

    print("Wrote asset bundle to: {}".format(args.out))
    print("  charset size : {}".format(len(assets["charset"])))
    print("  mean         : {}".format(assets["mean"]))
    print("  std          : {}".format(assets["std"]))
    print("  checkpoint   : {}".format(best))


if __name__ == "__main__":
    main()
