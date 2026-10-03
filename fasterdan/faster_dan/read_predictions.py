"""
read_predictions.py
====================
Runs the best R352 FasterDAN checkpoint on a few validation images and
prints ground truth vs prediction side by side.

Run from faster_dan/ root:
    python read_predictions.py

Optional args:
    --n       number of validation pages to show (default 5)
    --split   train / valid / test (default valid)
"""

import argparse
import os
import sys
import torch
import numpy as np
import random

# ------------------------------------------------------------------ #
# Reproducibility                                                      #
# ------------------------------------------------------------------ #
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from torch.optim import Adam
from faster_dan.basic.transforms import aug_config
from faster_dan.OCR.ocr_dataset_manager import OCRDataset, OCRDatasetManager
from faster_dan.OCR.document_OCR.faster_dan.trainer_faster_dan import Manager
from faster_dan.OCR.document_OCR.faster_dan.models_faster_dan import GlobalHTADecoder
from faster_dan.basic.models import FCN_Encoder
from faster_dan.basic.scheduler import exponential_dropout_scheduler, linear_scheduler
from faster_dan.OCR.ocr_utils import LM_ind_to_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",     type=int, default=5,       help="Number of pages to show")
    parser.add_argument("--split", type=str, default="valid",  help="train / valid / test")
    args = parser.parse_args()

    dataset_name    = "R352"
    dataset_level   = "page"
    dataset_variant = "_sem"
    max_nb_lines    = 62

    params = {
        "dataset_params": {
            "dataset_manager": OCRDatasetManager,
            "dataset_class":   OCRDataset,
            "datasets": {
                dataset_name: "Datasets/formatted/{}_{}{}".format(
                    dataset_name, dataset_level, dataset_variant),
            },
            "train": {
                "name":     "{}-train".format(dataset_name),
                "datasets": [(dataset_name, "train"), ],
            },
            "valid": {
                "{}-valid".format(dataset_name): [(dataset_name, "valid"), ],
            },
            "config": {
                "balance_datasets": False,
                "load_in_memory":   True,
                "worker_per_gpu":   2,
                "width_divisor":    8,
                "height_divisor":   32,
                "padding_value":    0,
                "padding_token":    None,
                "charset_mode":     "seq2seq",
                "constraints":      ["add_eot", "add_sot", "fdan_encoding"],
                "normalize":        True,
                "preprocessings":   [{"type": "to_RGB"}],
                "augmentation":     None,   # no augmentation during inference
                "synthetic_data":   None,   # no synthetic data during inference
            }
        },

        "model_params": {
            "models": {
                "encoder": FCN_Encoder,
                "decoder": GlobalHTADecoder,
            },
            "transfer_learning":  None,
            "transfered_charset": True,
            "additional_tokens":  1,
            "input_channels":     3,
            "dropout":            0.5,
            "enc_dim":            256,
            "nb_layers":          5,
            "pe_h_max":           500,
            "pe_w_max":           1000,
            "l_max":              15000,
            "dec_num_layers":     8,
            "dec_num_heads":      4,
            "dec_res_dropout":    0.1,
            "dec_pred_dropout":   0.1,
            "dec_att_dropout":    0.1,
            "dec_dim_feedforward":256,
            "attention_win":      100,
            "use_tokens_from_all_lines": True,
            "use_first_pass_tokens":     True,
            "use_line_indices":          True,
            "two_step_pos_enc_mode":     "cat",
            "dropout_scheduler": {
                "function": exponential_dropout_scheduler,
                "T":        5e4,
            }
        },

        "training_params": {
            "output_folder":          "fdan_R352_page_sem",
            "max_nb_epochs":          50000,
            "max_training_time":      3600 * 24 * 14,
            "load_epoch":             "best",   # ← load best checkpoint
            "interval_save_weights":  None,
            "batch_size":             1,
            "valid_batch_size":       1,
            "test_batch_size":        1,
            "use_amp":                True,
            "nb_gpu":                 torch.cuda.device_count(),
            "optimizers": {
                "all": {
                    "class": Adam,
                    "args":  {"lr": 0.0001, "amsgrad": False},
                },
            },
            "lr_schedulers":          None,
            "eval_on_valid":          True,
            "eval_on_valid_interval": 5,
            "focus_metric":           "cer",
            "expected_metric_value":  "low",
            "set_name_focus_metric":  "{}-valid".format(dataset_name),
            "train_metrics":          ["loss", "cer", "wer", "syn_max_lines"],
            "eval_metrics":           ["cer", "wer", "map_cer", "loer", "cer_first_pass"],
            "force_cpu":              False,
            "max_line_pred":          100,
            "max_pred_per_line":      150,
            "teacher_forcing_scheduler": {
                "min_error_rate":  0.2,
                "max_error_rate":  0.2,
                "total_num_steps": 5e4,
            },
            "ddp_rank": 0,
        },
    }

    # Initialise model and load best checkpoint
    model = Manager(params)
    model.load_model()
    model.models["encoder"].eval()
    model.models["decoder"].eval()

    print("\n" + "="*70)
    print("Loaded checkpoint | split: {} | showing {} pages".format(
        args.split, args.n))
    print("="*70)

    # Get the right dataset
    if args.split == "valid":
        dataset = model.dataset.valid_datasets["{}-valid".format(dataset_name)]
    elif args.split == "train":
        dataset = model.dataset.train_dataset
    else:
        model.dataset.generate_test_loader(
            "{}-test".format(dataset_name),
            [(dataset_name, "test")]
        )
        dataset = model.dataset.test_datasets["{}-test".format(dataset_name)]

    indices = list(range(min(args.n, len(dataset))))

    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx]

            # Collate single sample into batch
            batch = dataset.collate_function([sample])

            # Run evaluation
            values = model.evaluate_batch(batch, ["cer"])

            gt   = values["str_y"][0]
            pred = values["str_x"][0]
            name = values["names"][0]

            # Token presence
            gt_has_b = "ⓑ" in gt or "Ⓑ" in gt
            gt_has_m = "ⓜ" in gt or "Ⓜ" in gt
            pr_has_b = "ⓑ" in pred or "Ⓑ" in pred
            pr_has_m = "ⓜ" in pred or "Ⓜ" in pred

            # CER
            import editdistance
            cer = editdistance.eval(gt, pred) / max(len(gt), 1)

            print("\n--- Page {} ---".format(name))
            print("CER: {:.4f}".format(cer))
            print("GT tokens  — body: {} | margin: {}".format(gt_has_b, gt_has_m))
            print("PRED tokens — body: {} | margin: {}".format(pr_has_b, pr_has_m))
            print()
            print("GROUND TRUTH (first 300 chars):")
            print(repr(gt[:300]))
            print()
            print("PREDICTION (first 300 chars):")
            print(repr(pred[:300]))
            print("-"*70)


if __name__ == "__main__":
    main()
