"""
predict_combined_model.py
==========================
Run the trained combined R352+R335 FasterDAN model on both datasets.
Loads best checkpoint from fdan_R352_R335_page_sem.

Run from faster_dan/ root:
    conda activate fdan_env
    ulimit -n 65536
    python predict_combined_model.py

Output files saved to:
    OCR/document_OCR/faster_dan/outputs/combined_predictions/
        R352/
            test/   <page>_pred.txt  <page>_gt.txt
            valid/
            train/
        R335/
            test/
            valid/
            train/
        summary.txt   <- CER/WER/map_cer/loer per split and overall
"""

import os
import torch
import numpy as np
import random
from torch.optim import Adam

from faster_dan.OCR.ocr_dataset_manager import OCRDataset, OCRDatasetManager
from faster_dan.OCR.document_OCR.faster_dan.trainer_faster_dan import Manager
from faster_dan.OCR.document_OCR.faster_dan.models_faster_dan import GlobalHTADecoder
from faster_dan.basic.models import FCN_Encoder
from faster_dan.basic.scheduler import exponential_dropout_scheduler, linear_scheduler
from faster_dan.basic.metric_manager import MetricManager, compute_global_mAP

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

OUTPUT_DIR = "OCR/document_OCR/faster_dan/outputs/combined_predictions"
METRIC_NAMES = ["cer", "wer", "map_cer", "loer"]


def save_text(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def build_params():
    return {
        "dataset_params": {
            "dataset_manager": OCRDatasetManager,
            "dataset_class":   OCRDataset,
            "datasets": {
                "R352_R335": "../../../Datasets/formatted/R352_R335_page_sem",
                "R352":      "../../../Datasets/formatted/R352_page_sem",
                "R335":      "../../../Datasets/formatted/R335_page_sem",
            },
            "train": {
                "name":     "R352_R335-train",
                "datasets": [("R352_R335", "train")],
            },
            "valid": {
                "R352_R335-valid": [("R352_R335", "valid")],
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
                "augmentation":     None,
                "synthetic_data": {
                    "mode":                      "line_hw_to_printed",
                    "init_proba":                0,
                    "end_proba":                 0,
                    "num_steps_proba":           1,
                    "proba_scheduler_function":  linear_scheduler,
                    "config": {
                        "background_color_default": (255, 255, 255),
                        "background_color_eps":     0,
                        "text_color_default":       (0, 0, 0),
                        "text_color_eps":           0,
                        "font_size_min":            35,
                        "font_size_max":            45,
                        "color_mode":               "RGB",
                        "padding_left_ratio_min":   0.02,
                        "padding_left_ratio_max":   0.02,
                        "padding_right_ratio_min":  0.02,
                        "padding_right_ratio_max":  0.02,
                        "padding_top_ratio_min":    0.02,
                        "padding_top_ratio_max":    0.02,
                        "padding_bottom_ratio_min": 0.02,
                        "padding_bottom_ratio_max": 0.02,
                        "max_width":                1500,
                    },
                },
                "charset": [],
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
            "dec_dim_feedforward": 256,
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
            "output_folder":          "fdan_R352_R335_page_sem",
            "max_nb_epochs":          50000,
            "max_training_time":      3600 * 24 * 14,
            "load_epoch":             "best",
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
            "eval_on_valid":          False,
            "eval_on_valid_interval": 5,
            "focus_metric":           "cer",
            "expected_metric_value":  "low",
            "set_name_focus_metric":  "R352_R335-valid",
            "train_metrics":          ["loss", "cer", "wer"],
            "eval_metrics":           ["cer", "wer", "map_cer", "loer"],
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


def get_loaded_epoch(model):
    """Retrieve the epoch number that was actually restored from the checkpoint."""
    for attr in ("latest_epoch", "best_epoch", "current_epoch", "epoch"):
        if hasattr(model, attr):
            val = getattr(model, attr)
            if val is not None:
                return val
    load_epoch = model.params.get("training_params", {}).get("load_epoch", "best")
    return "best (epoch number unavailable — attr not exposed by Manager)"


def run_inference_on_dataset(model, dataset_key, split, split_dir):
    """
    Run inference on a single split and save per-page prediction files.

    Mirrors the predict() loop in generic_training_manager.py:
      - uses model.dataset.test_loaders (built by generate_test_loader with the
        correct my_collate_function, avoiding the collate bug that caused the
        'list indices must be integers or slices, not str' error on train sets)
      - uses MetricManager.compute_metrics() for map_cer and loer

    Returns
    -------
    overall      : dict  {cer, wer, map_cer, loer, ...}
    page_results : list[str]
    """
    set_label = "{}-{}".format(dataset_key, split)

    # generate_test_loader builds the loader with the correct my_collate_function
    model.dataset.generate_test_loader(set_label, [(dataset_key, split)])
    loader = model.dataset.test_loaders[set_label]

    os.makedirs(split_dir, exist_ok=True)

    # Fresh MetricManager per split — same pattern as generic_training_manager.predict()
    mm = MetricManager(metric_names=METRIC_NAMES, dataset_name=dataset_key)

    page_results = []

    with torch.no_grad():
        for i, batch_data in enumerate(loader):

            # --- inference ---
            batch_values = model.evaluate_batch(batch_data, METRIC_NAMES)

            # --- compute metrics via MetricManager (the only correct way for map_cer/loer) ---
            batch_metrics = mm.compute_metrics(batch_values, METRIC_NAMES)
            batch_metrics["names"] = batch_data["names"]
            batch_metrics["ids"]   = batch_data["ids"]
            mm.update_metrics(batch_metrics)

            # --- per-page scalars (batch_size == 1) ---
            page_cer = (
                batch_metrics["edit_chars"][0] /
                max(batch_metrics["nb_chars"][0], 1)
            )
            page_wer = (
                batch_metrics["edit_words"][0] /
                max(batch_metrics["nb_words"][0], 1)
            )
            # map_cer: weighted mAP computed over this single sample's AP dict
            page_map_cer = compute_global_mAP(batch_metrics["map_cer"])
            # loer: graph-edit / num-items for this single sample
            page_loer = (
                batch_metrics["edit_graph"][0] /
                max(batch_metrics["nb_nodes_and_edges"][0], 1)
            )

            # --- save per-page text files ---
            gt      = batch_values["str_y"][0]
            pred    = batch_values["str_x"][0]
            name    = batch_data["names"][0]
            page_id = os.path.splitext(os.path.basename(name))[0]

            save_text(os.path.join(split_dir, "{}_pred.txt".format(page_id)), pred)
            save_text(os.path.join(split_dir, "{}_gt.txt".format(page_id)),   gt)

            line = (
                "  {} | CER: {:.4f} | WER: {:.4f} | "
                "map_cer: {:.4f} | loer: {:.4f}".format(
                    page_id, page_cer, page_wer, page_map_cer, page_loer
                )
            )
            page_results.append(line)
            print(line)

    # --- epoch-level metrics aggregated by MetricManager ---
    overall = mm.get_display_values()
    print(
        "\nOverall {} {} | CER: {:.4f} | WER: {:.4f} | "
        "map_cer: {:.4f} | loer: {:.4f}".format(
            dataset_key, split.upper(),
            overall.get("cer",     0.0),
            overall.get("wer",     0.0),
            overall.get("map_cer", 0.0),
            overall.get("loer",    0.0),
        )
    )

    return overall, page_results


def main():
    print("Building params and loading combined model (best checkpoint)...")
    params = build_params()
    model = Manager(params)
    model.load_model()
    model.models["encoder"].eval()
    model.models["decoder"].eval()

    loaded_epoch = get_loaded_epoch(model)
    print("Combined model loaded.  Loaded epoch: {}\n".format(loaded_epoch))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_lines = [
        "Combined R352+R335 Model Inference Results\n",
        "=" * 60 + "\n",
        "Loaded epoch : {}\n".format(loaded_epoch),
        "=" * 60 + "\n\n",
    ]

    eval_targets = [
        ("R352",      "R352"),
        ("R335",      "R335"),
        ("R352_R335", "R352_R335"),
    ]

    for dataset_key, folder_name in eval_targets:
        summary_lines.append("Dataset: {}\n".format(dataset_key))
        summary_lines.append("-" * 40 + "\n")

        for split in ["test", "valid", "train"]:
            print("=" * 60)
            print("{} {} SET".format(dataset_key, split.upper()))
            print("=" * 60)

            split_dir = os.path.join(OUTPUT_DIR, folder_name, split)

            try:
                overall, page_results = run_inference_on_dataset(
                    model, dataset_key, split, split_dir
                )
                summary_lines.append(
                    "  {} | CER: {:.4f} | WER: {:.4f} | "
                    "map_cer: {:.4f} | loer: {:.4f}\n".format(
                        split.upper(),
                        overall.get("cer",     0.0),
                        overall.get("wer",     0.0),
                        overall.get("map_cer", 0.0),
                        overall.get("loer",    0.0),
                    )
                )
                for r in page_results:
                    summary_lines.append(r + "\n")

            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print("ERROR on {} {}: {}\n{}".format(dataset_key, split, e, tb))
                summary_lines.append("  {} ERROR: {}\n".format(split.upper(), e))

        summary_lines.append("\n")

    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    save_text(summary_path, "".join(summary_lines))
    print("\nAll predictions saved to: {}".format(OUTPUT_DIR))
    print("Summary saved to:         {}".format(summary_path))
    print("\nTo download:")
    print("  scp -r <user>@<server>:<path-to>/FasterDAN/faster_dan/{} ~/Downloads/".format(OUTPUT_DIR))


if __name__ == "__main__":
    main()
