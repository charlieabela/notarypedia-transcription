"""
predict_r335_with_r352.py
==========================
Run the trained R352 FasterDAN model (best_4435.pt) on the R335 dataset.
Reports CER, WER, map_cer and loer for all splits via model.predict().
Also saves per-page prediction and ground truth text files.

Run from faster_dan/ root:
    python predict_r335_with_r352.py

Download predictions:
    scp -r <user>@<server>:<path-to>/FasterDAN/faster_dan/OCR/document_OCR/faster_dan/outputs/R335_predictions ~/Downloads/
"""

import os
import pickle
import torch
import numpy as np
import random
import editdistance
from torch.optim import Adam
from torch.utils.data import DataLoader

from faster_dan.OCR.ocr_dataset_manager import OCRDataset, OCRDatasetManager
from faster_dan.OCR.document_OCR.faster_dan.trainer_faster_dan import Manager
from faster_dan.OCR.document_OCR.faster_dan.models_faster_dan import GlobalHTADecoder
from faster_dan.basic.models import FCN_Encoder
from faster_dan.basic.scheduler import exponential_dropout_scheduler, linear_scheduler

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# ------------------------------------------------------------------ #
# Configuration                                                        #
# ------------------------------------------------------------------ #
CHECKPOINT_FILE = "best_4435.pt"
EXPECTED_EPOCH  = 4435
OUTPUT_FOLDER   = "OCR/document_OCR/faster_dan/outputs/fdan_R352_page_sem"
OUTPUT_DIR      = "OCR/document_OCR/faster_dan/outputs/R335_predictions"
R352_LABELS_PKL = "Datasets/formatted/R352_page_sem/labels.pkl"
SEM_TOKENS      = ['ⓑ', 'Ⓑ', 'ⓜ', 'Ⓜ']


def clean_text(text):
    for tok in SEM_TOKENS:
        text = text.replace(tok, ' ')
    text = text.replace('\n', ' ')
    return ' '.join(text.split())


def save_text(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main():

    # Load R352 charset to match model dimensions
    print("Loading R352 charset...")
    with open(R352_LABELS_PKL, "rb") as f:
        r352_charset = pickle.load(f)["charset"]
    print("R352 charset size: {}".format(len(r352_charset)))

    params = {
        "dataset_params": {
            "dataset_manager": OCRDatasetManager,
            "dataset_class":   OCRDataset,
            "charset":         r352_charset,   # force R352 charset
            "datasets": {
                "R335": "Datasets/formatted/R335_page_sem",
            },
            "train": {
                "name":     "R335-train",
                "datasets": [("R335", "train")],
            },
            "valid": {
                "R335-valid": [("R335", "valid")],
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
            "output_folder":          OUTPUT_FOLDER,
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
            "set_name_focus_metric":  "R335-valid",
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

    # ------------------------------------------------------------------ #
    # Load model and verify checkpoint                                    #
    # ------------------------------------------------------------------ #
    print("\nLoading R352 model...")
    model = Manager(params)
    model.load_model()
    model.models["encoder"].eval()
    model.models["decoder"].eval()

    print("\n" + "=" * 60)
    print("CHECKPOINT VERIFICATION")
    print("=" * 60)
    print("Expected  : {}".format(CHECKPOINT_FILE))
    print("Loaded epoch : {}".format(model.latest_epoch))
    if model.latest_epoch == EXPECTED_EPOCH:
        print("STATUS    : CORRECT")
    else:
        print("STATUS    : WARNING — expected {} got {}".format(
            EXPECTED_EPOCH, model.latest_epoch))
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------ #
    # Run model.predict() for full metrics (CER, WER, map_cer, loer)     #
    # ------------------------------------------------------------------ #
    metrics = ["cer", "wer", "map_cer", "loer"]

    for split in ["test", "valid", "train"]:
        print("=" * 60)
        print("R335 {} set".format(split.upper()))
        print("=" * 60)
        model.predict(
            "R335-{}".format(split),
            [("R335", split)],
            metrics,
            output=True
        )

    # ------------------------------------------------------------------ #
    # Also save per-page prediction text files                            #
    # ------------------------------------------------------------------ #
    print("\nSaving per-page prediction files...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_lines = [
        "R335 Predictions using R352 model\n",
        "Checkpoint: {} (epoch {})\n".format(CHECKPOINT_FILE, model.latest_epoch),
        "=" * 60 + "\n\n"
    ]

    for split in ["test", "valid", "train"]:
        split_dir = os.path.join(OUTPUT_DIR, split)
        os.makedirs(split_dir, exist_ok=True)

        if split == "train":
            dataset = model.dataset.train_dataset
            dataset.training_info = {"step": 0}
            dataset.params["config"]["synthetic_data"]["init_proba"] = 0
            dataset.params["config"]["synthetic_data"]["end_proba"]  = 0
        elif split == "valid":
            dataset = model.dataset.valid_datasets["R335-valid"]
        else:
            model.dataset.generate_test_loader(
                "R335-test", [("R335", "test")]
            )
            dataset = model.dataset.test_datasets["R335-test"]

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=model.dataset.my_collate_function,
            num_workers=0,
        )

        split_cer_total = 0
        split_chars     = 0
        split_summary   = ["Split: {}\n".format(split.upper()), "-"*40+"\n"]

        with torch.no_grad():
            for i, batch in enumerate(loader):
                values = model.evaluate_batch(batch, ["cer"])
                gt   = values["str_y"][0]
                pred = values["str_x"][0]
                name = values["names"][0]

                gt_clean   = clean_text(gt)
                pred_clean = clean_text(pred)
                cer = editdistance.eval(gt_clean, pred_clean) / max(len(gt_clean), 1)
                split_cer_total += editdistance.eval(gt_clean, pred_clean)
                split_chars     += max(len(gt_clean), 1)

                page_id = os.path.splitext(os.path.basename(name))[0]
                save_text(os.path.join(split_dir, "{}_pred.txt".format(page_id)), pred)
                save_text(os.path.join(split_dir, "{}_gt.txt".format(page_id)),   gt)
                split_summary.append("  {} | CER: {:.4f}\n".format(page_id, cer))

        overall_cer = split_cer_total / max(split_chars, 1)
        split_summary.append("\nOverall {} CER: {:.4f}\n\n".format(split.upper(), overall_cer))
        summary_lines.extend(split_summary)

    save_text(os.path.join(OUTPUT_DIR, "summary.txt"), "".join(summary_lines))

    print("\n" + "=" * 60)
    print("Predictions saved to : {}".format(OUTPUT_DIR))
    print("Download with:")
    print("  scp -r <user>@<server>:<path-to>/FasterDAN/faster_dan/{} ~/Downloads/".format(OUTPUT_DIR))
    print("=" * 60)


if __name__ == "__main__":
    main()
