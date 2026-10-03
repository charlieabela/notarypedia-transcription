#  FasterDAN training on combined R352 + R335 dataset.
#  Uses the R352 CTC pretrained encoder (best_1040.pt).
#  Output folder: fdan_R352_R335_page_sem_marginwrap (separate from all other models)
#
#  At the end of training, saves per-page predictions for all splits to:
#      outputs/fdan_R352_R335_page_sem_marginwrap/predictions/
#          R352_R335-test/   R352_R335-valid/   R352_R335-train/
#              <page>_pred.txt  <page>_gt.txt
#          summary.txt
#
#  Run from: faster_dan/OCR/document_OCR/faster_dan/
#      python3 main_faster_dan_combined.py

import os
import editdistance
import torch
import numpy as np
import random
from torch.optim import Adam
from torch.utils.data import DataLoader

from faster_dan.basic.transforms import aug_config
from faster_dan.OCR.ocr_dataset_manager import OCRDataset, OCRDatasetManager
from faster_dan.OCR.document_OCR.faster_dan.trainer_faster_dan import Manager
from faster_dan.OCR.document_OCR.faster_dan.models_faster_dan import GlobalHTADecoder
from faster_dan.basic.models import FCN_Encoder
from faster_dan.basic.scheduler import exponential_dropout_scheduler, linear_scheduler

# Python 3.13's multiprocessing has a known regression with PyTorch's default
# "file_descriptor" tensor-sharing strategy between DataLoader worker
# processes: shared-tensor file descriptors accumulate over the run and
# eventually exceed the OS open-file limit (OSError: Too many open files),
# which is exactly what crashed this run at epoch 122. Switching to the
# "file_system" strategy avoids the fd-based handoff entirely. Must be set
# before any DataLoader workers spawn, so this needs to happen at import
# time, before train_and_test() constructs the dataset/DataLoader.
torch.multiprocessing.set_sharing_strategy("file_system")


SEM_TOKENS   = ['ⓑ', 'Ⓑ', 'ⓜ', 'Ⓜ']
PRED_OUT_DIR = "outputs/fdan_R352_R335_page_sem_marginwrap/predictions"


def clean_text(text):
    for tok in SEM_TOKENS:
        text = text.replace(tok, ' ')
    text = text.replace('\n', ' ')
    return ' '.join(text.split())


def save_text(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def save_predictions(model, dataset_name, params):
    """
    Run inference on all splits and save per-page prediction text files.
    Called automatically at the end of training.
    """
    print("\n" + "=" * 60)
    print("SAVING PREDICTIONS — {}".format(dataset_name))
    print("=" * 60)

    summary_lines = [
        "Combined model predictions — {}\n".format(dataset_name),
        "Checkpoint: best epoch {}\n".format(model.latest_epoch),
        "=" * 60 + "\n",
        "CER/WER computed on plain text (semantic tokens stripped)\n\n"
    ]

    for split in ["test", "valid", "train"]:
        split_dir = os.path.join(PRED_OUT_DIR, "{}-{}".format(dataset_name, split))
        os.makedirs(split_dir, exist_ok=True)

        if split == "train":
            dataset = model.dataset.train_dataset
            dataset.training_info = {"step": 0}
            dataset.params["config"]["synthetic_data"]["init_proba"] = 0
            dataset.params["config"]["synthetic_data"]["end_proba"]  = 0
        elif split == "valid":
            dataset = model.dataset.valid_datasets["{}-valid".format(dataset_name)]
        else:
            model.dataset.generate_test_loader(
                "{}-test".format(dataset_name),
                [(dataset_name, split)]
            )
            dataset = model.dataset.test_datasets["{}-test".format(dataset_name)]

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=model.dataset.my_collate_function,
            num_workers=0,
        )

        split_cer_total = 0
        split_wer_total = 0
        split_chars     = 0
        split_words     = 0
        split_summary   = [
            "Split: {}\n".format(split.upper()),
            "-" * 40 + "\n"
        ]

        print("\nProcessing {} {}...".format(dataset_name, split.upper()))

        with torch.no_grad():
            for i, batch in enumerate(loader):
                values = model.evaluate_batch(batch, ["cer", "wer"])

                gt   = values["str_y"][0]
                pred = values["str_x"][0]
                name = values["names"][0] if "names" in values else "{}_{}".format(split, i)

                gt_clean   = clean_text(gt)
                pred_clean = clean_text(pred)

                cer = editdistance.eval(gt_clean, pred_clean) / max(len(gt_clean), 1)
                gt_words   = gt_clean.split()
                pred_words = pred_clean.split()
                wer = editdistance.eval(gt_words, pred_words) / max(len(gt_words), 1)

                split_cer_total += editdistance.eval(gt_clean, pred_clean)
                split_wer_total += editdistance.eval(gt_words, pred_words)
                split_chars     += max(len(gt_clean), 1)
                split_words     += max(len(gt_words), 1)

                page_id = os.path.splitext(os.path.basename(name))[0]
                save_text(os.path.join(split_dir, "{}_pred.txt".format(page_id)), pred)
                save_text(os.path.join(split_dir, "{}_gt.txt".format(page_id)),   gt)

                page_line = "  {} | CER: {:.4f} | WER: {:.4f}".format(page_id, cer, wer)
                split_summary.append(page_line + "\n")
                print(page_line)

        overall_cer  = split_cer_total / max(split_chars, 1)
        overall_wer  = split_wer_total / max(split_words, 1)
        overall_line = "\nOverall {} {} | CER: {:.4f} | WER: {:.4f}\n\n".format(
            dataset_name, split.upper(), overall_cer, overall_wer)
        split_summary.append(overall_line)
        summary_lines.extend(split_summary)
        print(overall_line.strip())

    summary_path = os.path.join(PRED_OUT_DIR, "summary_{}.txt".format(dataset_name))
    save_text(summary_path, "".join(summary_lines))
    print("Summary saved to: {}".format(summary_path))


def train_and_test(rank, params):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    params["training_params"]["ddp_rank"] = rank
    model = Manager(params)
    model.load_model()
    model.train()

    # Load best checkpoint for evaluation and prediction saving
    model.params["training_params"]["load_epoch"] = "best"
    model.load_model()

    print("\nBest epoch: {} | Best CER: {}".format(
        model.latest_epoch, round(model.best, 4) if model.best else "N/A"))

    # Standard metrics evaluation
    metrics = ["cer", "wer", "time", "map_cer", "loer"]
    dataset_name = list(params["dataset_params"]["datasets"].keys())[0]
    for set_name in ["test", "valid", "train"]:
        model.predict(
            "{}-{}".format(dataset_name, set_name),
            [(dataset_name, set_name)],
            metrics,
            output=True
        )

    # Save per-page prediction text files
    model.models["encoder"].eval()
    model.models["decoder"].eval()
    save_predictions(model, dataset_name, params)

    print("\n" + "=" * 60)
    print("All predictions saved to: {}".format(PRED_OUT_DIR))
    print("\nTo download to your Mac:")
    print("  scp -r <user>@<server>:<path-to>/FasterDAN/faster_dan/OCR/document_OCR/faster_dan/{} ~/Downloads/".format(
        PRED_OUT_DIR))
    print("=" * 60)


if __name__ == "__main__":

    dataset_name    = "R352_R335"
    dataset_level   = "page"
    dataset_variant = "_sem"
    max_nb_lines    = 62

    params = {
        "dataset_params": {
            "dataset_manager": OCRDatasetManager,
            "dataset_class":   OCRDataset,
            "datasets": {
                dataset_name: "../../../Datasets/formatted/{}_{}{}".format(
                    dataset_name, dataset_level, dataset_variant),
            },
            "train": {
                "name":     "{}-train".format(dataset_name),
                "datasets": [(dataset_name, "train")],
            },
            "valid": {
                "{}-valid".format(dataset_name): [(dataset_name, "valid")],
            },
            "config": {
                "balance_datasets": True,
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
                "augmentation":     aug_config(0.9, 0.1),
                "synthetic_data": {
                    "mode":                        "document",
                    "page_syn_mode":               "typed",
                    "init_proba":                  0.9,
                    "end_proba":                   0.2,
                    "num_steps_proba":             200000,
                    "proba_scheduler_function":    linear_scheduler,
                    "start_scheduler_at_max_line": True,
                    "dataset_level":               dataset_level,
                    "curriculum":                  True,
                    "crop_curriculum":             True,
                    "curr_start":                  0,
                    "curr_step":                   10000,
                    "min_nb_lines":                1,
                    "max_nb_lines":                max_nb_lines,
                    "padding_value":               255,
                    "max_char_per_line":           100,
                    "mix_paragraphs":              True,
                    "rimes_sem_order":             False,
                    # Probability that a synthetic R352 margin note, once it
                    # runs out of room in its own perimeter band (top/left/
                    # right/bottom), continues ("wraps") into a neighbouring
                    # band instead of stopping -- this is what trains the
                    # model on pages where marginalia spill across corners
                    # (e.g. top margin spilling into the left margin), while
                    # the body region always stays an inviolable keep-out
                    # zone. Starts at 0 so this run begins on the clean
                    # two-region case before ramping into the harder
                    # spillover layouts, mirroring how curr_start/curr_step
                    # ramp synthetic line count elsewhere in this config.
                    "margin_wrap_proba":           0.0,
                    "config": {
                        "background_color_default": (255, 255, 255),
                        "background_color_eps":     15,
                        "text_color_default":       (0, 0, 0),
                        "text_color_eps":           15,
                        "font_size_min":            35,
                        "font_size_max":            45,
                        "color_mode":               "RGB",
                        "padding_left_ratio_min":   0.00,
                        "padding_left_ratio_max":   0.05,
                        "padding_right_ratio_min":  0.02,
                        "padding_right_ratio_max":  0.2,
                        "padding_top_ratio_min":    0.02,
                        "padding_top_ratio_max":    0.1,
                        "padding_bottom_ratio_min": 0.02,
                        "padding_bottom_ratio_max": 0.1,
                    },
                }
            }
        },

        "model_params": {
            "models": {
                "encoder": FCN_Encoder,
                "decoder": GlobalHTADecoder,
            },
            # Reuse R352 CTC pretrained encoder
            "transfer_learning": {

		"encoder": ["encoder", "../../../outputs/FCN_R352_line_syn/checkpoints/best_1040.pt", True, True],
		"decoder": ["decoder", "../../../outputs/FCN_R352_line_syn/checkpoints/best_1040.pt", True, False],
            },
            "transfered_charset":  True,
            "additional_tokens":   1,
            "input_channels":      3,
            "dropout":             0.5,
            "enc_dim":             256,
            "nb_layers":           5,
            "pe_h_max":            500,
            "pe_w_max":            1000,
            "l_max":               15000,
            "dec_num_layers":      8,
            "dec_num_heads":       4,
            "dec_res_dropout":     0.1,
            "dec_pred_dropout":    0.1,
            "dec_att_dropout":     0.1,
            "dec_dim_feedforward": 256,
            "attention_win":       100,
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
            "output_folder":           "fdan_R352_R335_page_sem_marginwrap",
            "max_nb_epochs":           50000,
            "max_training_time":       3600 * 24 * 1.9,
            "load_epoch":              "last",
            "interval_save_weights":   None,
            "batch_size":              1,
            "valid_batch_size":        4,
            "test_batch_size":         1,
            "use_amp":                 True,
            "nb_gpu":                  torch.cuda.device_count(),
            "optimizers": {
                "all": {
                    "class": Adam,
                    "args":  {"lr": 0.0001, "amsgrad": False},
                },
            },
            "lr_schedulers":           None,
            "eval_on_valid":           True,
            "eval_on_valid_interval":  5,
            "focus_metric":            "cer",
            "expected_metric_value":   "low",
            "set_name_focus_metric":   "{}-valid".format(dataset_name),
            "train_metrics":           ["loss", "cer", "wer", "syn_max_lines", "margin_wrap_rate"],
            "eval_metrics":            ["cer", "wer", "map_cer", "loer", "cer_first_pass"],
            "force_cpu":               False,
            "max_line_pred":           100,
            "max_pred_per_line":       150,
            "teacher_forcing_scheduler": {
                "min_error_rate":  0.2,
                "max_error_rate":  0.2,
                "total_num_steps": 5e4,
            },
        },
    }

    syn_config = params["dataset_params"]["config"]["synthetic_data"]
    params["training_params"]["start_valid_from_steps"] = (
        syn_config["curr_start"] +
        syn_config["curr_step"] * (syn_config["max_nb_lines"] - syn_config["min_nb_lines"])
    )

    train_and_test(0, params)
