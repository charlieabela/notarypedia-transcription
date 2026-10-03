#  Adapted from the original main_syn_line.py for the R352 dataset.
#  Changes from original:
#    1. dataset_name = "R352"
#    2. generate_syn_dataset call uses "R352_line_syn"
#    3. output_folder renamed to "FCN_R352_line_syn"
#
#  Run from: faster_dan/OCR/line_OCR/ctc/
#      python3 main_syn_line.py

from faster_dan.OCR.line_OCR.ctc.trainer_line_ctc import TrainerLineCTC
from faster_dan.OCR.line_OCR.ctc.models_line_ctc import Decoder
from faster_dan.basic.models import FCN_Encoder
from faster_dan.basic.transforms import line_aug_config
from faster_dan.basic.scheduler import exponential_dropout_scheduler, linear_scheduler
from faster_dan.OCR.ocr_dataset_manager import OCRDataset, OCRDatasetManager
from torch.optim import Adam
import torch
import numpy as np
import random


def train_and_test(rank, params):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    params["training_params"]["ddp_rank"] = rank
    model = TrainerLineCTC(params)
    model.load_model()

    # Generate typed synthetic line images from R352 page transcriptions.
    # Output → ../../../Datasets/formatted/R352_line_syn/
    model.generate_syn_dataset("R352_line_syn")          # ← changed


def main():
    dataset_name  = "R352"    # ← changed (was "READ_2016")
    dataset_level = "page"

    params = {
        "dataset_params": {
            "dataset_manager": OCRDatasetManager,
            "dataset_class":   OCRDataset,
            "datasets": {
                dataset_name: "../../../Datasets/formatted/{}_{}".format(
                    dataset_name, dataset_level),
            },
            "train": {
                "name":     "{}-train".format(dataset_name),
                "datasets": [(dataset_name, "train"), ],
            },
            "valid": {
                "{}-valid".format(dataset_name): [(dataset_name, "valid"), ],
            },
            "config": {
                "load_in_memory":  True,
                "worker_per_gpu":  8,
                "width_divisor":   8,
                "height_divisor":  32,
                "padding_value":   0,
                "padding_token":   1000,
                "padding_mode":    "br",
                "charset_mode":    "CTC",
                "constraints":     ["CTC_line", ],
                "normalize":       True,
                "padding": {
                    "min_height":  "max",
                    "min_width":   "max",
                    "min_pad":     None,
                    "max_pad":     None,
                    "mode":        "br",
                    "train_only":  False,
                },
                "preprocessings": [
                    {"type": "to_RGB"},
                ],
                "augmentation": line_aug_config(0.9, 0.1),
                "synthetic_data": {
                    "mode":                   "line_hw_to_printed",
                    "init_proba":             1,
                    "end_proba":              1,
                    "num_steps_proba":        1e5,
                    "proba_scheduler_function": linear_scheduler,
                    "config": {
                        "background_color_default": (255, 255, 255),
                        "background_color_eps":     15,
                        "text_color_default":       (0, 0, 0),
                        "text_color_eps":           15,
                        "font_size_min":            30,
                        "font_size_max":            50,
                        "color_mode":               "RGB",
                        "padding_left_ratio_min":   0.02,
                        "padding_left_ratio_max":   0.1,
                        "padding_right_ratio_min":  0.02,
                        "padding_right_ratio_max":  0.1,
                        "padding_top_ratio_min":    0.02,
                        "padding_top_ratio_max":    0.2,
                        "padding_bottom_ratio_min": 0.02,
                        "padding_bottom_ratio_max": 0.2,
                        "max_width":                1500,
                    },
                },
                "charset": [],
            }
        },

        "model_params": {
            "models": {
                "encoder": FCN_Encoder,
                "decoder": Decoder,
            },
            "transfer_learning": None,
            "input_channels": 3,
            "enc_size":       256,
            "dropout_scheduler": {
                "function": exponential_dropout_scheduler,
                "T":        5e4,
            },
            "dropout": 0.5,
        },

        "training_params": {
            "output_folder":           "FCN_R352_line_syn",  # ← changed
            "max_nb_epochs":           10000,
            "max_training_time":       3600 * 24 * 1.9,
            "load_epoch":              "last",
            "interval_save_weights":   None,
            "use_ddp":                 False,
            "use_amp":                 True,
            "nb_gpu":                  torch.cuda.device_count(),
            "batch_size":              16,
            "optimizers": {
                "all": {
                    "class": Adam,
                    "args":  {"lr": 0.0001, "amsgrad": False},
                }
            },
            "lr_schedulers":           None,
            "eval_on_valid":           True,
            "eval_on_valid_interval":  2,
            "focus_metric":            "cer",
            "expected_metric_value":   "low",
            "set_name_focus_metric":   "{}-valid".format(dataset_name),
            "train_metrics":           ["loss", "cer", "wer"],
            "eval_metrics":            ["loss", "cer", "wer"],
            "force_cpu":               False,
        },
    }

    train_and_test(0, params)


if __name__ == "__main__":
    main()
