#  Adapted from the original main_faster_dan.py for the R352 dataset.
#  Changes from original:
#    1. dataset_name = "R352"
#    2. dataset_variant = "_sem"
#    3. max_nb_lines set directly (no dict lookup)
#    4. transfer_learning points to FCN_R352_line_syn checkpoint
#    5. output_folder = "fdan_R352_page_sem"
#
#  Run from: faster_dan/OCR/document_OCR/faster_dan/
#      python3 main_faster_dan.py

from torch.optim import Adam
from faster_dan.basic.transforms import aug_config
from faster_dan.OCR.ocr_dataset_manager import OCRDataset, OCRDatasetManager
from faster_dan.OCR.document_OCR.faster_dan.trainer_faster_dan import Manager
from faster_dan.OCR.document_OCR.faster_dan.models_faster_dan import GlobalHTADecoder
from faster_dan.basic.models import FCN_Encoder
from faster_dan.basic.scheduler import exponential_dropout_scheduler, linear_scheduler
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
    model = Manager(params)
    model.load_model()

    model.train()

    # Load weights giving best CER on valid set
    model.params["training_params"]["load_epoch"] = "best"
    model.load_model()

    metrics = ["cer", "wer", "time", "map_cer", "loer"]

    for dataset_name in params["dataset_params"]["datasets"].keys():
        for set_name in ["test", "valid", "train"]:
            model.predict("{}-{}".format(dataset_name, set_name),
                          [(dataset_name, set_name)], metrics, output=True)


if __name__ == "__main__":

    dataset_name    = "R352"       # ← changed
    dataset_level   = "page"
    dataset_variant = "_sem"       # ← semantic tokens enabled

    # R352: ~30 lines per page maximum (body + margins combined)
    max_nb_lines = 62              # ← direct value, no dict lookup

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
                "datasets": [(dataset_name, "train"), ],
            },
            "valid": {
                "{}-valid".format(dataset_name): [(dataset_name, "valid"), ],
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
                "preprocessings": [
                    {"type": "to_RGB"},
                ],
                "augmentation": aug_config(0.9, 0.1),
                "synthetic_data": {
                    "mode":                      "document",
                    "page_syn_mode":             "typed",
                    "init_proba":                0.9,
                    "end_proba":                 0.2,
                    "num_steps_proba":           200000,
                    "proba_scheduler_function":  linear_scheduler,
                    "start_scheduler_at_max_line": True,
                    "dataset_level":             dataset_level,
                    "curriculum":                True,
                    "crop_curriculum":           True,
                    "curr_start":                0,
                    "curr_step":                 10000,
                    "min_nb_lines":              1,
                    "max_nb_lines":              max_nb_lines,  # ← 30
                    "padding_value":             255,
                    "max_char_per_line":         100,
                    "mix_paragraphs":            True,
                    "rimes_sem_order":           False,
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
            # Transfer encoder and decoder from CTC line pretraining
            "transfer_learning": {
                # [state_dict_name, checkpoint_path, learnable, strict]
                "encoder": ["encoder", "../../../outputs/FCN_R352_line_syn/checkpoints/best_1040.pt", True, True],
                "decoder": ["decoder", "../../../outputs/FCN_R352_line_syn/checkpoints/best_1040.pt", True, False],
            },
            "transfered_charset":  True,
            "additional_tokens":   1,      # for <eot>

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
            "output_folder":           "fdan_R352_page_sem",  # ← changed
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
            "train_metrics":           ["loss", "cer", "wer", "syn_max_lines"],
            "eval_metrics":            ["cer", "wer", "map_cer","loer", "cer_first_pass"],
            "force_cpu":               False,
            "max_line_pred":           100,
            "max_pred_per_line":       150,
            "teacher_forcing_scheduler": {
                "min_error_rate":   0.2,
                "max_error_rate":   0.2,
                "total_num_steps":  5e4,
            },
        },
    }

    syn_config = params["dataset_params"]["config"]["synthetic_data"]
    params["training_params"]["start_valid_from_steps"] = (
        syn_config["curr_start"] +
        syn_config["curr_step"] * (syn_config["max_nb_lines"] - syn_config["min_nb_lines"])
    )

    train_and_test(0, params)
