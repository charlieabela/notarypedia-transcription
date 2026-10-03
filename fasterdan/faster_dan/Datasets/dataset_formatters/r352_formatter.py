#  Formatter for R352 / R335 / combined R352+R335 Maltese notarial datasets.
#  Modelled exactly on read2016_formatter.py / OCRDatasetFormatter.
#  Level: "page" — semantic tokens ⓑ/Ⓑ (body) and ⓜ/Ⓜ (margin).
#
#  Raw data layout on server:
#      Datasets/raw/R352/R352_001_no_symbols/page/   ← R352 XMLs
#      Datasets/raw/R352/R352_001_no_symbols/         ← R352 JPGs
#      Datasets/raw/R335/r335_009/page/               ← R335 XMLs
#      Datasets/raw/R335/r335_009/                    ← R335 JPGs
#
#  Outputs:
#      Datasets/formatted/R352_page_sem/      ← R352 only
#      Datasets/formatted/R335_page_sem/      ← R335 only (OOV stripped for R352 model inference)
#      Datasets/formatted/R352_R335_page_sem/ ← combined  (full charset, no stripping)
#
#  Usage:
#      python Datasets/dataset_formatters/r352_formatter.py --dataset R352
#      python Datasets/dataset_formatters/r352_formatter.py --dataset R335
#      python Datasets/dataset_formatters/r352_formatter.py --dataset R352_R335

import os
import re
import random
import xml.etree.ElementTree as ET
import argparse

from faster_dan.Datasets.dataset_formatters.generic_dataset_formatter import OCRDatasetFormatter
from faster_dan.Datasets.dataset_formatters.utils_dataset import natural_sort


PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"

# Semantic tokens — same convention as READ 2016
SEM_TOKENS = {
    "body":   "ⓑ",
    "margin": "ⓜ",
}

# Region type → semantic category
TOP_MARGIN_TYPES    = {"Margin1", "Margin1a", "Foliation"}
SIDE_MARGIN_TYPES   = {"Margin2", "Margin2a", "Margin2b", "Margin3"}
BOTTOM_MARGIN_TYPES = {"Margin4"}

# Characters in R335 not present in R352 charset.
# Stripped ONLY when formatting R335 standalone — this allows the R352
# model to run inference on R335 without embedding dimension mismatch.
# NOT stripped for combined R352+R335 training (model learns full charset).
R335_OOV_CHARS = {'\t', '#', '='}


def _region_custom_type(custom_attr):
    m = re.search(r'type:([^;}\s]+)', custom_attr)
    return m.group(1).strip() if m else None


def _strip_r335_oov(text):
    """Remove R335-only characters not present in R352 charset."""
    for ch in R335_OOV_CHARS:
        text = text.replace(ch, '')
    return text


class R352DatasetFormatter(OCRDatasetFormatter):

    def __init__(self, level, dataset_name="R352",
                 set_names=("train", "valid", "test"),
                 train_ratio=0.8, valid_ratio=0.1, seed=42):

        super(R352DatasetFormatter, self).__init__(
            dataset_name, level, extra_name="_sem", set_names=list(set_names)
        )
        self.map_datasets_files.update({
            dataset_name: {
                "page": {
                    "arx_files":       [],
                    "needed_files":    [],
                    "format_function": self.format_r352_page,
                },
            }
        })
        self.train_ratio  = train_ratio
        self.valid_ratio  = valid_ratio
        self.seed         = seed
        self.tokens       = SEM_TOKENS
        self.dataset_name = dataset_name

        # Strip OOV chars only for R335 standalone (for R352 model inference)
        # Combined and R352-only formatting keeps all characters
        self._strip_oov = (dataset_name == "R335")

        if dataset_name == "R352_R335":
            self._source_paths = [
                self._resolve_paths("R352"),
                self._resolve_paths("R335"),
            ]
        else:
            self._source_paths = [self._resolve_paths(dataset_name)]

    def _resolve_paths(self, name):
        """Auto-detect the subfolder inside Datasets/raw/<name>/"""
        base = os.path.join("./Datasets/raw", name)
        subfolders = [f for f in os.listdir(base)
                      if os.path.isdir(os.path.join(base, f))]
        if not subfolders:
            raise FileNotFoundError("No subfolder found in {}".format(base))
        subfolder = subfolders[0]
        return {
            "xml_fold": os.path.join(base, subfolder, "page"),
            "img_fold": os.path.join(base, subfolder),
            "name":     name,
        }

    # ------------------------------------------------------------------ #
    # PAGE-XML helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _bbox(points_str):
        pts = points_str.strip().split()
        xs  = [int(p.split(",")[0]) for p in pts]
        ys  = [int(p.split(",")[1]) for p in pts]
        return {"left": min(xs), "right": max(xs),
                "top":  min(ys), "bottom": max(ys)}

    def _tag(self, name):
        return "{%s}%s" % (PAGE_NS, name)

    def _clean_line(self, text, src_name):
        """
        Clean a transcription line:
        - Always: normalise whitespace via format_text_label
        - R335 standalone only: strip chars not in R352 charset
        - Combined training: keep all characters intact
        """
        text = self.format_text_label(text)
        if self._strip_oov or src_name == "R335" and self.dataset_name == "R335":
            text = _strip_r335_oov(text)
        return text

    def _parse_page_xml(self, xml_path, src_name=""):
        root = ET.parse(xml_path).getroot()
        page = root.find(self._tag("Page"))
        if page is None:
            raise ValueError("No <Page> in {}".format(xml_path))

        width  = int(page.attrib["imageWidth"])
        height = int(page.attrib["imageHeight"])
        image_filename = page.attrib["imageFilename"]

        regions = []
        for region_elem in page.findall(self._tag("TextRegion")):
            custom = region_elem.attrib.get("custom", "")
            rtype  = _region_custom_type(custom)

            if rtype in TOP_MARGIN_TYPES:
                sem_cat = "top_margin"
            elif rtype in SIDE_MARGIN_TYPES:
                sem_cat = "side_margin"
            elif rtype in BOTTOM_MARGIN_TYPES:
                sem_cat = "bottom_margin"
            else:
                sem_cat = "body"

            region_coords = self._bbox(
                region_elem.find(self._tag("Coords")).attrib["points"])

            lines = []
            for tl in region_elem.findall(self._tag("TextLine")):
                te  = tl.find(self._tag("TextEquiv"))
                if te is None:
                    continue
                uni = te.find(self._tag("Unicode"))
                if uni is None or uni.text is None:
                    continue
                text = self._clean_line(uni.text, src_name)
                if not text:
                    continue

                line_coords = self._bbox(
                    tl.find(self._tag("Coords")).attrib["points"])
                bl = tl.find(self._tag("Baseline"))
                line_baseline = (self._bbox(bl.attrib["points"])
                                 if bl is not None else line_coords.copy())

                lines.append({
                    "text":            text,
                    "coords":          line_coords,
                    "baseline_coords": line_baseline,
                })

            if not lines:
                continue

            region_baseline = {
                "left":   min(ln["baseline_coords"]["left"]   for ln in lines),
                "right":  max(ln["baseline_coords"]["right"]  for ln in lines),
                "top":    min(ln["baseline_coords"]["top"]    for ln in lines),
                "bottom": max(ln["baseline_coords"]["bottom"] for ln in lines),
            }
            regions.append({
                "sem_category":    sem_cat,
                "coords":          region_coords,
                "baseline_coords": region_baseline,
                "lines":           lines,
            })

        return image_filename, regions, width, height

    # ------------------------------------------------------------------ #
    # Image resolution                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _find_image(img_fold, image_filename, xml_fname):
        """
        Find the image file for this page.
        Tries XML filename stem first (fixes R335 naming mismatch),
        then falls back to imageFilename attribute from <Page>.
        """
        xml_stem = os.path.splitext(xml_fname)[0]
        for stem in [xml_stem, os.path.splitext(image_filename)[0]]:
            for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
                candidate = os.path.join(img_fold, stem + ext)
                if os.path.isfile(candidate):
                    return candidate
        return None

    # ------------------------------------------------------------------ #
    # Text assembly with semantic tokens                                   #
    # ------------------------------------------------------------------ #

    def _assemble_page_text(self, regions):
        groups = {k: [] for k in ["top_margin", "body", "side_margin", "bottom_margin"]}
        for r in regions:
            groups[r["sem_category"]].append(r)
        for key in groups:
            groups[key].sort(key=lambda r: r["coords"]["top"])

        has_side = len(groups["side_margin"]) > 0
        nb_cols  = 2 if has_side else 1

        tok_map = {
            "top_margin":    self.tokens["margin"],
            "body":          self.tokens["body"],
            "side_margin":   self.tokens["margin"],
            "bottom_margin": self.tokens["margin"],
        }
        parts = []
        for group_key in ["top_margin", "body", "side_margin", "bottom_margin"]:
            for region in groups[group_key]:
                sorted_lines = sorted(region["lines"],
                                      key=lambda ln: ln["baseline_coords"]["top"])
                region_text = self.format_text_label(
                    "\n".join(ln["text"] for ln in sorted_lines))
                if not region_text:
                    continue
                tok = tok_map[group_key]
                parts.append(tok + region_text + tok.upper())

        return "".join(parts), nb_cols

    def _build_paragraphs(self, regions):
        groups = {k: [] for k in ["top_margin", "body", "side_margin", "bottom_margin"]}
        for r in regions:
            groups[r["sem_category"]].append(r)
        for key in groups:
            groups[key].sort(key=lambda r: r["coords"]["top"])

        paragraphs = []
        tok_map = {
            "top_margin":    self.tokens["margin"],
            "body":          self.tokens["body"],
            "side_margin":   self.tokens["margin"],
            "bottom_margin": self.tokens["margin"],
        }
        for group_key in ["top_margin", "body", "side_margin", "bottom_margin"]:
            for region in groups[group_key]:
                sorted_lines = sorted(region["lines"],
                                      key=lambda ln: ln["baseline_coords"]["top"])
                para_lines = [{
                    "text":   ln["text"],
                    "top":    ln["coords"]["top"],
                    "bottom": ln["coords"]["bottom"],
                    "left":   ln["coords"]["left"],
                    "right":  ln["coords"]["right"],
                } for ln in sorted_lines]
                if not para_lines:
                    continue
                tok = tok_map[group_key]
                region_text = self.format_text_label(
                    "\n".join(ln["text"] for ln in sorted_lines))
                paragraphs.append({
                    "label":  tok + region_text + tok.upper(),
                    "lines":  para_lines,
                    "top":    min(l["top"]    for l in para_lines),
                    "bottom": max(l["bottom"] for l in para_lines),
                    "left":   min(l["left"]   for l in para_lines),
                    "right":  max(l["right"]  for l in para_lines),
                })
        return paragraphs

    def add_tokens_in_charset(self):
        begin_tokens = list(self.tokens.values())
        end_tokens   = [t.upper() for t in begin_tokens]
        self.charset = self.charset.union(set(begin_tokens + end_tokens))

    # ------------------------------------------------------------------ #
    # Main format function                                                 #
    # ------------------------------------------------------------------ #

    def format_r352_page(self):
        # Collect all XML files across all source paths
        all_xml_entries = []
        for src in self._source_paths:
            xml_files = natural_sort([
                f for f in os.listdir(src["xml_fold"])
                if f.lower().endswith(".xml")
            ])
            for f in xml_files:
                all_xml_entries.append((
                    os.path.join(src["xml_fold"], f),
                    src["img_fold"],
                    src["name"],
                    f
                ))

        if not all_xml_entries:
            raise FileNotFoundError("No .xml files found in any source path")

        # Shuffle and split
        rng      = random.Random(self.seed)
        shuffled = all_xml_entries[:]
        rng.shuffle(shuffled)
        n        = len(shuffled)
        n_train  = int(n * self.train_ratio)
        n_valid  = int(n * self.valid_ratio)

        split_map = {}
        for entry in shuffled[:n_train]:
            split_map[entry[0]] = "train"
        for entry in shuffled[n_train:n_train + n_valid]:
            split_map[entry[0]] = "valid"
        for entry in shuffled[n_train + n_valid:]:
            split_map[entry[0]] = "test"

        print("Total pages: {} | Split: {} train / {} valid / {} test".format(
            n, n_train, n_valid, n - n_train - n_valid))

        counters = {"train": 0, "valid": 0, "test": 0}

        for xml_path, img_fold, src_name, xml_fname in all_xml_entries:
            set_name = split_map[xml_path]

            try:
                image_filename, regions, page_width, page_height = \
                    self._parse_page_xml(xml_path, src_name)
            except Exception as e:
                print("  [WARN] Skipping {} — {}".format(xml_fname, e))
                continue

            if not regions:
                print("  [WARN] No lines in {}, skipping.".format(xml_fname))
                continue

            src_img = self._find_image(img_fold, image_filename, xml_fname)
            if src_img is None:
                print("  [WARN] Image not found for {}, skipping.".format(xml_fname))
                continue

            i            = counters[set_name]
            new_img_name = "{}_{}.jpeg".format(set_name, i)
            new_img_path = os.path.join(
                self.target_fold_path, set_name, new_img_name)
            self.load_resize_save(src_img, new_img_path, 300, 150)
            counters[set_name] += 1

            page_text, nb_cols = self._assemble_page_text(regions)
            self.charset = self.charset.union(set(page_text))

            paragraphs = self._build_paragraphs(regions)
            if not paragraphs:
                continue

            page_label = {
                "text":       page_text,
                "paragraphs": paragraphs,
                "nb_cols":    nb_cols,
                "top":        min(p["top"]    for p in paragraphs),
                "bottom":     max(p["bottom"] for p in paragraphs),
                "left":       min(p["left"]   for p in paragraphs),
                "right":      max(p["right"]  for p in paragraphs),
                "page_width": page_width,
            }

            self.gt[set_name][new_img_name] = {
                "text":    page_text,
                "nb_cols": nb_cols,
                "pages":   [page_label],
            }

        self.add_tokens_in_charset()

        print("\n=== {} formatting complete (sem) ===".format(self.dataset_name))
        for sn in ("train", "valid", "test"):
            print("  {:5s}: {:4d} pages".format(sn, len(self.gt[sn])))
        print("  charset ({} chars): {}".format(
            len(self.charset), "".join(sorted(self.charset))))
        print("  tokens: body={}/{} margin={}/{}".format(
            self.tokens["body"], self.tokens["body"].upper(),
            self.tokens["margin"], self.tokens["margin"].upper()))
        if self._strip_oov:
            print("  OOV stripped (R335 standalone mode): {}".format(
                sorted(repr(c) for c in R335_OOV_CHARS)))
        else:
            print("  OOV stripping: OFF (full charset preserved)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Format R352 / R335 / combined for FasterDAN")
    parser.add_argument("--dataset", default="R352",
                        choices=["R352", "R335", "R352_R335"],
                        help="Which dataset to format")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio",  type=float, default=0.1)
    args = parser.parse_args()

    R352DatasetFormatter(
        "page",
        dataset_name=args.dataset,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
    ).format()
