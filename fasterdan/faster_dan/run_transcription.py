"""
run_transcription.py — dev/test runner (NOT part of the shipped library).

Transcribes a folder of raw page images and writes one <page_id>.txt per page,
so the results can be downloaded. The service does not use this; it imports
TranscriptionEngine directly. This is only for local/server verification.

Place this next to predict_combined_model.py and run it the same way, so the
internal ``faster_dan.*`` imports resolve identically:

    cd src/notarypedia_transcription_framework
    python run_transcription.py <model_dir> <raw_images_folder> <output_dir>
    # e.g.
    python run_transcription.py outputs/fdan_R352_R335_page_sem /data/R335_raw out_transcriptions

On the server the device defaults to CUDA (faster). Override with --device cpu
to mirror a Mac run.
"""

import argparse

from inference import TranscriptionEngine   # sibling import (same context as predict_combined_model.py)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", help="Training output folder, e.g. outputs/fdan_R352_R335_page_sem")
    ap.add_argument("images", help="Folder of raw page images")
    ap.add_argument("output_dir", help="Where to write <page_id>.txt files")
    ap.add_argument("--device", default=None, help="cuda | cpu | mps (default: auto)")
    args = ap.parse_args()

    engine = TranscriptionEngine.from_pretrained(args.model_dir, device=args.device)
    results = engine.transcribe(args.images, output_dir=args.output_dir)

    print("Transcribed {} page(s) -> {}".format(len(results), args.output_dir))
    for page_id in results:
        print("  {}".format(page_id))


if __name__ == "__main__":
    main()
