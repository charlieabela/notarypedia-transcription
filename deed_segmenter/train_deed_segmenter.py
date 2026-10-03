#!/usr/bin/env python3
"""
Deed Segmentation Model — Training Pipeline
============================================
Trains a multilingual BERT classifier to label each line in a PAGE XML
transcription as:
  B-DEED  — first line of a new deed  (SOD tag)
  I-DEED  — continuation line inside a deed
  E-DEED  — last line of a deed       (EOD tag)
  O       — outside any deed (margin, foliation, etc.)

The model takes a sliding context window of ±3 lines around each target line,
concatenated with [SEP] separators, and classifies the centre line using the
[CLS] embedding → linear head.

Usage:
    python3 train_deed_segmenter.py \
        --page-dir   data/R352/page \
        --output-dir outputs \
        --epochs 10

After training, inference is run with:
    python3 train_deed_segmenter.py --infer \
        --model-dir outputs/best_model \
        --page-dir  <page-xml-dir> \
        --output    outputs/deeds.json
"""

import os, re, json, argparse, random
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import classification_report, f1_score

# ── Constants ────────────────────────────────────────────────────────────────
NS       = '{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}'
LABELS   = ['O', 'B-DEED', 'I-DEED', 'E-DEED']
L2I      = {l: i for i, l in enumerate(LABELS)}
I2L      = {i: l for i, l in enumerate(LABELS)}
SOD_RE   = re.compile(r'structure\s*\{[^}]*type\s*:\s*SOD\s*;', re.I)
EOD_RE   = re.compile(r'structure\s*\{[^}]*type\s*:\s*EOD\s*;', re.I)
CONTEXT  = 3          # lines of context on each side
MAX_TOK  = 256        # BERT max tokens per sample
MODEL_ID = 'bert-base-multilingual-cased'


# ── Data extraction ──────────────────────────────────────────────────────────

def parse_page(xml_path: str) -> list[dict]:
    """Return list of {text, label} for all lines in one PAGE XML."""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return []

    raw = []
    for region in root.findall(f'.//{NS}TextRegion'):
        for line in region.findall(f'{NS}TextLine'):
            custom = line.get('custom', '')
            text_el = line.find(f'{NS}TextEquiv/{NS}Unicode')
            text = (text_el.text or '').strip() if text_el is not None else ''
            if not text:
                continue
            if SOD_RE.search(custom):
                label = 'B-DEED'
            elif EOD_RE.search(custom):
                label = 'E-DEED'
            else:
                label = None          # resolved below
            raw.append({'text': text, 'custom': custom, 'label': label})

    # Fill I-DEED / O using state machine
    inside = False
    for item in raw:
        if item['label'] == 'B-DEED':
            inside = True
        elif item['label'] == 'E-DEED':
            inside = False
        elif item['label'] is None:
            item['label'] = 'I-DEED' if inside else 'O'

    return [{'text': r['text'], 'label': r['label']} for r in raw]


def load_all_pages(page_dir: str) -> list[dict]:
    """Load all PAGE XMLs; return flat list of line dicts with 'page' key."""
    page_dir = Path(page_dir)
    all_lines = []
    for xml_file in sorted(page_dir.glob('*.xml')):
        if xml_file.name == 'metadata.xml':
            continue
        lines = parse_page(str(xml_file))
        for l in lines:
            l['page'] = xml_file.stem
        all_lines.extend(lines)
    return all_lines


def build_contexts(all_lines: list[dict], ctx: int = CONTEXT) -> list[dict]:
    """
    For each line i, build a context window of [i-ctx … i … i+ctx].
    The centre line is the classification target.
    """
    samples = []
    n = len(all_lines)
    for i, item in enumerate(all_lines):
        window_texts = []
        for j in range(max(0, i - ctx), min(n, i + ctx + 1)):
            prefix = '>>> ' if j == i else ''
            window_texts.append(prefix + all_lines[j]['text'])
        samples.append({
            'input':  ' [SEP] '.join(window_texts),
            'label':  L2I[item['label']],
            'page':   item['page'],
            'text':   item['text'],
        })
    return samples


# ── Dataset ──────────────────────────────────────────────────────────────────

class LineDataset(Dataset):
    def __init__(self, samples, tokenizer):
        self.samples   = samples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        enc = self.tokenizer(
            s['input'],
            max_length=MAX_TOK,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label':          torch.tensor(s['label'], dtype=torch.long),
        }


# ── Training ─────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── load data ─────────────────────────────────────────────────────────
    print('Loading PAGE XML files...')
    all_lines = load_all_pages(args.page_dir)
    print(f'  Total lines: {len(all_lines)}')
    label_counts = {l: sum(1 for x in all_lines if x['label'] == l)
                    for l in LABELS}
    print(f'  Label distribution: {label_counts}')

    # ── split by page (not line) to avoid data leakage ────────────────────
    pages = list(dict.fromkeys(l['page'] for l in all_lines))
    random.seed(42)
    random.shuffle(pages)
    n_test  = max(1, int(len(pages) * 0.05))
    n_val   = max(1, int(len(pages) * 0.10))
    test_pages  = set(pages[:n_test])
    val_pages   = set(pages[n_test:n_test + n_val])
    train_pages = set(pages[n_test + n_val:])

    def split_lines(page_set):
        return [l for l in all_lines if l['page'] in page_set]

    train_lines = split_lines(train_pages)
    val_lines   = split_lines(val_pages)
    test_lines  = split_lines(test_pages)

    print(f'  Train pages: {len(train_pages)}  ({len(train_lines)} lines)')
    print(f'  Val   pages: {len(val_pages)}    ({len(val_lines)} lines)')
    print(f'  Test  pages: {len(test_pages)}   ({len(test_lines)} lines)')

    # ── build context windows ─────────────────────────────────────────────
    train_samples = build_contexts(train_lines)
    val_samples   = build_contexts(val_lines)
    test_samples  = build_contexts(test_lines)

    # ── tokeniser & model ─────────────────────────────────────────────────
    print(f'\nLoading {MODEL_ID}...')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(LABELS)
    ).to(device)

    train_loader = DataLoader(
        LineDataset(train_samples, tokenizer),
        batch_size=16, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        LineDataset(val_samples, tokenizer),
        batch_size=32, shuffle=False, num_workers=0
    )

    # ── optimiser with class weights ──────────────────────────────────────
    # Up-weight B-DEED and E-DEED (rare classes)
    total = len(train_lines)
    weights = torch.tensor([
        total / max(1, label_counts[l]) for l in LABELS
    ], dtype=torch.float).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_f1   = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        # ── train ──────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['label'].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ── validate ───────────────────────────────────────────────────
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels         = batch['label'].to(device)
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(labels.cpu().numpy())

        f1 = f1_score(all_true, all_preds, average='macro',
                      zero_division=0)
        deed_f1 = f1_score(
            all_true, all_preds,
            labels=[L2I['B-DEED'], L2I['I-DEED'], L2I['E-DEED']],
            average='macro', zero_division=0
        )

        print(f'Epoch {epoch:02d}/{args.epochs}  '
              f'loss={avg_loss:.4f}  '
              f'macro_f1={f1:.4f}  '
              f'deed_f1={deed_f1:.4f}')

        if deed_f1 > best_f1:
            best_f1    = deed_f1
            best_epoch = epoch
            model.save_pretrained(str(output_dir / 'best_model'))
            tokenizer.save_pretrained(str(output_dir / 'best_model'))
            print(f'  ✓ New best model saved (deed_f1={best_f1:.4f})')

    print(f'\nBest epoch: {best_epoch}  deed_f1={best_f1:.4f}')

    # ── test set evaluation ───────────────────────────────────────────────
    print('\nTest set evaluation:')
    best_model = AutoModelForSequenceClassification.from_pretrained(
        str(output_dir / 'best_model')
    ).to(device)
    best_model.eval()
    test_loader = DataLoader(
        LineDataset(test_samples, tokenizer),
        batch_size=32, shuffle=False, num_workers=0
    )
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = best_model(input_ids=input_ids,
                                 attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(batch['label'].numpy())

    print(classification_report(
        all_true, all_preds,
        labels=list(range(len(LABELS))),
        target_names=LABELS, zero_division=0
    ))

    # Save label map
    with open(output_dir / 'label_map.json', 'w') as f:
        json.dump({'labels': LABELS, 'l2i': L2I, 'i2l': {str(k): v
                   for k, v in I2L.items()}}, f, indent=2)
    print(f'\nModel saved to: {output_dir}/best_model')


# ── Inference ─────────────────────────────────────────────────────────────────

def infer(args):
    """
    Apply trained deed segmenter to a directory of PAGE XML files
    (e.g. Kraken output) and produce deed-segmented JSON.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_dir = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir)
    ).to(device).eval()

    all_lines = load_all_pages(args.page_dir)
    print(f'Loaded {len(all_lines)} lines from {args.page_dir}')

    samples = build_contexts(all_lines)
    loader  = DataLoader(
        LineDataset(samples, tokenizer),
        batch_size=32, shuffle=False, num_workers=0
    )

    preds = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device)
            )
            preds.extend(outputs.logits.argmax(dim=-1).cpu().numpy())

    # Reconstruct deeds from predicted labels.
    # O lines are never discarded:
    #   - O lines before the first deed go into a preamble entry (deed_id=0)
    #   - O lines inside an open deed are appended to that deed
    #   - O lines after a closed deed are appended to the last closed deed
    #     until the next B-DEED opens a new deed
    deeds = []
    current = None
    last_closed = None
    first_deed_seen = False
    preamble = {
        'deed_id':    0,
        'type':       'preamble',
        'start_page': None,
        'end_page':   None,
        'lines':      [],
    }

    for i, (line, pred) in enumerate(zip(all_lines, preds)):
        label = I2L[pred]
        if label == 'B-DEED':
            if current:
                deeds.append(current)
                last_closed = deeds[-1]
            current = {
                'deed_id':    len(deeds) + 1,
                'start_page': line['page'],
                'end_page':   line['page'],
                'lines':      [line['text']],
            }
            last_closed = None
            first_deed_seen = True
        elif label in ('I-DEED', 'E-DEED') and current:
            current['lines'].append(line['text'])
            current['end_page'] = line['page']
            if label == 'E-DEED':
                deeds.append(current)
                last_closed = deeds[-1]
                current = None
        elif label == 'O':
            if current:
                # O inside an open deed
                current['lines'].append(line['text'])
                current['end_page'] = line['page']
            elif last_closed is not None:
                # O after a closed deed — attach to last closed deed
                last_closed['lines'].append(line['text'])
                last_closed['end_page'] = line['page']
            else:
                # O before the first deed — preamble
                preamble['lines'].append(line['text'])
                preamble['end_page'] = line['page']
                if preamble['start_page'] is None:
                    preamble['start_page'] = line['page']

    if current:
        deeds.append(current)

    # Prepend preamble if it contains any lines
    if preamble['lines']:
        preamble['text'] = '\n'.join(preamble['lines'])
        preamble['line_count'] = len(preamble['lines'])
        deeds.insert(0, preamble)

    # Annotate with full text
    for d in deeds:
        if 'text' not in d:
            d['text'] = '\n'.join(d['lines'])
        d['line_count'] = len(d['lines'])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(deeds, f, ensure_ascii=False, indent=2)

    print(f'\nFound {len(deeds)} deeds → {out_path}')
    multi = sum(1 for d in deeds if d['start_page'] != d['end_page'])
    print(f'Multi-page deeds: {multi}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer', action='store_true',
                        help='Run inference instead of training')
    # Training args
    parser.add_argument('--page-dir',
                        default='data/R352/page')
    parser.add_argument('--output-dir',
                        default='outputs')
    parser.add_argument('--epochs', type=int, default=10)
    # Inference args
    parser.add_argument('--model-dir',
                        default='outputs/best_model')
    parser.add_argument('--output',
                        default='outputs/deeds.json')
    args = parser.parse_args()

    if args.infer:
        infer(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
