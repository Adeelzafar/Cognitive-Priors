"""
MedMentions Data Adapter (ST21pv subset)
==========================================
Maps MedMentions (PubTator format, char-offset UMLS entity annotations)
to the exact same tensor format as BioScope / JNLPBA.
No changes to model.py, train.py, or run.py needed.

Unlike JNLPBA, MedMentions ships RAW TEXT + CHARACTER OFFSETS, not
pre-tokenized token+tag columns. This adapter tokenizes the text itself
(simple regex word/punctuation tokenizer with offset tracking) and
converts UMLS mention spans into token-level BIO tags.

Default label scheme is boundary-only (O / B-Entity / I-Entity, 3 classes):
MedMentions' primary annotation signal is UMLS entity linking, and most
NER benchmarks built on it evaluate boundary detection rather than
21-way semantic-type classification. Set `type_aware=True` in
`load_medmentions()` to instead produce full 21-type B-/I- tags
(43 classes) if you want a harder, type-aware variant.

Every detected mention also carries its gold UMLS CUI, so this dataset
additionally lets you validate the architecture's dictionary-based
context module (which does its own independent UMLS lookup) against
gold UMLS links, rather than only against another flat NER benchmark.

Usage:
    python medmentions_adapter.py --quick --device cuda
    python medmentions_adapter.py --device cuda
"""

import os
import sys
import re
import gzip
import urllib.request
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from collections import defaultdict

# Ensure this script's own directory is importable regardless of how/where
# it's launched from (job schedulers, wrapper scripts, folder names with
# spaces, etc. can all cause the auto-added sys.path[0] to not be reliable).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from data import find_entities_in_tokens

SECTION_TO_IDX = {"abstract": 13}  # PubMed abstracts -> same slot as BioScope abstracts

# 21 ST21pv semantic types (see MedMentions ReadMe)
ST21PV_TYPES = [
    "T005", "T007", "T017", "T022", "T031", "T033", "T037", "T038",
    "T058", "T062", "T074", "T082", "T091", "T092", "T097", "T098",
    "T103", "T168", "T170", "T201", "T204",
]

# Simple boundary-only scheme (default)
BOUNDARY_LABELS = ["O", "B-Entity", "I-Entity"]
NUM_BOUNDARY_CLASSES = len(BOUNDARY_LABELS)

# Type-aware scheme (optional, harder variant)
TYPE_AWARE_LABELS = ["O"] + [f"{p}-{t}" for t in ST21PV_TYPES for p in ("B", "I")]
NUM_TYPE_AWARE_CLASSES = len(TYPE_AWARE_LABELS)

_WORD_RE = re.compile(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]")

RAW_URL = (
    "https://raw.githubusercontent.com/chanzuckerberg/MedMentions/master/"
    "st21pv/data/corpus_pubtator.txt.gz"
)
SPLIT_BASE = (
    "https://raw.githubusercontent.com/chanzuckerberg/MedMentions/master/"
    "full/data/"
)


@dataclass
class MedMentionsInstance:
    tokens: List[str]
    ner_labels: List[int]
    section_id: int = 13

    @property
    def scope_labels(self):
        return self.ner_labels


# ---------------------------------------------------------------------------
# Tokenization with character-offset tracking
# ---------------------------------------------------------------------------

def _tokenize_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """Regex word/punctuation tokenizer. Returns (token, start_char, end_char)."""
    return [(m.group(), m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def _spans_to_bio(
    tok_spans: List[Tuple[str, int, int]],
    mentions: List[Tuple[int, int, str]],  # (start, end, semantic_type)
    type_aware: bool,
) -> List[int]:
    """Convert char-offset mentions into per-token BIO label ids."""
    # Resolve overlaps: greedy, longest-first, non-overlapping mentions
    mentions_sorted = sorted(mentions, key=lambda m: (m[0], -(m[1] - m[0])))
    kept = []
    last_end = -1
    for start, end, sem_type in mentions_sorted:
        if start >= last_end:
            kept.append((start, end, sem_type))
            last_end = end
    kept.sort(key=lambda m: m[0])

    labels = [0] * len(tok_spans)  # 0 == "O" in both schemes
    m_idx = 0
    for i, (tok, t_start, t_end) in enumerate(tok_spans):
        while m_idx < len(kept) and kept[m_idx][1] <= t_start:
            m_idx += 1
        if m_idx >= len(kept):
            continue
        m_start, m_end, sem_type = kept[m_idx]
        if t_start >= m_start and t_end <= m_end:
            is_first = (t_start == m_start) or (
                i > 0 and not (tok_spans[i - 1][1] <= m_start <= t_start)
            )
            # simpler, robust check: first token whose start >= m_start
            is_first = (i == 0) or (tok_spans[i - 1][2] <= m_start)
            if type_aware:
                base = 1 + 2 * ST21PV_TYPES.index(sem_type) if sem_type in ST21PV_TYPES else 0
                labels[i] = base if base == 0 else (base if is_first else base + 1)
            else:
                labels[i] = 1 if is_first else 2
    return labels


# ---------------------------------------------------------------------------
# PubTator parsing
# ---------------------------------------------------------------------------

def _parse_pubtator(path: str) -> Dict[str, Dict]:
    """Parse a (decompressed) PubTator file into {pmid: {"text": ..., "mentions": [...]}}."""
    docs = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if "|t|" in line:
                pmid, _, title = line.split("|", 2)
                docs.setdefault(pmid, {"title": "", "abstract": "", "mentions": []})
                docs[pmid]["title"] = title
            elif "|a|" in line:
                pmid, _, abstract = line.split("|", 2)
                docs.setdefault(pmid, {"title": "", "abstract": "", "mentions": []})
                docs[pmid]["abstract"] = abstract
            else:
                parts = line.split("\t")
                if len(parts) < 6:
                    continue
                pmid, start, end, mention_text, sem_type, cui = parts[:6]
                docs.setdefault(pmid, {"title": "", "abstract": "", "mentions": []})
                docs[pmid]["mentions"].append(
                    (int(start), int(end), sem_type, cui.replace("UMLS:", ""))
                )
    return docs


def load_medmentions(data_dir: str = "/tmp/medmentions", type_aware: bool = False):
    os.makedirs(data_dir, exist_ok=True)

    gz_path = os.path.join(data_dir, "corpus_pubtator.txt.gz")
    txt_path = os.path.join(data_dir, "corpus_pubtator.txt")
    if not os.path.exists(txt_path):
        if not os.path.exists(gz_path):
            print("Downloading MedMentions (ST21pv)...")
            urllib.request.urlretrieve(RAW_URL, gz_path)
        print("  Decompressing...")
        with gzip.open(gz_path, "rt", encoding="utf-8") as fin, \
             open(txt_path, "w", encoding="utf-8") as fout:
            fout.write(fin.read())

    split_files = {
        "train": "corpus_pubtator_pmids_trng.txt",
        "dev": "corpus_pubtator_pmids_dev.txt",
        "test": "corpus_pubtator_pmids_test.txt",
    }
    splits = {}
    for name, fname in split_files.items():
        local = os.path.join(data_dir, fname)
        if not os.path.exists(local):
            urllib.request.urlretrieve(SPLIT_BASE + fname, local)
        with open(local) as f:
            splits[name] = {line.strip() for line in f if line.strip()}

    print("  Parsing PubTator annotations...")
    docs = _parse_pubtator(txt_path)

    out = {"train": [], "dev": [], "test": []}
    for pmid, doc in docs.items():
        text = doc["title"] + " " + doc["abstract"]
        # abstract text is offset by len(title) + 1 space in the original PubTator
        # convention used to author the char offsets, so this concatenation matches them.
        tok_spans = _tokenize_with_offsets(text)
        mentions = [(s, e, t) for (s, e, t, _cui) in doc["mentions"]]
        labels = _spans_to_bio(tok_spans, mentions, type_aware)
        tokens = [t for (t, _s, _e) in tok_spans]

        inst = MedMentionsInstance(tokens=tokens, ner_labels=labels, section_id=13)
        for split_name, pmid_set in splits.items():
            if pmid in pmid_set:
                out[split_name].append(inst)
                break

    for name, insts in out.items():
        n_ent = sum(1 for inst in insts for l in inst.ner_labels if l > 0)
        print(f"  {name.capitalize()}: {len(insts)} docs, {n_ent} entity tokens")

    return out["train"], out["dev"], out["test"]


# ---------------------------------------------------------------------------
# PyTorch Dataset (identical alignment logic to data.py / ner_adapter.py)
# ---------------------------------------------------------------------------

class MedMentionsDataset(Dataset):
    def __init__(self, instances, tokenizer_name="dmis-lab/biobert-v1.1",
                 max_length=256, max_entities=32):
        self.instances = instances
        self.max_length = max_length
        self.max_entities = max_entities
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        inst = self.instances[idx]
        encoding = self.tokenizer(
            inst.tokens, is_split_into_words=True,
            max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        word_ids = encoding.word_ids()

        aligned_ner = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                aligned_ner.append(-100)
            elif word_id != prev_word_id:
                aligned_ner.append(
                    inst.ner_labels[word_id] if word_id < len(inst.ner_labels) else -100
                )
            else:
                # inside a subword continuation: keep "I-" of the same entity,
                # or -100 if this is an "O" token
                if word_id < len(inst.ner_labels) and inst.ner_labels[word_id] != 0:
                    aligned_ner.append(inst.ner_labels[word_id])
                else:
                    aligned_ner.append(-100)
            prev_word_id = word_id

        scope_labels = torch.tensor(aligned_ner, dtype=torch.long)
        cue_labels = torch.full_like(scope_labels, -100)
        section_id = torch.tensor(inst.section_id, dtype=torch.long)

        subword_tokens = self.tokenizer.convert_ids_to_tokens(input_ids.tolist())
        entity_mask, entity_positions, entity_distances = find_entities_in_tokens(
            subword_tokens, self.max_entities
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "cue_labels": cue_labels,
            "scope_labels": scope_labels,
            "section_ids": section_id,
            "entity_mask": torch.tensor(entity_mask, dtype=torch.long),
            "entity_positions": torch.tensor(entity_positions, dtype=torch.long),
            "entity_distances": torch.tensor(entity_distances, dtype=torch.float),
        }


def create_medmentions_stratified_subsets(dataset, fractions=None, seed=42):
    if fractions is None:
        fractions = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    rng = np.random.RandomState(seed)
    label_groups = defaultdict(list)
    for i, inst in enumerate(dataset.instances):
        n_ent = sum(1 for l in inst.ner_labels if l > 0)
        bucket = "dense" if n_ent >= 5 else ("sparse" if n_ent > 0 else "none")
        label_groups[bucket].append(i)
    subsets = {}
    for frac in fractions:
        if frac >= 1.0:
            subsets[frac] = Subset(dataset, list(range(len(dataset))))
            continue
        selected = []
        for indices in label_groups.values():
            n = max(1, int(len(indices) * frac))
            chosen = rng.choice(indices, size=n, replace=False)
            selected.extend(chosen.tolist())
        rng.shuffle(selected)
        subsets[frac] = Subset(dataset, selected)
    return subsets


# ---------------------------------------------------------------------------
# Experiment runner (mirrors ner_adapter.py's run_ner_experiments)
# ---------------------------------------------------------------------------

def run_medmentions_experiments(encoder_name="dmis-lab/biobert-v1.1",
                                 batch_size=16, device="cuda", quick=False,
                                 type_aware=False):
    import json, time, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from train import (
        TrainConfig, Trainer,
        run_learning_curve_experiment,
        run_baseline_learning_curve,
        run_ablation_experiment,
        run_calibration_analysis,
    )

    train_inst, val_inst, test_inst = load_medmentions(type_aware=type_aware)
    num_classes = NUM_TYPE_AWARE_CLASSES if type_aware else NUM_BOUNDARY_CLASSES

    print("\nBuilding MedMentions datasets...")
    train_ds = MedMentionsDataset(train_inst, encoder_name)
    val_ds = MedMentionsDataset(val_inst, encoder_name)
    test_ds = MedMentionsDataset(test_inst, encoder_name)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    seeds = [42] if quick else [42, 123, 456, 789, 1024]
    fractions = [0.2, 1.0] if quick else [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    config = TrainConfig(
        encoder_name=encoder_name,
        num_classes=num_classes,
        num_cue_labels=3,
        batch_size=batch_size,
        max_epochs=50 if not quick else 10,
        patience=7 if not quick else 3,
        seeds=seeds,
        data_fractions=fractions,
        output_dir="results_medmentions",
    )

    os.makedirs("results_medmentions", exist_ok=True)
    trainer = Trainer(config, device=device)
    results = {"config": {
        "dataset": "MedMentions-ST21pv",
        "type_aware": type_aware,
        "num_classes": num_classes,
        "encoder": encoder_name,
        "seeds": seeds,
        "fractions": fractions,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
    }}

    print("\n" + "=" * 60)
    print("MedMentions: Cognitive Priors Learning Curves")
    print("=" * 60)
    t0 = time.time()
    lc = run_learning_curve_experiment(trainer, train_ds, val_loader, test_loader, config)
    results["learning_curves"] = lc
    print(f"Done in {(time.time()-t0)/60:.1f} min")

    print("\n" + "=" * 60)
    print("MedMentions: Baseline Learning Curves")
    print("=" * 60)
    t0 = time.time()
    bl = run_baseline_learning_curve(trainer, train_ds, val_loader, test_loader, config)
    results["learning_curves_baseline"] = bl
    print(f"Done in {(time.time()-t0)/60:.1f} min")

    print("\n  CP vs Baseline:")
    print(f"  {'Frac':>8} {'CP':>10} {'BL':>10} {'Delta':>8}")
    for frac in sorted(config.data_fractions):
        if frac in lc["aggregated"] and frac in bl["aggregated"]:
            cp = lc["aggregated"][frac]["scope_f1_mean"]
            base = bl["aggregated"][frac]["scope_f1_mean"]
            print(f"  {frac:>8.0%} {cp:>10.4f} {base:>10.4f} {cp-base:>+8.4f}")

    print("\n" + "=" * 60)
    print("MedMentions: Ablation Study (at 20% data)")
    print("=" * 60)
    t0 = time.time()
    train_loader_full = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                    num_workers=2, pin_memory=True)
    abl = run_ablation_experiment(
        trainer, train_loader_full, val_loader, test_loader, config,
        data_fraction=0.2, dataset=train_ds,
    )
    results["ablation"] = abl
    print(f"Done in {(time.time()-t0)/60:.1f} min")
    print("\n  Ablation Results (20% data):")
    print(f"  {'Config':>15} {'Scope F1':>12} {'±std':>8}")
    for model_name, vals in abl.items():
        print(f"  {model_name:>15} {vals['scope_f1_mean']:>12.4f} {vals['scope_f1_std']:>8.4f}")

    print("\n" + "=" * 60)
    print("MedMentions: Calibration")
    print("=" * 60)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=2, pin_memory=True)
    best_run = trainer.train_single_run(train_loader, val_loader, test_loader, seed=42)
    model = trainer.build_model()
    model.load_state_dict(best_run["model_state"])
    cal = run_calibration_analysis(model, test_loader, trainer.device)
    results["calibration"] = cal
    print(f"  Pearson r  = {cal['pearson_r']:.4f}")
    print(f"  Spearman r = {cal['spearman_r']:.4f}")

    def convert(obj):
        if hasattr(obj, "item"): return obj.item()
        if isinstance(obj, (np.floating, np.float64, np.float32)): return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open("results_medmentions/all_results.json", "w") as f:
        json.dump(json.loads(json.dumps(results, default=convert)), f, indent=2)

    print("\nSaved to results_medmentions/all_results.json")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--type_aware", action="store_true",
                         help="Use 21-type B-/I- scheme instead of boundary-only")
    args = parser.parse_args()
    run_medmentions_experiments(batch_size=args.batch_size, device=args.device,
                                 quick=args.quick, type_aware=args.type_aware)