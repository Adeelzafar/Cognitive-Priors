"""
NER Data Adapter for JNLPBA
=============================
Maps JNLPBA to the exact same tensor format as BioScope.
No changes to model.py, train.py, or run.py needed.

NER labels go into scope_labels, cue_labels set to -100 (ignored).
The scope head becomes the NER head. The cue head produces zero loss.

Usage:
    python ner_adapter.py --quick --device cuda
    python ner_adapter.py --device cuda
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from typing import List, Dict, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

from data import find_entities_in_tokens

SECTION_TO_IDX = {"abstract": 13}

NER_LABELS = [
    "O",
    "B-protein", "I-protein",
    "B-DNA", "I-DNA",
    "B-RNA", "I-RNA",
    "B-cell_line", "I-cell_line",
    "B-cell_type", "I-cell_type",
]
NUM_NER_CLASSES = len(NER_LABELS)

@dataclass
class NERInstance:
    tokens: List[str]
    ner_labels: List[int]
    section_id: int = 13
    
    @property
    def scope_labels(self):
        return self.ner_labels



def load_jnlpba():
    import urllib.request, os, random

    print("Downloading JNLPBA directly...")
    base = "https://raw.githubusercontent.com/cambridgeltl/MTL-Bioinformatics-2016/master/data/JNLPBA"
    tag_map = {"O": 0, "B-protein": 1, "I-protein": 2,
               "B-DNA": 3, "I-DNA": 4, "B-RNA": 5, "I-RNA": 6,
               "B-cell_line": 7, "I-cell_line": 8,
               "B-cell_type": 9, "I-cell_type": 10}

    def parse_file(filepath):
        instances = []
        tokens, labels = [], []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    if tokens:
                        ner = [tag_map.get(l, 0) for l in labels]
                        instances.append(NERInstance(tokens=tokens, ner_labels=ner, section_id=13))
                        tokens, labels = [], []
                else:
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        tokens.append(parts[0])
                        labels.append(parts[-1])
        if tokens:
            ner = [tag_map.get(l, 0) for l in labels]
            instances.append(NERInstance(tokens=tokens, ner_labels=ner, section_id=13))
        return instances

    os.makedirs("/tmp/jnlpba", exist_ok=True)
    splits = {}
    for name, fname in [("train", "train.tsv"), ("test", "test.tsv")]:
        local = f"/tmp/jnlpba/{fname}"
        if not os.path.exists(local):
            print(f"  Downloading {fname}...")
            urllib.request.urlretrieve(f"{base}/{fname}", local)
        splits[name] = parse_file(local)

    random.seed(42)
    all_train = splits["train"]
    random.shuffle(all_train)
    n_val = len(all_train) // 10
    val = all_train[:n_val]
    train = all_train[n_val:]
    test = splits["test"]

    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        n_ent = sum(1 for inst in split for l in inst.ner_labels if l > 0)
        print(f"  {name}: {len(split)} sentences, {n_ent} entity tokens")

    return train, val, test


class NERDataset(Dataset):
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
                if word_id < len(inst.ner_labels):
                    aligned_ner.append(inst.ner_labels[word_id])
                else:
                    aligned_ner.append(-100)
            else:
                if word_id < len(inst.ner_labels):
                    label = inst.ner_labels[word_id]
                    if label in [1, 3, 5, 7, 9]:
                        aligned_ner.append(label + 1)
                    else:
                        aligned_ner.append(label)
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


def create_ner_stratified_subsets(dataset, fractions=None, seed=42):
    if fractions is None:
        fractions = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    rng = np.random.RandomState(seed)
    label_groups = defaultdict(list)
    for i, inst in enumerate(dataset.instances):
        has_protein = any(l in [1, 2] for l in inst.ner_labels)
        has_dna = any(l in [3, 4] for l in inst.ner_labels)
        has_other = any(l in [5, 6, 7, 8, 9, 10] for l in inst.ner_labels)
        if has_protein:
            label_groups["protein"].append(i)
        elif has_dna:
            label_groups["dna"].append(i)
        elif has_other:
            label_groups["other"].append(i)
        else:
            label_groups["none"].append(i)
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


def run_ner_experiments(encoder_name="dmis-lab/biobert-v1.1",
                        batch_size=16, device="cuda", quick=False):
    import json, time, os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from train import (
        TrainConfig, Trainer,
        run_learning_curve_experiment,
        run_baseline_learning_curve,
        run_calibration_analysis,
    )

    train_inst, val_inst, test_inst = load_jnlpba()

    print("\nBuilding NER datasets...")
    train_ds = NERDataset(train_inst, encoder_name)
    val_ds = NERDataset(val_inst, encoder_name)
    test_ds = NERDataset(test_inst, encoder_name)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    sample = next(iter(DataLoader(train_ds, batch_size=2)))
    print("Sample batch shapes:")
    for k, v in sample.items():
        print(f"  {k}: {v.shape}")

    seeds = [42] if quick else [42, 123, 456, 789, 1024]
    fractions = [0.2, 1.0] if quick else [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    config = TrainConfig(
        encoder_name=encoder_name,
        num_classes=NUM_NER_CLASSES,
        num_cue_labels=3,
        batch_size=batch_size,
        max_epochs=50 if not quick else 10,
        patience=7 if not quick else 3,
        seeds=seeds,
        data_fractions=fractions,
        output_dir="results_ner",
    )

    os.makedirs("results_ner", exist_ok=True)
    trainer = Trainer(config, device=device)
    results = {"config": {
        "dataset": "JNLPBA",
        "num_classes": NUM_NER_CLASSES,
        "encoder": encoder_name,
        "seeds": seeds,
        "fractions": fractions,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
    }}

    # Cognitive Priors
    print("\n" + "="*60)
    print("NER: Cognitive Priors Learning Curves")
    print("="*60)
    t0 = time.time()
    lc = run_learning_curve_experiment(trainer, train_ds, val_loader, test_loader, config)
    results["learning_curves"] = lc
    print(f"Done in {(time.time()-t0)/60:.1f} min")

    print("\n  Cognitive Priors:")
    print(f"  {'Frac':>8} {'F1':>10} {'std':>8}")
    for frac, vals in sorted(lc["aggregated"].items()):
        print(f"  {float(frac):>8.0%} {vals['scope_f1_mean']:>10.4f} {vals['scope_f1_std']:>8.4f}")

    # Baseline
    print("\n" + "="*60)
    print("NER: Baseline Learning Curves")
    print("="*60)
    t0 = time.time()
    bl = run_baseline_learning_curve(trainer, train_ds, val_loader, test_loader, config)
    results["learning_curves_baseline"] = bl
    print(f"Done in {(time.time()-t0)/60:.1f} min")

    print("\n  Baseline:")
    print(f"  {'Frac':>8} {'F1':>10} {'std':>8}")
    for frac, vals in sorted(bl["aggregated"].items()):
        print(f"  {float(frac):>8.0%} {vals['scope_f1_mean']:>10.4f} {vals['scope_f1_std']:>8.4f}")

    # Comparison
    print("\n  CP vs Baseline:")
    print(f"  {'Frac':>8} {'CP':>10} {'BL':>10} {'Delta':>8}")
    for frac in sorted(config.data_fractions):
        if frac in lc["aggregated"] and frac in bl["aggregated"]:
            cp = lc["aggregated"][frac]["scope_f1_mean"]
            base = bl["aggregated"][frac]["scope_f1_mean"]
            print(f"  {frac:>8.0%} {cp:>10.4f} {base:>10.4f} {cp-base:>+8.4f}")

    # Calibration
    print("\n" + "="*60)
    print("NER: Calibration")
    print("="*60)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    best_run = trainer.train_single_run(train_loader, val_loader, test_loader, seed=42)
    model = trainer.build_model()
    model.load_state_dict(best_run["model_state"])
    cal = run_calibration_analysis(model, test_loader, trainer.device)
    results["calibration"] = cal
    print(f"  Pearson r  = {cal['pearson_r']:.4f}")
    print(f"  Spearman r = {cal['spearman_r']:.4f}")

    # Save
    def convert(obj):
        if hasattr(obj, 'item'): return obj.item()
        if isinstance(obj, (np.floating, np.float64, np.float32)): return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open("results_ner/all_results.json", "w") as f:
        json.dump(json.loads(json.dumps(results, default=convert)), f, indent=2)

    print(f"\nSaved to results_ner/all_results.json")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    run_ner_experiments(batch_size=args.batch_size, device=args.device, quick=args.quick)