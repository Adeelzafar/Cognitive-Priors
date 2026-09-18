#!/usr/bin/env python3
"""
run.py — Single-command entry point
====================================

Usage:
    python run.py                          # uses default paths
    python run.py --abstracts /path/to/abstracts.xml --papers /path/to/full_papers.xml
    python run.py --quick                  # fast smoke test (1 seed, 2 fractions)

What it does:
    1. Parses real BioScope XML (abstracts + full papers)
    2. Creates train/val/test splits at document level
    3. Runs learning curve experiment (epistemic architecture)
    4. Runs BioBERT baseline for comparison
    5. Runs ablation study
    6. Runs calibration + gate analysis
    7. Saves all results to JSON
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Epistemic Architecture Experiments")
    parser.add_argument("--abstracts", default="data/abstracts.xml",
                        help="Path to BioScope abstracts.xml")
    parser.add_argument("--papers", default="data/full_papers.xml",
                        help="Path to BioScope full_papers.xml")
    parser.add_argument("--clinical", default=None,
                        help="Path to merged clinical XML (after ScopeMerger)")
    parser.add_argument("--output_dir", default="results",
                        help="Output directory for results")
    parser.add_argument("--encoder", default="dmis-lab/biobert-v1.1",
                        help="Pretrained encoder name")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test: 1 seed, 2 data fractions")
    parser.add_argument("--device", default="cuda",
                        help="Device: cuda or cpu")
    args = parser.parse_args()

    # Resolve data paths — check common locations
    abstracts_path = _find_file(args.abstracts, [
        "data/abstracts.xml",
        "../data/abstracts.xml",
        "/mnt/user-data/uploads/abstracts.xml",
    ])
    papers_path = _find_file(args.papers, [
        "data/full_papers.xml",
        "../data/full_papers.xml",
        "/mnt/user-data/uploads/full_papers.xml",
    ])
    clinical_path = None
    if args.clinical:
        clinical_path = _find_file(args.clinical, [
            "data/clinical_merged.xml",
            "../data/clinical_merged.xml",
        ])

    if not abstracts_path:
        print("ERROR: Cannot find abstracts.xml. Provide path with --abstracts")
        sys.exit(1)

    print(f"Abstracts: {abstracts_path}")
    print(f"Papers:    {papers_path or 'not found (using abstracts only)'}")
    print(f"Clinical:  {clinical_path or 'not provided (run ScopeMerger first)'}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Step 1: Parse BioScope ----
    print("\n" + "="*60)
    print("STEP 1: Parsing BioScope Corpus")
    print("="*60)

    from bioscope_parser import load_and_split, print_corpus_stats

    train_inst, val_inst, test_inst = load_and_split(
        abstracts_path, papers_path, clinical_path=clinical_path
    )

    print_corpus_stats(train_inst, "Train")
    print_corpus_stats(val_inst, "Validation")
    print_corpus_stats(test_inst, "Test")

    # ---- Step 2: Build Datasets ----
    print("\n" + "="*60)
    print("STEP 2: Building PyTorch Datasets")
    print("="*60)

    from data import BioScopeDataset, create_stratified_subsets
    from torch.utils.data import DataLoader

    train_ds = BioScopeDataset(train_inst, args.encoder, max_length=256)
    val_ds = BioScopeDataset(val_inst, args.encoder, max_length=256)
    test_ds = BioScopeDataset(test_inst, args.encoder, max_length=256)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Verify a batch
    sample_batch = next(iter(DataLoader(train_ds, batch_size=2)))
    print(f"Sample batch shapes:")
    for k, v in sample_batch.items():
        print(f"  {k}: {v.shape}")

    # ---- Step 3: Configure Experiments ----
    from train import TrainConfig, Trainer
    from train import (
        run_learning_curve_experiment,
        run_baseline_learning_curve,
        run_ablation_experiment,
        run_calibration_analysis,
        run_gate_analysis,
    )

    if args.quick:
        seeds = [42]
        fractions = [0.2, 1.0]
        max_epochs = 10
        patience = 3
    else:
        seeds = [42, 123, 456, 789, 1024]
        fractions = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        max_epochs = args.max_epochs
        patience = 7

    config = TrainConfig(
        encoder_name=args.encoder,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_epochs=max_epochs,
        patience=patience,
        seeds=seeds,
        data_fractions=fractions,
        output_dir=args.output_dir,
    )

    trainer = Trainer(config, device=args.device)
    all_results = {"config": {
        "encoder": args.encoder,
        "data_fractions": fractions,
        "seeds": seeds,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
    }}

    # ---- Step 4: Learning Curves (Epistemic Architecture) ----
    print("\n" + "="*60)
    print("STEP 4: Learning Curve Experiment")
    print("="*60)

    t0 = time.time()
    lc_results = run_learning_curve_experiment(
        trainer, train_ds, val_loader, test_loader, config
    )
    all_results["learning_curves"] = lc_results
    print(f"\nLearning curves completed in {(time.time()-t0)/60:.1f} minutes")

    # Print summary
    print("\n  Aggregated Results:")
    print(f"  {'Fraction':>10} {'Scope F1':>12} {'±std':>8} {'Cue F1':>12} {'±std':>8}")
    for frac, vals in sorted(lc_results["aggregated"].items()):
        print(f"  {frac:>10.0%} {vals['scope_f1_mean']:>12.4f} {vals['scope_f1_std']:>8.4f}"
              f" {vals['cue_f1_mean']:>12.4f} {vals['cue_f1_std']:>8.4f}")

    # ---- Step 4b: Baseline BioBERT Learning Curves ----
    print("\n" + "="*60)
    print("STEP 4b: Baseline BioBERT Learning Curves")
    print("="*60)

    t0 = time.time()
    baseline_results = run_baseline_learning_curve(
        trainer, train_ds, val_loader, test_loader, config
    )
    all_results["learning_curves_baseline"] = baseline_results
    print(f"\nBaseline curves completed in {(time.time()-t0)/60:.1f} minutes")

    print("\n  Baseline Results:")
    print(f"  {'Fraction':>10} {'Scope F1':>12} {'±std':>8} {'Cue F1':>12} {'±std':>8}")
    for frac, vals in sorted(baseline_results["aggregated"].items()):
        print(f"  {frac:>10.0%} {vals['scope_f1_mean']:>12.4f} {vals['scope_f1_std']:>8.4f}"
              f" {vals['cue_f1_mean']:>12.4f} {vals['cue_f1_std']:>8.4f}")

    # ---- Comparison ----
    print("\n  Epistemic vs Baseline Comparison:")
    print(f"  {'Fraction':>10} {'Epistemic':>12} {'Baseline':>12} {'Δ':>8}")
    for frac in sorted(config.data_fractions):
        frac_key = frac
        if frac_key in lc_results["aggregated"] and frac_key in baseline_results["aggregated"]:
            ep = lc_results["aggregated"][frac_key]["scope_f1_mean"]
            bl = baseline_results["aggregated"][frac_key]["scope_f1_mean"]
            print(f"  {frac:>10.0%} {ep:>12.4f} {bl:>12.4f} {ep-bl:>+8.4f}")

    # ---- Step 5: Ablation Study ----
    print("\n" + "="*60)
    print("STEP 5: Ablation Study (at 20% data)")
    print("="*60)

    t0 = time.time()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)

    abl_results = run_ablation_experiment(
        trainer, train_loader, val_loader, test_loader, config,
        data_fraction=0.2, dataset=train_ds,
    )
    all_results["ablation"] = abl_results
    print(f"\nAblation completed in {(time.time()-t0)/60:.1f} minutes")

    print("\n  Ablation Results (20% data):")
    print(f"  {'Model':>20} {'Scope F1':>12} {'±std':>8}")
    for model_name, vals in abl_results.items():
        print(f"  {model_name:>20} {vals['scope_f1_mean']:>12.4f} {vals['scope_f1_std']:>8.4f}")

    # ---- Step 6: Calibration & Gate Analysis ----
    print("\n" + "="*60)
    print("STEP 6: Calibration & Gate Analysis")
    print("="*60)

    # Train full model for analysis
    best_run = trainer.train_single_run(
        train_loader, val_loader, test_loader, seed=42
    )
    model = trainer.build_model()
    model.load_state_dict(best_run["model_state"])

    import torch
    cal_results = run_calibration_analysis(model, test_loader, trainer.device)
    all_results["calibration"] = cal_results

    print(f"\n  Uncertainty-Error Correlation:")
    print(f"    Pearson r  = {cal_results['pearson_r']:.4f} (p={cal_results['pearson_p']:.2e})")
    print(f"    Spearman r = {cal_results['spearman_r']:.4f} (p={cal_results['spearman_p']:.2e})")

    gate_results = run_gate_analysis(model, test_loader, trainer.device)
    all_results["gate_analysis"] = gate_results

    print(f"\n  Gate Analysis (which module dominates):")
    print(f"  {'Token Type':>20} {'Intent':>8} {'Epistemic':>10} {'Context':>8}")
    for role, vals in gate_results.items():
        print(f"  {role:>20} {vals['mean_intent_gate']:>8.3f}"
              f" {vals['mean_epistemic_gate']:>10.3f}"
              f" {vals['mean_context_gate']:>8.3f}")

    # ---- Save Results ----
    results_path = os.path.join(args.output_dir, "all_results.json")

    def convert(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, 'item'):
            return obj.item()
        return obj

    serializable = json.loads(json.dumps(all_results, default=convert))
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\n{'='*60}")
    print(f"ALL RESULTS SAVED TO: {results_path}")
    print(f"{'='*60}")


def _find_file(primary: str, fallbacks: list) -> str:
    """Find file at primary path or fallback locations."""
    if os.path.exists(primary):
        return primary
    for path in fallbacks:
        if os.path.exists(path):
            return path
    return None


if __name__ == "__main__":
    main()