"""
Training Pipeline & Experiments
================================

1. Main training loop with early stopping
2. Learning curve experiment (5%-100% data fractions)
3. Ablation study (remove each module)
4. Calibration analysis (uncertainty vs prediction difficulty)
5. Gate analysis (which module dominates when)
"""

import os
import json
import time
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support
from transformers import AutoModel

from model import EpistemicNegationModel, EpistemicLoss
from data import (
    BioScopeDataset, create_stratified_subsets
)
try:
    from bioscope_parser import BioScopeInstance, parse_bioscope_file as parse_bioscope_xml
except ImportError:
    from data import BioScopeInstance
    parse_bioscope_xml = None


# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    encoder_name: str = "dmis-lab/biobert-v1.1"
    freeze_encoder: bool = True
    num_classes: int = 4
    num_cue_labels: int = 3
    max_entities: int = 32

    learning_rate: float = 2e-4
    encoder_lr: float = 2e-5
    weight_decay: float = 0.01
    batch_size: int = 16
    max_epochs: int = 50
    patience: int = 7
    max_length: int = 256

    lambda_edl: float = 0.5
    lambda_kl: float = 0.05
    lambda_section: float = 0.3
    kl_annealing_steps: int = 500

    seeds: List[int] = None
    data_fractions: List[float] = None
    output_dir: str = "./results"

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 123, 456, 789, 1024]
        if self.data_fractions is None:
            self.data_fractions = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, config: TrainConfig, device: str = "cuda"):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def build_model(self, ablation: Optional[str] = None) -> EpistemicNegationModel:
        model = EpistemicNegationModel(
            encoder_name=self.config.encoder_name,
            num_classes=self.config.num_classes,
            num_cue_labels=self.config.num_cue_labels,
            freeze_encoder=self.config.freeze_encoder,
            max_entities=self.config.max_entities,
        ).to(self.device)
        if ablation:
            model = self._apply_ablation(model, ablation)
        return model

    def _apply_ablation(self, model, ablation):
        if ablation == "no_intent":
            model.intent_module = IdentityIntentModule(model.hidden_dim)
        elif ablation == "no_epistemic":
            model.epistemic_module = DummyEpistemicModule(
                model.hidden_dim, self.config.num_classes
            )
        elif ablation == "no_context":
            model.context_module = IdentityContextModule(model.hidden_dim)
        return model.to(self.device)

    def build_optimizer(self, model):
        encoder_params = []
        module_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "encoder" in name:
                encoder_params.append(param)
            else:
                module_params.append(param)
        return optim.AdamW([
            {"params": module_params, "lr": self.config.learning_rate},
            {"params": encoder_params, "lr": self.config.encoder_lr},
        ], weight_decay=self.config.weight_decay)

    def train_epoch(self, model, loader, optimizer, criterion):
        model.train()
        epoch_losses = defaultdict(float)
        n_batches = 0
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                section_ids=batch["section_ids"],
                entity_mask=batch["entity_mask"],
                entity_positions=batch["entity_positions"],
                entity_distances=batch["entity_distances"],
            )
            losses = criterion(
                outputs,
                cue_labels=batch["cue_labels"],
                scope_labels=batch["scope_labels"],
                section_labels=batch["section_ids"],
                attention_mask=batch["attention_mask"],
            )
            if not torch.isfinite(losses["total"]):
                print(f"  [warn] skipping non-finite batch (loss={losses['total'].item()})")
                n_batches += 1
                continue
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for k, v in losses.items():
                if torch.is_tensor(v) and not torch.isnan(v):
                    epoch_losses[k] += v.item()
            n_batches += 1
        return {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}

    @torch.no_grad()
    def evaluate(self, model, loader, criterion):
        model.eval()
        all_cue_preds, all_cue_labels = [], []
        all_scope_preds, all_scope_labels = [], []
        all_uncertainties = []
        all_gate_values = []
        total_loss = 0
        n_batches = 0

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                section_ids=batch["section_ids"],
                entity_mask=batch["entity_mask"],
                entity_positions=batch["entity_positions"],
                entity_distances=batch["entity_distances"],
            )
            losses = criterion(
                outputs,
                cue_labels=batch["cue_labels"],
                scope_labels=batch["scope_labels"],
                section_labels=batch["section_ids"],
                attention_mask=batch["attention_mask"],
            )
            loss_val = losses["total"].item()
            if not np.isnan(loss_val):
                total_loss += loss_val
            n_batches += 1

            mask = batch["attention_mask"].bool()
            cue_preds = outputs["cue_logits"].argmax(dim=-1)
            cue_labels = batch["cue_labels"]
            scope_preds = outputs["scope_logits"].argmax(dim=-1)
            scope_labels = batch["scope_labels"]

            for b in range(mask.size(0)):
                valid_cue = (cue_labels[b] != -100) & mask[b]
                valid_scope = (scope_labels[b] != -100) & mask[b]

                if valid_cue.any():
                    all_cue_preds.extend(cue_preds[b][valid_cue].cpu().tolist())
                    all_cue_labels.extend(cue_labels[b][valid_cue].cpu().tolist())

                if valid_scope.any():
                    all_scope_preds.extend(scope_preds[b][valid_scope].cpu().tolist())
                    all_scope_labels.extend(scope_labels[b][valid_scope].cpu().tolist())
                    all_uncertainties.extend(
                        outputs["uncertainty"][b][valid_scope].cpu().tolist()
                    )

            all_gate_values.append(outputs["gate_values"].cpu().numpy())

        cue_f1 = f1_score(all_cue_labels, all_cue_preds, average="macro",
                          zero_division=0) if all_cue_labels else float('nan')
        scope_f1 = f1_score(all_scope_labels, all_scope_preds, average="macro",
                            zero_division=0) if all_scope_labels else 0.0

        scope_f1_per_class = []
        if all_scope_labels:
            _, _, f1_pc, _ = precision_recall_fscore_support(
                all_scope_labels, all_scope_preds, average=None, zero_division=0
            )
            scope_f1_per_class = f1_pc.tolist()

        gate_values = np.concatenate(all_gate_values, axis=0)
        mean_gates = gate_values.mean(axis=(0, 1))

        return {
            "loss": total_loss / max(n_batches, 1),
            "cue_f1": cue_f1,
            "scope_f1": scope_f1,
            "scope_f1_per_class": scope_f1_per_class,
            "mean_uncertainty": np.mean(all_uncertainties) if all_uncertainties else 0,
            "gate_intent": float(mean_gates[0]),
            "gate_epistemic": float(mean_gates[1]),
            "gate_context": float(mean_gates[2]),
        }

    def train_single_run(self, train_loader, val_loader, test_loader,
                         seed=42, ablation=None):
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = self.build_model(ablation)
        optimizer = self.build_optimizer(model)
        criterion = EpistemicLoss(
            num_classes=self.config.num_classes,
            lambda_edl=self.config.lambda_edl,
            lambda_kl=self.config.lambda_kl,
            lambda_section=self.config.lambda_section,
            kl_annealing_steps=self.config.kl_annealing_steps,
        )

        best_val_f1 = 0
        best_epoch = 0
        patience_counter = 0
        best_state = None

        print(f"\n{'='*60}")
        print(f"Training | seed={seed} | ablation={ablation}")
        print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
        print(f"{'='*60}")

        for epoch in range(self.config.max_epochs):
            t0 = time.time()
            train_losses = self.train_epoch(model, train_loader, optimizer, criterion)
            val_metrics = self.evaluate(model, val_loader, criterion)
            elapsed = time.time() - t0

            print(
                f"Epoch {epoch:3d} | "
                f"train_loss={train_losses.get('total', float('nan')):.4f} | "
                f"val_scope_f1={val_metrics['scope_f1']:.4f} | "
                f"val_cue_f1={val_metrics['cue_f1']:.4f} | "
                f"uncertainty={val_metrics['mean_uncertainty']:.4f} | "
                f"{elapsed:.1f}s"
            )

            if val_metrics["scope_f1"] > best_val_f1:
                best_val_f1 = val_metrics["scope_f1"]
                best_epoch = epoch
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if best_state:
            model.load_state_dict(best_state)

        test_metrics = self.evaluate(model, test_loader, criterion)
        print(f"\nBest val epoch: {best_epoch} | val_scope_f1={best_val_f1:.4f}")
        print(f"Test scope_f1={test_metrics['scope_f1']:.4f} | "
              f"Test cue_f1={test_metrics['cue_f1']:.4f}")

        return {
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
            "test_metrics": test_metrics,
            "model_state": best_state,
        }

    # ------ Baseline ------

    def build_baseline_model(self):
        model = BioBERTBaseline(
            encoder_name=self.config.encoder_name,
            num_classes=self.config.num_classes,
            num_cue_labels=self.config.num_cue_labels,
            freeze_encoder=self.config.freeze_encoder,
        ).to(self.device)
        return model

    def train_baseline_run(self, train_loader, val_loader, test_loader, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = self.build_baseline_model()
        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.config.encoder_lr,
            weight_decay=self.config.weight_decay
        )

        cue_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        scope_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        eval_criterion = EpistemicLoss(
            num_classes=self.config.num_classes,
            lambda_edl=0.0, lambda_kl=0.0, lambda_section=0.0,
        )

        best_val_f1 = 0
        best_epoch = 0
        patience_counter = 0
        best_state = None

        print(f"\n{'='*60}")
        print(f"Baseline BioBERT | seed={seed}")
        print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
        print(f"{'='*60}")

        for epoch in range(self.config.max_epochs):
            t0 = time.time()
            model.train()
            total_loss = 0
            n_batches = 0

            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                cue_logits = outputs["cue_logits"]
                scope_logits = outputs["scope_logits"]

                # Handle NaN when all cue_labels are -100
                cue_labels = batch["cue_labels"]
                scope_labels = batch["scope_labels"]

                if (cue_labels != -100).any():
                    l_cue = cue_loss_fn(
                        cue_logits.view(-1, cue_logits.size(-1)),
                        cue_labels.view(-1)
                    )
                else:
                    l_cue = torch.tensor(0.0, device=self.device)

                if (scope_labels != -100).any():
                    l_scope = scope_loss_fn(
                        scope_logits.view(-1, scope_logits.size(-1)),
                        scope_labels.view(-1)
                    )
                else:
                    l_scope = torch.tensor(0.0, device=self.device)

                loss = l_cue + l_scope

                if not torch.isfinite(loss):
                    print(
                        f"  [warn] skipping non-finite batch "
                        f"(cue_valid={(cue_labels != -100).any().item()}, "
                        f"scope_valid={(scope_labels != -100).any().item()}, "
                        f"l_cue={l_cue.item():.4f}, l_scope={l_scope.item():.4f})"
                    )
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            val_metrics = self.evaluate(model, val_loader, eval_criterion)
            elapsed = time.time() - t0

            print(
                f"Epoch {epoch:3d} | "
                f"train_loss={avg_loss:.4f} | "
                f"val_scope_f1={val_metrics['scope_f1']:.4f} | "
                f"val_cue_f1={val_metrics['cue_f1']:.4f} | "
                f"{elapsed:.1f}s"
            )

            if val_metrics["scope_f1"] > best_val_f1:
                best_val_f1 = val_metrics["scope_f1"]
                best_epoch = epoch
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if best_state:
            model.load_state_dict(best_state)

        test_metrics = self.evaluate(model, test_loader, eval_criterion)
        print(f"\nBaseline best epoch: {best_epoch} | val_scope_f1={best_val_f1:.4f}")
        print(f"Test scope_f1={test_metrics['scope_f1']:.4f} | "
              f"Test cue_f1={test_metrics['cue_f1']:.4f}")

        return {
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
            "test_metrics": test_metrics,
        }


# ---------------------------------------------------------------------------
# Learning Curves
# ---------------------------------------------------------------------------

def run_learning_curve_experiment(trainer, dataset, val_loader, test_loader, config):
    results = {}
    for seed in config.seeds:
        subsets = create_stratified_subsets(dataset, config.data_fractions, seed)
        for frac, subset in subsets.items():
            key = f"frac={frac:.2f}_seed={seed}"
            print(f"\n{'#'*60}")
            print(f"Learning Curve: {key} | n_samples={len(subset)}")
            print(f"{'#'*60}")
            train_loader = DataLoader(
                subset, batch_size=config.batch_size,
                shuffle=True, num_workers=2, pin_memory=True
            )
            run_result = trainer.train_single_run(
                train_loader, val_loader, test_loader, seed
            )
            results[key] = {
                "fraction": frac, "seed": seed,
                "n_samples": len(subset),
                "test_scope_f1": run_result["test_metrics"]["scope_f1"],
                "test_cue_f1": run_result["test_metrics"]["cue_f1"],
                "best_val_f1": run_result["best_val_f1"],
                "best_epoch": run_result["best_epoch"],
            }

    aggregated = {}
    for frac in config.data_fractions:
        frac_results = [v for k, v in results.items() if v["fraction"] == frac]
        scope_f1s = [r["test_scope_f1"] for r in frac_results]
        cue_f1s = [r["test_cue_f1"] for r in frac_results]
        aggregated[frac] = {
            "scope_f1_mean": np.mean(scope_f1s),
            "scope_f1_std": np.std(scope_f1s),
            "cue_f1_mean": np.mean(cue_f1s),
            "cue_f1_std": np.std(cue_f1s),
            "n_runs": len(frac_results),
        }
    return {"per_run": results, "aggregated": aggregated}


def run_baseline_learning_curve(trainer, dataset, val_loader, test_loader, config):
    results = {}
    for seed in config.seeds:
        subsets = create_stratified_subsets(dataset, config.data_fractions, seed)
        for frac, subset in subsets.items():
            key = f"frac={frac:.2f}_seed={seed}"
            print(f"\n{'#'*60}")
            print(f"Baseline Learning Curve: {key} | n_samples={len(subset)}")
            print(f"{'#'*60}")
            train_loader = DataLoader(
                subset, batch_size=config.batch_size,
                shuffle=True, num_workers=2, pin_memory=True
            )
            run_result = trainer.train_baseline_run(
                train_loader, val_loader, test_loader, seed
            )
            results[key] = {
                "fraction": frac, "seed": seed,
                "n_samples": len(subset),
                "test_scope_f1": run_result["test_metrics"]["scope_f1"],
                "test_cue_f1": run_result["test_metrics"]["cue_f1"],
                "best_val_f1": run_result["best_val_f1"],
                "best_epoch": run_result["best_epoch"],
            }

    aggregated = {}
    for frac in config.data_fractions:
        frac_results = [v for k, v in results.items() if v["fraction"] == frac]
        scope_f1s = [r["test_scope_f1"] for r in frac_results]
        cue_f1s = [r["test_cue_f1"] for r in frac_results]
        aggregated[frac] = {
            "scope_f1_mean": np.mean(scope_f1s),
            "scope_f1_std": np.std(scope_f1s),
            "cue_f1_mean": np.mean(cue_f1s),
            "cue_f1_std": np.std(cue_f1s),
            "n_runs": len(frac_results),
        }
    return {"per_run": results, "aggregated": aggregated}


# ---------------------------------------------------------------------------
# Ablation Study
# ---------------------------------------------------------------------------

def run_ablation_experiment(trainer, train_loader, val_loader, test_loader,
                            config, data_fraction=0.2, dataset=None):
    ablations = [None, "no_intent", "no_epistemic", "no_context"]
    results = {}

    for ablation in ablations:
        label = ablation or "full_model"
        print(f"\n{'#'*60}")
        print(f"Ablation: {label}")
        print(f"{'#'*60}")

        if dataset and data_fraction < 1.0:
            subsets = create_stratified_subsets(dataset, [data_fraction], seed=42)
            abl_train_loader = DataLoader(
                subsets[data_fraction], batch_size=config.batch_size,
                shuffle=True, num_workers=2, pin_memory=True
            )
        else:
            abl_train_loader = train_loader

        seed_results = []
        for seed in config.seeds[:3]:
            run_result = trainer.train_single_run(
                abl_train_loader, val_loader, test_loader, seed, ablation
            )
            seed_results.append(run_result["test_metrics"])

        scope_f1s = [r["scope_f1"] for r in seed_results]
        results[label] = {
            "scope_f1_mean": np.mean(scope_f1s),
            "scope_f1_std": np.std(scope_f1s),
            "per_seed": seed_results,
        }
    return results


# ---------------------------------------------------------------------------
# Calibration Analysis (with ECE, Brier, NLL, AUROC)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_calibration_analysis(model, loader, device, n_bins=10):
    """
    Analyze whether model uncertainty correlates with actual difficulty.
    Reports: Pearson r, Spearman r, ECE, Brier score, NLL, AUROC for error detection.
    """
    model.eval()

    uncertainties = []
    correctnesses = []
    predictions = []
    labels = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            section_ids=batch["section_ids"],
            entity_mask=batch["entity_mask"],
            entity_positions=batch["entity_positions"],
            entity_distances=batch["entity_distances"],
        )

        scope_preds = outputs["scope_logits"].argmax(dim=-1)
        scope_labels = batch["scope_labels"]
        mask = batch["attention_mask"].bool()

        for b in range(mask.size(0)):
            valid = (scope_labels[b] != -100) & mask[b]
            pred = scope_preds[b][valid].cpu()
            label = scope_labels[b][valid].cpu()
            unc = outputs["uncertainty"][b][valid].cpu()
            correct = (pred == label).float()

            uncertainties.extend(unc.tolist())
            correctnesses.extend(correct.tolist())
            predictions.extend(pred.tolist())
            labels.extend(label.tolist())

    uncertainties = np.array(uncertainties)
    correctnesses = np.array(correctnesses)

    # Bin by uncertainty
    bin_edges = np.linspace(uncertainties.min(), uncertainties.max(), n_bins + 1)
    calibration_data = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (uncertainties >= lo) & (uncertainties < hi)
        if mask.sum() > 0:
            calibration_data.append({
                "bin_center": float((lo + hi) / 2),
                "mean_uncertainty": float(uncertainties[mask].mean()),
                "accuracy": float(correctnesses[mask].mean()),
                "n_samples": int(mask.sum()),
                "error_rate": float(1 - correctnesses[mask].mean()),
            })

    # Pearson and Spearman
    from scipy.stats import pearsonr, spearmanr
    errors = 1.0 - correctnesses
    pearson_r, pearson_p = pearsonr(uncertainties, errors)
    spearman_r, spearman_p = spearmanr(uncertainties, errors)

    # Additional calibration metrics
    from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss

    all_confidences = 1.0 - uncertainties

    # ECE (Expected Calibration Error)
    def compute_ece(conf, correct, bins=15):
        edges = np.linspace(0, 1, bins + 1)
        ece = 0.0
        for i in range(bins):
            m = (conf > edges[i]) & (conf <= edges[i + 1])
            if m.sum() > 0:
                ece += m.sum() * abs(correct[m].mean() - conf[m].mean())
        return ece / len(conf)

    ece = compute_ece(all_confidences, correctnesses)
    brier = brier_score_loss(correctnesses, all_confidences)
    nll = log_loss(correctnesses, np.clip(all_confidences, 1e-7, 1 - 1e-7))

    try:
        auroc = roc_auc_score(errors, uncertainties)
    except ValueError:
        auroc = 0.0

    print(f"  Pearson r  = {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"  Spearman r = {spearman_r:.4f} (p={spearman_p:.2e})")
    print(f"  ECE        = {ece:.4f}")
    print(f"  Brier      = {brier:.4f}")
    print(f"  NLL        = {nll:.4f}")
    print(f"  AUROC (error detection) = {auroc:.4f}")

    return {
        "calibration_bins": calibration_data,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "overall_accuracy": float(correctnesses.mean()),
        "mean_uncertainty": float(uncertainties.mean()),
        "ece": float(ece),
        "brier_score": float(brier),
        "nll": float(nll),
        "auroc_error_detection": float(auroc),
    }


# ---------------------------------------------------------------------------
# Gate Analysis
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_gate_analysis(model, loader, device):
    model.eval()
    gate_by_role = {
        "cue_tokens": [],
        "scope_tokens": [],
        "outside_tokens": [],
    }

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            section_ids=batch["section_ids"],
            entity_mask=batch["entity_mask"],
            entity_positions=batch["entity_positions"],
            entity_distances=batch["entity_distances"],
        )

        gates = outputs["gate_values"]
        cue_labels = batch["cue_labels"]
        scope_labels = batch["scope_labels"]
        mask = batch["attention_mask"].bool()

        for b in range(mask.size(0)):
            valid = mask[b]
            is_cue = (cue_labels[b] > 0) & valid
            is_scope = (scope_labels[b] > 0) & (cue_labels[b] == 0) & valid
            is_outside = (scope_labels[b] == 0) & (cue_labels[b] == 0) & valid

            if is_cue.any():
                gate_by_role["cue_tokens"].append(gates[b][is_cue].cpu().numpy())
            if is_scope.any():
                gate_by_role["scope_tokens"].append(gates[b][is_scope].cpu().numpy())
            if is_outside.any():
                gate_by_role["outside_tokens"].append(gates[b][is_outside].cpu().numpy())

    results = {}
    for role, gate_list in gate_by_role.items():
        if gate_list:
            all_gates = np.concatenate(gate_list, axis=0)
            results[role] = {
                "mean_intent_gate": float(all_gates[:, 0].mean()),
                "mean_epistemic_gate": float(all_gates[:, 1].mean()),
                "mean_context_gate": float(all_gates[:, 2].mean()),
                "n_tokens": len(all_gates),
            }
    return results


# ---------------------------------------------------------------------------
# Ablation Module Replacements
# ---------------------------------------------------------------------------

class IdentityIntentModule(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

    def forward(self, token_embeddings, section_ids=None, cls_embedding=None):
        return torch.zeros_like(token_embeddings), None


class DummyEpistemicModule(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, token_embeddings, attention_mask):
        batch, seq, _ = token_embeddings.shape
        uniform_alpha = torch.ones(
            batch, seq, self.num_classes,
            device=token_embeddings.device
        ) * 2.0
        return {
            "alpha": uniform_alpha,
            "sentence_alpha": uniform_alpha[:, 0, :],
            "uncertainty": torch.ones(batch, seq, device=token_embeddings.device) * 0.5,
            "delta": torch.zeros_like(token_embeddings),
        }


class IdentityContextModule(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

    def forward(self, token_embeddings, entity_mask=None,
                entity_positions=None, entity_distances=None,
                entity_memory=None):
        return torch.zeros_like(token_embeddings)


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

class BioBERTBaseline(nn.Module):
    def __init__(self, encoder_name="dmis-lab/biobert-v1.1", num_classes=4,
                 num_cue_labels=3, dropout=0.1, freeze_encoder=True,
                 unfreeze_last_n=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.encoder.encoder.layer[-unfreeze_last_n:].parameters():
                param.requires_grad = True

        self.cue_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, num_cue_labels)
        )
        self.scope_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = enc.last_hidden_state
        nc = self.scope_head[-1].out_features
        return {
            "cue_logits": self.cue_head(h),
            "scope_logits": self.scope_head(h),
            "alpha": torch.ones(*h.shape[:2], nc, device=h.device) * 2.0,
            "sentence_alpha": torch.ones(h.shape[0], nc, device=h.device) * 2.0,
            "uncertainty": torch.ones(*h.shape[:2], device=h.device) * 0.5,
            "gate_values": torch.ones(*h.shape[:2], 3, device=h.device) / 3,
            "section_logits": None,
        }


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def run_all_experiments(train_path, val_path=None, test_path=None,
                        doc_type="clinical", output_dir="./results"):
    config = TrainConfig(output_dir=output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("Parsing BioScope data...")
    train_instances = parse_bioscope_xml(train_path, doc_type)

    if val_path:
        val_instances = parse_bioscope_xml(val_path, doc_type)
    else:
        n = len(train_instances)
        np.random.RandomState(42).shuffle(train_instances)
        val_instances = train_instances[int(0.8*n):int(0.9*n)]
        test_instances = train_instances[int(0.9*n):]
        train_instances = train_instances[:int(0.8*n)]

    if test_path:
        test_instances = parse_bioscope_xml(test_path, doc_type)

    print(f"Train: {len(train_instances)} | Val: {len(val_instances)} | "
          f"Test: {len(test_instances)}")

    train_dataset = BioScopeDataset(train_instances, config.encoder_name)
    val_dataset = BioScopeDataset(val_instances, config.encoder_name)
    test_dataset = BioScopeDataset(test_instances, config.encoder_name)

    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    trainer = Trainer(config)
    all_results = {}

    print("\n" + "="*60)
    print("EXPERIMENT 1: Learning Curves")
    print("="*60)
    lc_results = run_learning_curve_experiment(
        trainer, train_dataset, val_loader, test_loader, config
    )
    all_results["learning_curves"] = lc_results

    print("\n" + "="*60)
    print("EXPERIMENT 2: Baseline Learning Curves")
    print("="*60)
    bl_results = run_baseline_learning_curve(
        trainer, train_dataset, val_loader, test_loader, config
    )
    all_results["learning_curves_baseline"] = bl_results

    print("\n" + "="*60)
    print("EXPERIMENT 3: Ablation Study")
    print("="*60)
    abl_results = run_ablation_experiment(
        trainer, train_loader, val_loader, test_loader, config,
        data_fraction=0.2, dataset=train_dataset,
    )
    all_results["ablation"] = abl_results

    print("\n" + "="*60)
    print("EXPERIMENT 4: Calibration Analysis")
    print("="*60)
    best_run = trainer.train_single_run(
        train_loader, val_loader, test_loader, seed=42
    )
    model = trainer.build_model()
    model.load_state_dict(best_run["model_state"])
    cal_results = run_calibration_analysis(model, test_loader, trainer.device)
    all_results["calibration"] = cal_results

    print("\n" + "="*60)
    print("EXPERIMENT 5: Gate Analysis")
    print("="*60)
    gate_results = run_gate_analysis(model, test_loader, trainer.device)
    all_results["gate_analysis"] = gate_results
    for role, vals in gate_results.items():
        print(f"  {role}: intent={vals['mean_intent_gate']:.3f} "
              f"epistemic={vals['mean_epistemic_gate']:.3f} "
              f"context={vals['mean_context_gate']:.3f} "
              f"(n={vals['n_tokens']})")

    results_path = os.path.join(output_dir, "all_results.json")
    def convert(obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, torch.Tensor): return obj.tolist()
        return obj

    serializable = json.loads(json.dumps(all_results, default=convert))
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nAll results saved to {results_path}")
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", default=None)
    parser.add_argument("--test", default=None)
    parser.add_argument("--doc_type", default="clinical",
                        choices=["clinical", "abstract", "full_paper"])
    parser.add_argument("--output_dir", default="./results")
    args = parser.parse_args()
    run_all_experiments(
        train_path=args.train, val_path=args.val,
        test_path=args.test, doc_type=args.doc_type,
        output_dir=args.output_dir,
    )