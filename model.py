"""
Epistemic Architecture v3 — Cognitive Priors
=============================================
FIXES:
  1. All modules operate in PARALLEL from raw encoder output
  2. Additive residual fusion (no softmax winner-take-all)
  3. Configurable encoder freezing for fair baseline comparison
  4. Each module adds its signal as a learned residual
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from typing import Optional, Dict, Tuple
import math


# ---------------------------------------------------------------------------
# Module 1 — Intent Prior (FiLM conditioning)
# ---------------------------------------------------------------------------

class IntentPriorModule(nn.Module):
    SECTION_TYPES = [
        "findings", "impression", "history", "indication", "plan",
        "technique", "comparison", "recommendation", "conclusion",
        "introduction", "methods", "results", "discussion",
        "abstract", "unknown"
    ]
    
    def __init__(self, hidden_dim: int = 768, intent_dim: int = 64):
        super().__init__()
        self.num_sections = len(self.SECTION_TYPES)
        self.section_to_idx = {s: i for i, s in enumerate(self.SECTION_TYPES)}
        
        self.section_embedding = nn.Embedding(self.num_sections, intent_dim)
        self.film_scale = nn.Linear(intent_dim, hidden_dim)
        self.film_shift = nn.Linear(intent_dim, hidden_dim)
        
        self.section_classifier = nn.Sequential(
            nn.Linear(hidden_dim, intent_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(intent_dim, self.num_sections)
        )
        
        # Learned residual weight — how much intent modulation to apply
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
        self._init_film_near_identity()
    
    def _init_film_near_identity(self):
        nn.init.ones_(self.film_scale.weight[:, :1])
        nn.init.zeros_(self.film_scale.weight[:, 1:])
        nn.init.zeros_(self.film_scale.bias)
        nn.init.zeros_(self.film_shift.weight)
        nn.init.zeros_(self.film_shift.bias)
    
    def forward(self, token_embeddings, section_ids=None, cls_embedding=None):
        section_logits = None
        
        if section_ids is not None:
            intent = self.section_embedding(section_ids)
        else:
            assert cls_embedding is not None
            section_logits = self.section_classifier(cls_embedding)
            probs = F.softmax(section_logits, dim=-1)
            intent = probs @ self.section_embedding.weight
        
        gamma = self.film_scale(intent).unsqueeze(1)
        beta = self.film_shift(intent).unsqueeze(1)
        
        modulated = gamma * token_embeddings + beta
        
        # Return the DELTA (what intent adds), not the full modulated output
        delta = modulated - token_embeddings
        
        return delta, section_logits


# ---------------------------------------------------------------------------
# Module 2 — Epistemic State (Evidential Deep Learning)
# ---------------------------------------------------------------------------

class EpistemicStateModule(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 768,
        num_classes: int = 4,
        conv_kernels: Tuple[int, ...] = (3, 5, 7),
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        
        self.word_evidence = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.phrase_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(num_classes, num_classes, k, padding=k // 2),
                nn.GELU()
            )
            for k in conv_kernels
        ])
        self.phrase_merge = nn.Linear(num_classes * len(conv_kernels), num_classes)
        
        self.sent_attention = nn.Linear(num_classes, 1)
        self.concentration_proj = nn.Linear(num_classes, num_classes)
        
        # Project evidence back to hidden_dim as a residual delta
        self.evidence_to_delta = nn.Sequential(
            nn.Linear(num_classes, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )
        
        # Initialize last layer near zero so it starts as a small residual
        nn.init.zeros_(self.evidence_to_delta[-1].weight)
        nn.init.zeros_(self.evidence_to_delta[-1].bias)
    
    def forward(self, token_embeddings, attention_mask):
        batch, seq_len, _ = token_embeddings.shape
        
        word_ev = self.word_evidence(token_embeddings)
        
        ev_transposed = word_ev.transpose(1, 2)
        phrase_outputs = []
        for conv in self.phrase_convs:
            phrase_outputs.append(conv(ev_transposed).transpose(1, 2))
        phrase_cat = torch.cat(phrase_outputs, dim=-1)
        phrase_ev = self.phrase_merge(phrase_cat)
        
        combined_ev = word_ev + phrase_ev
        
        attn_scores = self.sent_attention(combined_ev).squeeze(-1)
        attn_scores = attn_scores.masked_fill(~attention_mask.bool(), -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)
        sentence_ev = torch.einsum("bs,bsc->bc", attn_weights, combined_ev)
        
        token_alpha = F.softplus(self.concentration_proj(combined_ev)) + 1.0
        sentence_alpha = F.softplus(self.concentration_proj(sentence_ev)) + 1.0
        
        alpha_sum = token_alpha.sum(dim=-1)
        uncertainty = self.num_classes / alpha_sum
        
        # Residual delta from evidence
        delta = self.evidence_to_delta(combined_ev)
        
        return {
            "alpha": token_alpha,
            "sentence_alpha": sentence_alpha,
            "uncertainty": uncertainty,
            "delta": delta,
        }


# ---------------------------------------------------------------------------
# Module 3 — Context Prior (Entity Memory with Decay)
# ---------------------------------------------------------------------------

class ContextPriorModule(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 768,
        memory_dim: int = 128,
        max_entities: int = 32,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.max_entities = max_entities
        self.memory_dim = memory_dim
        self.num_heads = num_heads
        self.head_dim = memory_dim // num_heads
        
        self.entity_proj = nn.Linear(hidden_dim, memory_dim)
        self.log_beta = nn.Parameter(torch.tensor(0.5))
        
        self.query_proj = nn.Linear(hidden_dim, memory_dim)
        self.key_proj = nn.Linear(memory_dim, memory_dim)
        self.value_proj = nn.Linear(memory_dim, memory_dim)
        
        # Project back to hidden_dim as a delta (not full replacement)
        self.output_proj = nn.Sequential(
            nn.Linear(memory_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Initialize last layer near zero
        nn.init.zeros_(self.output_proj[0].weight)
        nn.init.zeros_(self.output_proj[0].bias)
    
    def forward(self, token_embeddings, entity_mask, entity_positions,
                entity_distances, entity_memory=None):
        batch, seq_len, hidden = token_embeddings.shape
        
        if entity_memory is None:
            entity_memory = self._build_memory(token_embeddings, entity_mask,
                                                entity_positions)
        
        beta = F.softplus(self.log_beta)
        decay = torch.exp(-beta * entity_distances)
        
        Q = self.query_proj(token_embeddings)
        K = self.key_proj(entity_memory)
        V = self.value_proj(entity_memory)
        
        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, self.max_entities, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, self.max_entities, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores * decay.unsqueeze(1)
        
        entity_valid = (entity_positions > -1).unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(~entity_valid, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        
        # Return delta only
        delta = self.output_proj(context)
        return delta
    
    def _build_memory(self, token_embeddings, entity_mask, entity_positions):
        batch, seq_len, hidden = token_embeddings.shape
        projected = self.entity_proj(token_embeddings)
        memory = torch.zeros(
            batch, self.max_entities, self.memory_dim,
            device=token_embeddings.device, dtype=token_embeddings.dtype
        )
        for b in range(batch):
            for e in range(self.max_entities):
                pos = entity_positions[b, e].item()
                if pos >= 0 and pos < seq_len:
                    memory[b, e] = projected[b, pos]
        return memory


# ---------------------------------------------------------------------------
# Additive Residual Fusion (replaces softmax gate)
# ---------------------------------------------------------------------------

class AdditiveResidualFusion(nn.Module):
    """
    Each module contributes a weighted residual delta to the base representation.
    No winner-take-all — all modules contribute simultaneously.
    
    output = base + α₁·delta_intent + α₂·delta_epistemic + α₃·delta_context
    
    Weights α are learned scalars (unconstrained, not softmax).
    LayerNorm stabilizes the combined output.
    """
    
    def __init__(self, hidden_dim: int = 768, num_modules: int = 3):
        super().__init__()
        # Learned scalar weights for each module (initialized small but nonzero)
        self.weights = nn.Parameter(torch.ones(num_modules) * 0.3)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, base: torch.Tensor, *deltas: torch.Tensor):
        """
        Args:
            base: (batch, seq, hidden) — raw encoder output
            deltas: tuple of (batch, seq, hidden) — each module's residual
        Returns:
            fused: (batch, seq, hidden)
            weights: (num_modules,) — for logging
        """
        combined = base.clone()
        for i, delta in enumerate(deltas):
            combined = combined + self.weights[i] * delta
        
        fused = self.layer_norm(combined)
        return fused, self.weights.detach()


# ---------------------------------------------------------------------------
# Full Architecture v3
# ---------------------------------------------------------------------------

class EpistemicNegationModel(nn.Module):
    """
    v3: Parallel modules with additive residual fusion.
    
    All three modules receive the SAME raw encoder output (parallel).
    Each produces a DELTA (what it adds to the representation).
    Fusion combines base + weighted deltas (no winner-take-all).
    """
    
    def __init__(
        self,
        encoder_name: str = "dmis-lab/biobert-v1.1",
        num_classes: int = 4,
        num_cue_labels: int = 3,
        freeze_encoder: bool = True,
        unfreeze_last_n: int = 2,
        hidden_dim: int = 768,
        dropout: float = 0.1,
        max_entities: int = 32
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # --- Shared Encoder ---
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.hidden_dim = self.encoder.config.hidden_size
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            # Unfreeze last N layers
            for param in self.encoder.encoder.layer[-unfreeze_last_n:].parameters():
                param.requires_grad = True
        
        # --- Three Parallel Modules (each produces a delta) ---
        self.intent_module = IntentPriorModule(self.hidden_dim)
        self.epistemic_module = EpistemicStateModule(self.hidden_dim, num_classes)
        self.context_module = ContextPriorModule(
            self.hidden_dim, max_entities=max_entities
        )
        
        # --- Additive Residual Fusion ---
        self.fusion = AdditiveResidualFusion(self.hidden_dim, num_modules=3)
        
        # --- Output Heads ---
        self.cue_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 2, num_cue_labels)
        )
        
        self.scope_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 2, num_classes)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        section_ids: Optional[torch.Tensor] = None,
        entity_mask: Optional[torch.Tensor] = None,
        entity_positions: Optional[torch.Tensor] = None,
        entity_distances: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        
        # --- Encode ---
        enc_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        base = enc_output.last_hidden_state  # (batch, seq, hidden)
        cls_emb = base[:, 0, :]
        
        # --- All modules operate on the SAME base (parallel) ---
        
        # Module 1: Intent delta
        intent_delta, section_logits = self.intent_module(
            base, section_ids, cls_emb
        )
        
        # Module 2: Epistemic delta + Dirichlet outputs
        epistemic_out = self.epistemic_module(base, attention_mask)
        epistemic_delta = epistemic_out["delta"]
        
        # Module 3: Context delta
        if entity_mask is not None and entity_positions is not None:
            context_delta = self.context_module(
                base, entity_mask, entity_positions, entity_distances
            )
        else:
            context_delta = torch.zeros_like(base)
        
        # --- Additive fusion: base + weighted deltas ---
        fused, fusion_weights = self.fusion(
            base, intent_delta, epistemic_delta, context_delta
        )
        
        # --- Output Heads ---
        cue_logits = self.cue_head(fused)
        scope_logits = self.scope_head(fused)
        
        # Expand fusion_weights for compatibility with gate analysis code
        batch, seq = base.shape[:2]
        gate_values = fusion_weights.unsqueeze(0).unsqueeze(0).expand(batch, seq, -1)
        
        return {
            "cue_logits": cue_logits,
            "scope_logits": scope_logits,
            "alpha": epistemic_out["alpha"],
            "sentence_alpha": epistemic_out["sentence_alpha"],
            "uncertainty": epistemic_out["uncertainty"],
            "gate_values": gate_values,
            "section_logits": section_logits,
            "fusion_weights": fusion_weights,
        }


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

class EpistemicLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        lambda_edl: float = 0.5,
        lambda_kl: float = 0.05,
        lambda_section: float = 0.3,
        kl_annealing_steps: int = 1000,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_edl = lambda_edl
        self.lambda_kl = lambda_kl
        self.lambda_section = lambda_section
        self.kl_annealing_steps = kl_annealing_steps
        
        self.cue_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.scope_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.section_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self._step = 0
    
    def forward(self, outputs, cue_labels, scope_labels,
                section_labels=None, attention_mask=None):
        self._step += 1
        
        cue_logits = outputs["cue_logits"]
        if (cue_labels != -100).any():
            L_cue = self.cue_loss_fn(
                cue_logits.view(-1, cue_logits.size(-1)), cue_labels.view(-1)
            )
        else:
            # No valid cue targets in this batch (e.g. NER-style datasets
            # like JNLPBA/MedMentions, which only use the scope/NER head).
            # CrossEntropyLoss(ignore_index=-100) would otherwise compute
            # 0/0 = NaN here and poison the total loss.
            L_cue = torch.tensor(0.0, device=cue_logits.device)
        
        scope_logits = outputs["scope_logits"]
        L_scope = self.scope_loss_fn(
            scope_logits.view(-1, scope_logits.size(-1)), scope_labels.view(-1)
        )
        
        alpha = outputs["alpha"]
        L_edl = self._edl_loss(alpha, scope_labels, attention_mask)
        
        kl_weight = min(1.0, self._step / self.kl_annealing_steps)
        L_kl = self._kl_divergence(alpha, attention_mask)
        
        L_section = torch.tensor(0.0, device=L_cue.device)
        if outputs["section_logits"] is not None and section_labels is not None:
            L_section = self.section_loss_fn(
                outputs["section_logits"], section_labels
            )
        
        total = (
            L_cue + L_scope
            + self.lambda_edl * L_edl
            + self.lambda_kl * kl_weight * L_kl
            + self.lambda_section * L_section
        )
        
        return {
            "total": total,
            "cue_loss": L_cue,
            "scope_loss": L_scope,
            "edl_loss": L_edl,
            "kl_loss": L_kl,
            "section_loss": L_section,
            "kl_weight": torch.tensor(kl_weight),
        }
    
    def _edl_loss(self, alpha, labels, mask):
        batch, seq, C = alpha.shape
        valid = (labels != -100)
        safe_labels = labels.clamp(min=0)
        y = F.one_hot(safe_labels, C).float()
        S = alpha.sum(dim=-1, keepdim=True)
        loss = (y * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=-1)
        if mask is not None:
            loss = loss * mask.float() * valid.float()
            return loss.sum() / (mask.float() * valid.float()).sum().clamp(min=1)
        loss = loss * valid.float()
        return loss.sum() / valid.float().sum().clamp(min=1)
    
    def _kl_divergence(self, alpha, mask):
        ones = torch.ones_like(alpha)
        S_alpha = alpha.sum(dim=-1, keepdim=True)
        S_ones = ones.sum(dim=-1, keepdim=True)
        kl = (
            torch.lgamma(S_alpha) - torch.lgamma(S_ones)
            - (torch.lgamma(alpha) - torch.lgamma(ones)).sum(dim=-1, keepdim=True)
            + ((alpha - ones) * (torch.digamma(alpha) - torch.digamma(S_alpha))).sum(dim=-1, keepdim=True)
        )
        kl = kl.squeeze(-1)
        if mask is not None:
            kl = kl * mask.float()
            return kl.sum() / mask.float().sum().clamp(min=1)
        return kl.mean()