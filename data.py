"""
BioScope Data Pipeline (Real Data)
===================================

Uses bioscope_parser to load actual BioScope XML files.
Produces all inputs for the epistemic architecture:
  - Token labels: cue + scope (from BioScope gold)
  - Section IDs: from document structure (free)
  - Entity masks/positions/distances: from dictionary lookup (free)

No additional annotation required.
"""

import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader, Subset

from bioscope_parser import (
    BioScopeInstance, parse_bioscope_file, load_and_split
)


# ---------------------------------------------------------------------------
# Lightweight Entity Detection (UMLS dictionary substitute)
# ---------------------------------------------------------------------------
# In production, swap this for QuickUMLS or ScispaCy.
# This bootstrap covers common biomedical terms for development.

CLINICAL_ENTITIES = {
    "lung", "liver", "kidney", "heart", "brain", "bone", "lymph",
    "chest", "abdomen", "pelvis", "spine", "breast", "thyroid", "pancreas",
    "colon", "prostate", "ovary", "uterus", "bladder",
    "tumor", "tumour", "mass", "lesion", "nodule", "carcinoma", "malignancy",
    "metastasis", "metastatic", "fracture", "effusion", "embolism",
    "pneumonia", "atelectasis", "edema", "hemorrhage", "infarction",
    "stenosis", "thrombosis", "aneurysm", "calcification", "opacity",
    "fibrosis", "necrosis", "inflammation", "infection",
    "protein", "gene", "cell", "cells", "receptor", "kinase", "factor",
    "expression", "activation", "inhibition", "transcription", "binding",
    "mutation", "deletion", "promoter", "enhancer", "antibody",
    "cytokine", "apoptosis", "proliferation", "differentiation",
    "lymphocyte", "monocyte", "macrophage", "neutrophil",
    "nf-kappa", "nf-kb", "tnf", "il-2", "ifn", "hiv",
}


def find_entities_in_tokens(
    tokens: List[str],
    max_entities: int = 32
) -> Tuple[List[int], List[int], np.ndarray]:
    """
    Find biomedical entities via dictionary lookup (zero annotation cost).
    
    Returns:
        entity_mask: list of 0/1 per token
        entity_positions: list of up to max_entities token positions
        entity_distances: (seq_len, max_entities) sentence-distance array
    """
    seq_len = len(tokens)
    entity_mask = [0] * seq_len
    entity_positions_list = []
    
    # Track sentence boundaries
    sentence_ids = []
    current_sent = 0
    for t in tokens:
        sentence_ids.append(current_sent)
        if t in (".", "!", "?", "[SEP]"):
            current_sent += 1
    
    lower_tokens = [t.lower().rstrip(".,;:!?()[]") for t in tokens]
    
    for i, token in enumerate(lower_tokens):
        # Check against entity dictionary
        for entity in CLINICAL_ENTITIES:
            if len(entity) <= 2:
                if token == entity:
                    entity_mask[i] = 1
                    if len(entity_positions_list) < max_entities:
                        entity_positions_list.append(i)
                    break
            elif token == entity or (len(token) >= 4 and token.startswith(entity[:4])):
                entity_mask[i] = 1
                if len(entity_positions_list) < max_entities:
                    entity_positions_list.append(i)
                break
    
    # Pad to max_entities
    while len(entity_positions_list) < max_entities:
        entity_positions_list.append(-1)
    entity_positions_list = entity_positions_list[:max_entities]
    
    # Compute sentence distances
    entity_distances = np.zeros((seq_len, max_entities), dtype=np.float32)
    for e_idx, e_pos in enumerate(entity_positions_list):
        if e_pos >= 0:
            e_sent = sentence_ids[min(e_pos, len(sentence_ids)-1)]
            for t_idx in range(seq_len):
                t_sent = sentence_ids[t_idx]
                entity_distances[t_idx, e_idx] = abs(t_sent - e_sent)
        else:
            entity_distances[:, e_idx] = 999.0
    
    return entity_mask, entity_positions_list, entity_distances


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class BioScopeDataset(Dataset):
    """
    Full dataset producing all inputs for the epistemic architecture.
    
    All module inputs are derived from free signals:
        - section_ids: from document structure
        - entity_*: from dictionary lookup
        - cue/scope labels: from BioScope gold
    """
    
    def __init__(
        self,
        instances: List[BioScopeInstance],
        tokenizer_name: str = "dmis-lab/biobert-v1.1",
        max_length: int = 256,
        max_entities: int = 32,
    ):
        self.instances = instances
        self.max_length = max_length
        self.max_entities = max_entities
        
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        inst = self.instances[idx]
        
        # --- Tokenize with subword alignment ---
        encoding = self.tokenizer(
            inst.tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        word_ids = encoding.word_ids()
        
        # --- Align labels to subword tokens ---
        aligned_cue = []
        aligned_scope = []
        prev_word_id = None
        
        for word_id in word_ids:
            if word_id is None:
                aligned_cue.append(-100)
                aligned_scope.append(-100)
            elif word_id != prev_word_id:
                if word_id < len(inst.cue_labels):
                    aligned_cue.append(inst.cue_labels[word_id])
                    aligned_scope.append(inst.scope_labels[word_id])
                else:
                    aligned_cue.append(-100)
                    aligned_scope.append(-100)
            else:
                aligned_cue.append(-100)
                if word_id < len(inst.scope_labels):
                    aligned_scope.append(inst.scope_labels[word_id])
                else:
                    aligned_scope.append(-100)
            prev_word_id = word_id
        
        cue_labels = torch.tensor(aligned_cue, dtype=torch.long)
        scope_labels = torch.tensor(aligned_scope, dtype=torch.long)
        
        # --- Section ID (free from document structure) ---
        section_id = torch.tensor(inst.section_id, dtype=torch.long)
        
        # --- Entity detection (free from dictionary) ---
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


# ---------------------------------------------------------------------------
# Stratified Subsampling for Learning Curve Experiments
# ---------------------------------------------------------------------------

def create_stratified_subsets(
    dataset: BioScopeDataset,
    fractions: List[float] = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
    seed: int = 42
) -> Dict[float, Subset]:
    """Create stratified subsets preserving label distribution."""
    rng = np.random.RandomState(seed)
    
    label_groups = defaultdict(list)
    for i, inst in enumerate(dataset.instances):
        if 1 in inst.scope_labels:
            label_groups["negation"].append(i)
        elif 2 in inst.scope_labels:
            label_groups["speculation"].append(i)
        else:
            label_groups["assertion"].append(i)
    
    subsets = {}
    for frac in fractions:
        if frac >= 1.0:
            subsets[frac] = Subset(dataset, list(range(len(dataset))))
            continue
        
        selected = []
        for group_name, indices in label_groups.items():
            n_select = max(1, int(len(indices) * frac))
            chosen = rng.choice(indices, size=n_select, replace=False)
            selected.extend(chosen.tolist())
        
        rng.shuffle(selected)
        subsets[frac] = Subset(dataset, selected)
    
    return subsets
