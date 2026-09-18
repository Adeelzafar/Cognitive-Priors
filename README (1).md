# Epistemic Architecture for Negation/Speculation Detection

Structural priors replace data volume. Three lightweight modules (~500K params) encode communicative intent, epistemic uncertainty, and cognitive bias as architectural inductive biases on top of a frozen BioBERT encoder. The architecture achieves competitive performance with a fraction of the training data — no additional annotation required.

## Quick Start

```bash
pip install torch transformers numpy scikit-learn scipy

# Copy your BioScope files into data/
mkdir data
cp /path/to/abstracts.xml data/
cp /path/to/full_papers.xml data/

# Run full experiment suite
cd src
python run.py --abstracts ../data/abstracts.xml --papers ../data/full_papers.xml

# Quick smoke test (1 seed, 2 fractions, ~10 min on GPU)
python run.py --quick

# Results saved to results/all_results.json
```

## Data: Where Every Label Comes From

| Component | Label Source | Cost |
|---|---|---|
| Cue detection head | BioScope gold XML annotations | **Standard** |
| Scope resolution head | BioScope gold XML annotations | **Standard** |
| Intent module (section type) | Parsed from `<DocumentPart type="...">` tags | **Zero** |
| Intent module (FiLM params) | End-to-end via task loss | **Zero** |
| Epistemic module (Dirichlet α) | End-to-end via task loss | **Zero** |
| Context module (entities) | Dictionary lookup (UMLS/biomedical terms) | **Zero** |
| Context module (decay params) | End-to-end via task loss | **Zero** |
| Fusion gate values | End-to-end via task loss | **Zero** |

## Corpus Statistics (parsed from your files)

| | Abstracts | Full Papers | Total |
|---|---|---|---|
| Documents | 1,273 | 9 | 1,282 |
| Sentences | 11,871 | 2,530 | 14,401 |
| With negation | 1,473 (12.4%) | 287 (11.3%) | 1,760 |
| With speculation | 2,067 (17.4%) | 507 (20.0%) | 2,574 |
| Plain | 8,468 (71.3%) | 1,780 (70.4%) | 10,248 |

## Architecture

```
BioBERT Encoder (frozen, last 2 layers unfrozen)
        │
        ▼
┌───────────────┐  Section headers from XML ──→ embedding ──→ FiLM scale/shift
│ Intent Prior  │  ~100K params, zero annotation
│ (FiLM)        │
└───────┬───────┘
        ▼
┌───────────────┐  Dirichlet concentrations, not logits
│ Epistemic     │  Word → Phrase (multi-scale conv) → Sentence (attention)
│ State (EDL)   │  ~200K params, zero annotation
└───────┬───────┘
        │
┌───────────────┐  Entity memory + exponential decay attention
│ Context Prior │  UMLS dictionary lookup, learned decay rate
│ (Memory)      │  ~150K params, zero annotation
└───────┬───────┘
        ▼
┌───────────────┐  Per-token learned gates (interpretable)
│ Gated Fusion  │  ~50K params
└───────┬───────┘
        ▼
   Cue Head + Scope Head
```

## Experiments

1. **Learning Curves** — Train at {5%, 10%, 20%, 40%, 60%, 80%, 100%} × 5 seeds. Hypothesis: architecture matches baseline F1 at ~20% data.
2. **Ablation** — Remove each module at 20% data. Shows which prior contributes most.
3. **Calibration** — Correlate Dirichlet uncertainty with prediction errors. Key claim: uncertainty tracks difficulty without uncertainty labels.
4. **Gate Analysis** — Mean gate values for cue vs scope vs outside tokens. Shows interpretable module specialization.

## File Structure

```
src/
  run.py              ← Single-command entry point
  bioscope_parser.py  ← Parses real BioScope XML (handles nested scopes)
  data.py             ← Dataset + entity detection + stratified sampling
  model.py            ← Architecture: 3 modules + fusion + loss
  train.py            ← Training loop + all 4 experiments + baseline
  generate_data.py    ← Synthetic data (for development without BioScope)
```

## Extending to Other Domains

Swap the entity dictionary (UMLS → domain ontology), update section types in `SECTION_MAP`, and adjust the cue lexicon. Everything else — the Dirichlet parameterization, FiLM conditioning, decay attention, gated fusion — transfers directly to any expert-document domain (clinical notes, legal briefs, financial filings).
