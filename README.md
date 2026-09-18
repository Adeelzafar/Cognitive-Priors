# Cognitive Priors Beyond Flat Labels: Evidential Uncertainty in Biomedical NLP

Code and data adapters for our paper testing whether information discarded by flat
biomedical NLP labels (document intent, entity context, and evidential uncertainty)
can be recovered architecturally, and whether recovering it actually helps.

**Short version of the finding:** giving a pretrained encoder explicit intent and
entity-context signals (via lightweight modules) produces no measurable benefit on
either task we test. Giving it an explicit evidential uncertainty signal, via a
hierarchical evidential deep learning (EDL) module whose output feeds back into the
shared representation, does: removing it is the only ablation that reliably costs
task performance on both datasets. We call this variant **HF-EDL** (Hierarchical
Feedback EDL) to distinguish it from the original EDL formulation
([Sensoy et al., 2018](https://arxiv.org/abs/1806.01768)), which we build on rather
than replace.

## What's in this repo

| File | Purpose |
|---|---|
| `bioscope_parser.py` | Parses raw BioScope XML (abstracts + full papers) into token-level cue/scope instances |
| `data.py` | BioScope PyTorch `Dataset`, UMLS dictionary entity lookup, stratified data-fraction sampling |
| `ner_adapter.py` | Adapter mapping JNLPBA into the same tensor interface as BioScope, no changes to `model.py`/`train.py` needed |
| `medmentions_adapter.py` | Adapter mapping MedMentions ST21pv (PubTator format) into the same interface; downloads and parses the corpus directly from GitHub |
| `model.py` | The architecture: intent module (control), HF-EDL evidential uncertainty module (central component), context module (control), additive fusion, task heads |
| `train.py` | Training loop, learning-curve experiments, ablation study, calibration analysis, gate analysis |
| `uncertainty_baselines.py` | Post-hoc uncertainty baselines: MC Dropout, Temperature Scaling, Deep Ensembles |
| `run.py` | Single-command entry point for the full BioScope experiment suite |
| `make_dataset_plots.py` | Generates the cross-dataset ablation/calibration/learning-curve figures from `all_results.json` files |

## Quick start

```bash
pip install torch transformers numpy scikit-learn scipy

# BioScope (requires the corpus files from https://www.inf.u-szeged.hu/rgai/bioscope)
python run.py --abstracts data/abstracts.xml --papers data/full_papers.xml

# MedMentions (downloads automatically, no manual data acquisition needed)
python medmentions_adapter.py --device cuda

# Quick smoke test on either (1 seed, 2 data fractions, ~10 min on GPU)
python run.py --quick
python medmentions_adapter.py --quick --device cuda
```

Results are saved to `results/all_results.json` (BioScope) and
`results_medmentions/all_results.json` (MedMentions). Generate the paper's
cross-dataset figures with:

```bash
python make_dataset_plots.py --bioscope results/all_results.json \
                              --medmentions results_medmentions/all_results.json \
                              --outdir figures/
```

## Architecture, briefly

```
BioBERT Encoder (frozen, last 2 layers fine-tuned)
        |
        v
+----------------+   Control: section type from document structure
| Intent Module  |   -> FiLM conditioning
+-------+--------+
        |
+----------------+   Central component: multi-scale token evidence ->
| HF-EDL         |   Dirichlet concentration parameters -> feeds back
| (Evidential    |   into the shared representation
| Uncertainty)   |
+-------+--------+
        |
+----------------+   Control: UMLS dictionary lookup, recency-weighted
| Context Module |   entity memory attention
+-------+--------+
        v
  Additive residual fusion (LayerNorm, skip connection)
        v
   Task heads (cue / scope / entity boundary)
```

None of the three modules requires new human annotation; every signal source is
already present in the data (document structure tags, a UMLS dictionary, and the
training objective itself).

## Datasets

- **BioScope** ([Vincze et al., 2008](https://www.inf.u-szeged.hu/rgai/bioscope)):
  negation and speculation cue/scope detection, 14,401 sentences. Not redistributed
  here; download from the original source and point `run.py` at the XML files.
- **MedMentions ST21pv** ([Mohan & Li, 2019](https://github.com/chanzuckerberg/MedMentions)):
  general biomedical entity-boundary detection, 4,392 PubMed abstracts. Downloaded
  and parsed automatically by `medmentions_adapter.py`, no manual steps needed.
- **JNLPBA**: adapter included (`ner_adapter.py`), downloaded automatically from a
  public mirror. Not yet included in the paper's reported results; contributions
  extending the cross-dataset comparison to a third dataset are welcome.

## Honest limitations (see the paper's Limitations section for the full list)

- Both reported datasets are English-language biomedical text; generalization
  outside this domain is untested.
- We use the term *evidential uncertainty*, not *epistemic uncertainty*: we do not
  test whether this signal behaves as formal epistemic uncertainty under data
  scarcity or distribution shift. All evaluation here is in-distribution.
- The ablation removing the HF-EDL module removes its loss term, KL regularizer,
  and representation-feedback path simultaneously. It does not isolate which of
  these three components drives the task-performance effect; a finer-grained
  ablation is left to future work.
- The post-hoc uncertainty comparison (`uncertainty_baselines.py`) has only been
  run on BioScope, not MedMentions.
- The BioScope fallback section classifier's own accuracy has not yet been
  separately evaluated.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{zafar2026cognitivepriors,
  title     = {Cognitive Priors Beyond Flat Labels: Evidential Uncertainty in Biomedical NLP},
  author    = {Zafar, Adeel},
  year      = {2026},
  note      = {Preprint}
}
```

## License

[Add license here, e.g. MIT]
