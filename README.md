# PHSM source code

**PHSM: A Post–Hashtag Semantic Matching Framework for Hashtag Recommendation from Multimodal Posts**

This release contains a curated subset of the author's source files corresponding to five manuscript components: the Parallel Co-Attention (PCA) post encoder and Baseline/PHSM-T/PHSM-TP/PHSM-Full models, four-stage training, Instagram preprocessing, strict cold-label partitioning, and semantic C4Eval.

## Contents

All runtime Python modules are in `phsm_pca/` so their original flat imports work after running `cd phsm_pca`.

- `model.py`, `model_matching_v1.py`, `model_matching_v2.py`, `model_matching_v3.py`: Baseline, T, TP, Full and PCA post encoder.
- `train_pca_v2_softmax.py`, `train_helpers.py`, `make_shared_split.py`: training and independent validation selection. `train_helpers.py` consolidates unchanged helper functions from historical scripts; this avoids publishing duplicate outdated stage runners.
- `build_tag_meta.py`, `build_tag_text_inputs.py`, `extract_imgfeat_vgg16_original.py`, `extract_vgg16_torch.py`, `dataset.py`: metadata, VGG16 features and loaders.
- `split_coldstart.py`, `build_tag_prototypes.py`, `build_tag_prototypes_original.py`: strict cold splitting, user-history reconstruction, and leakage-aware prototype generation (choose the prototype builder for the relevant original/strict setting).
- `c4_semantic_core.py`, `eval_c4_semantic_embedding_pca.py`: frozen MiniLM embeddings, similarity threshold and one-to-one Hungarian matching.

## Prerequisites

Install PyTorch and torchvision suitable for your GPU/CUDA version, then run `pip install -r requirements.txt`. The original Instagram post metadata, tags, images and VGG16 feature files are **not redistributed**. Users must obtain any dataset under its original terms and build the documented input splits/label mappings independently. Pickle input files should be loaded only from trusted sources.

## Training entry point

```bash
cd phsm_pca
python train_pca_v2_softmax.py --help
```

The trainer exposes four values for `--stage`: `baseline`, `v1`, `v2`, `v3`. The Baseline uses weighted BCE. **PHSM-T, PHSM-TP and PHSM-Full must be run with `--v3-loss macon_softmax`** to reproduce the current multi-positive Softmax objective. The source defaults to BCE to retain historical CLI compatibility, so omitting this flag changes the training objective. The V3 multi-positive Softmax branch additionally uses a stepwise learning-rate schedule. Checkpoint initialization is sequential: Baseline → V1 → V2 → V3. A best checkpoint is selected by **independent validation MeanF1@7**, without using the held-out test split. Supply your real, non-public dataset paths via CLI arguments; no pre-trained checkpoint or dataset is included here.

**Important:** This repository has passed static code and import smoke checks; full training/metric reproduction has not been rerun from scratch as part of this source publication. This archive may not include every experiment-specific command-line setting or frozen checkpoint from the final manuscript. Do not claim numerical reproduction without the original split, hyperparameter settings, and appropriate checkpoints.

## Strict Cold-start

The source `split_coldstart.py` uses seed 42; eligibility requires at least 100 training and 5 test appearances, uniformly samples 5% of eligible hashtag IDs (up to 100 labels), removes *all* training posts containing a selected cold hashtag, rebuilds user histories using retained training posts, and performs selected leakage and train/test overlap checks. The loader in `dataset.py` rejects missing safe user histories. The protocol reflects the original code; inspect potential edge cases and data licenses before executing it with new datasets.

## Semantic C4Eval

Run `cd phsm_pca && python eval_c4_semantic_embedding_pca.py --help` for the CLI. Use `--sim-threshold 0.80`, `--embedding-model sentence-transformers/all-MiniLM-L6-v2`, frozen embedding inference, and maximum-weight one-to-one Hungarian matching. The text normalizer performs NFKC, lowercase, leading `#` stripping, and punctuation/whitespace handling. It does not split arbitrary concatenated strings such as `sunsetview`.

## Source provenance

Files were selected from the supplied `PHSM_core_sources.zip` only. Exact duplicate copies and unneeded historical V4/BCE-only training runners and legacy string-based C4Eval have been excluded. Two import lines were adjusted to reference the published Softmax trainer and consolidated helpers, without changing the model architecture or loss expressions. See `SOURCE_AUDIT.md`.

## Repository visibility

Availability to external readers is governed by the GitHub repository's visibility setting. A **private** repository is not a public code release. Do not state that the source is publicly released until the repository is set to Public and independently accessible.
