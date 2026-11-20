# Source audit

Source archive: `PHSM_core_sources.zip` (43 entries, 42 candidates plus manifest).

- Kept current PCA implementation under `phsm_pca/` and removed corresponding root-directory copies.
- Original `model.py` uses SimpleCoAttention. Kept `phsm_pca/model.py` containing ParallelCoAttention.
- Discarded historical `model_matching_v4.py` and `train_matching_v4.py` unrelated to the requested release.
- Discarded old lexical `eval_c4_semantic.py` (threshold 0.30) in favor of `c4_semantic_core.py` and `eval_c4_semantic_embedding_pca.py` (MiniLM and configurable threshold, use 0.80).
- Kept the `train_pca_v2_softmax.py` variant where the Softmax objective covers V1, V2, V3. Do not run original `train_pca.py` or the V3-only Softmax variant for stages V1/V2.
- Removed duplicates of train_baseline and train_matching_v1 stage runners, extracting their unchanged `compute_pos_weight`, `build_train_mask_and_pos_weight`, and `masked_bce_loss` functions into `train_helpers.py` because the selected training implementation imports these helpers.
- Changed C4Eval model factory import from `train_pca` to `train_pca_v2_softmax`. Both source scripts define the same `make_model` function.
- Selected both original and strict dataset VGG16 feature extractors and the respective prototype-build scripts. They target different partitions and are not duplicates.
- The source contains environment-specific default paths under `/root/macon2` and `/root/autodl-tmp`. These are sample/default paths, not included data; supply correct paths for your environment.
- Inspected all source files for hardcoded credentials, embedded connection secrets, and email addresses; no findings with simple static patterns.
- Verified Python syntax and basic importability on Python/PyTorch CPU runtime. Full original-data training and original-model checkpoint evaluation were not available and remain unverified.
- **Data caveat:** Preprocessing and split scripts use pickle files and must be run only on trusted data. No dataset, checkpoint, raw user history, or prediction CSV is included.
