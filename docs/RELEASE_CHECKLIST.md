# PHSM source-release checklist

Status: preparation only. The exact final AutoDL training source is not yet present in this repository.

## Main-paper reproducibility deliverables

- [ ] Baseline and PHSM model classes, dataset loaders, and train/validation splits.
- [ ] Original-setting PHSM-T, PHSM-TP, PHSM-Full training scripts, including the multi-positive Softmax objective and recorded optimization settings.
- [ ] Strict cold-start splitting and safe user-history reconstruction; exclude held-out cold labels from training positives.
- [ ] Prototype construction using retained training posts only, with leakage audits.
- [ ] Independent validation-based checkpoint selection (never select checkpoints using the held-out test set).
- [ ] Original and strict-cold evaluation scripts, including mean sample-level F1 and F1 computed from aggregate P/R; document the difference.
- [ ] Cold Precision / Recall / F1 / Hit@K, preserving all 3,896 candidate labels.
- [ ] C4Eval normalization, frozen MiniLM encoder, threshold 0.80, and maximum-weight one-to-one matching, together with calibration and sensitivity analysis scripts.
- [ ] Controlled MaCon PyTorch reproduction and independent multi-seed evaluation; cite the original MaCon implementation.
- [ ] Encoder and representation component ablations, Tail bootstrap, significance and efficiency benchmarks, where reported.
- [ ] Tested environment requirements, smoke tests, example commands, config snapshots, expected output schema and checksums.
- [ ] Dataset identifiers/allowed reconstruction instructions, **subject to original data license and privacy restrictions**.

## Never commit

Instagram photos or raw post text, private user histories, credentials, full HDF5 or PKL data, model weights/checkpoints, personally identifying data, unreviewed prediction CSVs, proprietary datasets, and temporary investigation artifacts.

## Provenance and accuracy

Archives prepared earlier in the project include outdated implementations that use weighted BCE for PHSM or the old lexical C4Eval at threshold 0.30. Do not present them as the final paper implementation.

Every released experiment should identify the precise source files, dataset split, random seeds, checkpoint selection rule, evaluation formulas and relevant figure/table numbers.

Repository is currently private. A public release statement in the manuscript/response must only be made once its public visibility and reproducible contents have been verified.
