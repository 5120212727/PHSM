#!/usr/bin/env python3
"""Evaluate PHSM-PCA checkpoints with pretrained hashtag semantic embeddings.

This evaluator is deliberately separate from eval_c4_semantic.py (the original
lexical C4 implementation). It reconstructs each PCA model through the same
make_model() factory used by train_pca.py and loads checkpoint weights with
strict=True.
"""

import os
import csv
import math
import pickle
import argparse
from types import SimpleNamespace

import torch
from tqdm import tqdm

from dataset import build_loader
from model import move_batch_to_device
from train_pca_v2_softmax import make_model
from c4_semantic_core import (
    load_tag_texts,
    SemanticSimilarityCache,
    max_one_to_one_matching,
    f1_from_pr,
)


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def parse_model_specs(specs):
    models = []
    allowed = {"baseline", "v1", "v2", "v3"}
    for spec in specs:
        parts = spec.split(",", 2)
        if len(parts) != 3:
            raise ValueError(
                "Each --model must be name,type,checkpoint_path; "
                f"received: {spec}"
            )
        name, model_type, checkpoint = [x.strip() for x in parts]
        model_type = model_type.lower()
        if model_type not in allowed:
            raise ValueError(f"Unsupported PCA model type {model_type!r}; expected {sorted(allowed)}")
        models.append((name, model_type, checkpoint))
    return models


def build_pca_model(model_type, checkpoint_path, cli_args, device):
    """Recreate the exact PCA architecture from checkpoint args, then strict-load."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = safe_torch_load(checkpoint_path, map_location="cpu")
    if "model_state" not in ckpt:
        raise KeyError(f"Checkpoint lacks model_state: {checkpoint_path}")

    ckpt_args = ckpt.get("args")
    if not isinstance(ckpt_args, dict):
        raise ValueError(
            f"Checkpoint does not contain the train_pca.py args dictionary: {checkpoint_path}"
        )

    saved_stage = str(ckpt_args.get("stage", "")).lower()
    if saved_stage and saved_stage != model_type:
        raise ValueError(
            f"Model type mismatch: requested {model_type}, checkpoint stage is {saved_stage}: "
            f"{checkpoint_path}"
        )

    architecture = str(ckpt.get("architecture", ""))
    if architecture and "PHSM-PCA" not in architecture:
        raise ValueError(
            f"Checkpoint architecture does not identify PHSM-PCA: {architecture!r}"
        )

    # Use the exact argument values recorded by train_pca.py. CLI defaults are
    # only fallbacks for older checkpoints with incomplete args dictionaries.
    defaults = {
        "stage": model_type,
        "num_tags": cli_args.num_tags,
        "vocab_size": cli_args.vocab_size,
        "emb_dim": cli_args.emb_dim,
        "text_hidden_dim": cli_args.text_hidden_dim,
        "post_dim": cli_args.post_dim,
        "dropout": cli_args.dropout,
        "tag_text_inputs": cli_args.tag_text_inputs,
        "tag_prototypes": cli_args.tag_prototypes,
    }
    model_args = dict(ckpt_args)
    for key, value in defaults.items():
        model_args.setdefault(key, value)
    model_args["stage"] = model_type

    # make_model() is the same factory used during training.
    model = make_model(SimpleNamespace(**model_args))
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()

    print(f"[INFO] strict-loaded PCA {model_type}: {checkpoint_path}")
    print(f"[INFO] checkpoint epoch={ckpt.get('epoch')}")
    print(f"[INFO] checkpoint validation metrics={ckpt.get('warm_metrics', {})}")
    print(f"[INFO] architecture={architecture or 'architecture field absent'}")
    return model, ckpt


def load_target_tags(split_info_path, target_group):
    if target_group == "all":
        return None
    if not split_info_path or not os.path.isfile(split_info_path):
        raise FileNotFoundError(
            f"--split-info is required and must exist for target_group={target_group}"
        )

    with open(split_info_path, "rb") as f:
        obj = pickle.load(f)

    possible_keys = [
        f"{target_group}_tags",
        f"{target_group}_tag_ids",
        target_group,
    ]
    for key in possible_keys:
        if key in obj:
            tags = set(map(int, obj[key]))
            print(f"[INFO] target group={target_group}, key={key}, size={len(tags)}")
            return tags

    raise KeyError(
        f"Cannot find target group {target_group!r} in {split_info_path}; "
        f"available keys={list(obj.keys())}"
    )


@torch.no_grad()
def evaluate_model(model, loader, device, ks, sim_cache, target_tags=None, max_cases=30):
    max_k = max(ks)
    stat = {
        k: {
            "n": 0,
            "exact_p": 0.0,
            "exact_r": 0.0,
            "exact_f1": 0.0,
            "sem_p": 0.0,
            "sem_r": 0.0,
            "sem_f1": 0.0,
            "sem_ndcg": 0.0,
            "exact_hits": 0.0,
            "sem_score": 0.0,
            "avg_gt": 0.0,
        }
        for k in ks
    }

    cases = []
    sample_offset = 0

    for batch in tqdm(loader, desc="Eval-Semantic-C4"):
        batch = move_batch_to_device(batch, device)
        logits = model(batch)["logits"]
        y_cpu = batch["y"].detach().cpu()
        top_idx = torch.topk(logits, k=max_k, dim=1).indices.detach().cpu().tolist()

        for i in range(y_cpu.shape[0]):
            gt_ids = torch.nonzero(y_cpu[i] > 0.5, as_tuple=False).view(-1).tolist()
            gt_ids = list(map(int, gt_ids))
            if target_tags is not None:
                gt_ids = [tag_id for tag_id in gt_ids if tag_id in target_tags]
            if not gt_ids:
                continue

            for k in ks:
                pred_ids = list(map(int, top_idx[i][:k]))

                exact_hits = len(set(pred_ids) & set(gt_ids))
                exact_p = exact_hits / float(k)
                exact_r = exact_hits / float(len(gt_ids))
                exact_f1 = f1_from_pr(exact_p, exact_r)

                sem_score, sem_pairs = max_one_to_one_matching(
                    pred_ids, gt_ids, sim_cache, rank_aware=False
                )
                sem_p = sem_score / float(k)
                sem_r = sem_score / float(len(gt_ids))
                sem_f1 = f1_from_pr(sem_p, sem_r)

                sem_dcg, _ = max_one_to_one_matching(
                    pred_ids, gt_ids, sim_cache, rank_aware=True
                )
                ideal_len = min(k, len(gt_ids))
                ideal_dcg = sum(1.0 / math.log2(rank + 2.0) for rank in range(ideal_len))
                sem_ndcg = sem_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

                row = stat[k]
                row["n"] += 1
                row["exact_p"] += exact_p
                row["exact_r"] += exact_r
                row["exact_f1"] += exact_f1
                row["sem_p"] += sem_p
                row["sem_r"] += sem_r
                row["sem_f1"] += sem_f1
                row["sem_ndcg"] += sem_ndcg
                row["exact_hits"] += exact_hits
                row["sem_score"] += sem_score
                row["avg_gt"] += len(gt_ids)

                if (
                    k == max_k
                    and len(cases) < max_cases
                    and exact_f1 == 0.0
                    and sem_f1 > 0.0
                ):
                    cases.append({
                        "sample_index": sample_offset + i,
                        "K": k,
                        "exact_f1": exact_f1,
                        "sem_f1": sem_f1,
                        "sem_score": sem_score,
                        "gt_ids": gt_ids,
                        "pred_ids": pred_ids,
                        "pairs": sem_pairs,
                    })

        sample_offset += y_cpu.shape[0]

    results = {}
    for k, row in stat.items():
        n = row["n"]
        if n == 0:
            raise ValueError(
                f"No evaluable samples for target_group={target_tags}; check split_info and labels."
            )
        results[k] = {
            "num_samples": n,
            "avg_gt": row["avg_gt"] / n,
            "exact_p": row["exact_p"] / n,
            "exact_r": row["exact_r"] / n,
            "exact_f1": row["exact_f1"] / n,
            "sem_p": row["sem_p"] / n,
            "sem_r": row["sem_r"] / n,
            "sem_f1": row["sem_f1"] / n,
            "sem_ndcg": row["sem_ndcg"] / n,
            "semantic_gain": (row["sem_f1"] - row["exact_f1"]) / n,
            "exact_hits_total": row["exact_hits"],
            "sem_score_total": row["sem_score"],
        }
    return results, cases


def tag_fmt(tag_id, sim_cache):
    return f"{int(tag_id)}:{sim_cache.text(int(tag_id))}"


def save_cases(path, model_name, cases, sim_cache):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        fieldnames = [
            "model", "sample_index", "K", "exact_f1", "sem_f1", "sem_score",
            "gt_tags", "pred_tags", "matched_pairs",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for case in cases:
            gt_tags = " | ".join(tag_fmt(x, sim_cache) for x in case["gt_ids"])
            pred_tags = " | ".join(tag_fmt(x, sim_cache) for x in case["pred_ids"])
            pairs = []
            for rank0, pred_id, gt_id, raw_sim, weighted_sim in case["pairs"]:
                pairs.append(
                    f"rank{rank0 + 1}:{tag_fmt(pred_id, sim_cache)} -> "
                    f"{tag_fmt(gt_id, sim_cache)} cosine={raw_sim:.4f}"
                )
            writer.writerow({
                "model": model_name,
                "sample_index": case["sample_index"],
                "K": case["K"],
                "exact_f1": f"{case['exact_f1']:.6f}",
                "sem_f1": f"{case['sem_f1']:.6f}",
                "sem_score": f"{case['sem_score']:.6f}",
                "gt_tags": gt_tags,
                "pred_tags": pred_tags,
                "matched_pairs": " || ".join(pairs),
            })


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PHSM-PCA using independent pretrained hashtag text embeddings."
    )
    parser.add_argument("--split-pkl", required=True)
    parser.add_argument("--user-history-pkl", required=True)
    parser.add_argument("--img-feat-h5", required=True)
    parser.add_argument(
        "--model", action="append", required=True,
        help="Repeatable: name,type,checkpoint_path; type=baseline/v1/v2/v3",
    )
    parser.add_argument("--tag-meta", default="tag_meta.pkl")
    parser.add_argument("--tags-file", default="data/tags")
    parser.add_argument("--num-tags", type=int, default=3896)
    parser.add_argument("--split-info", default="")
    parser.add_argument("--target-group", default="all", choices=["all", "cold", "tail", "head"])
    parser.add_argument("--ks", default="1,3,5,7,9")
    parser.add_argument(
        "--sim-threshold", type=float, default=0.50,
        help="Provisional cosine threshold; calibrate on validation/human judgments before final reporting.",
    )
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-revision", default="")
    parser.add_argument(
        "--embedding-cache",
        default="/root/autodl-tmp/phsm_c4_semantic/tag_embeddings_all-MiniLM-L6-v2.npz",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--embedding-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--force-rebuild-embeddings", action="store_true")
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--case-csv", default="")
    parser.add_argument("--max-cases", type=int, default=30)

    # Fallback architecture arguments for older/incomplete checkpoints.
    # Current train_pca.py checkpoints contain these in ckpt['args'].
    parser.add_argument("--tag-text-inputs", default="/root/macon2/tag_text_inputs.pkl")
    parser.add_argument("--tag-prototypes", default="/root/autodl-tmp/phsm_pca_seed42/prototypes/prototypes.pkl")
    parser.add_argument("--vocab-size", type=int, default=212003)
    parser.add_argument("--emb-dim", type=int, default=300)
    parser.add_argument("--text-hidden-dim", type=int, default=150)
    parser.add_argument("--post-dim", type=int, default=600)
    parser.add_argument("--dropout", type=float, default=0.5)
    args = parser.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("--ks must contain positive integers")
    model_specs = parse_model_specs(args.model)

    tag_id_to_text = load_tag_texts(args.tag_meta, args.tags_file)
    expected_ids = set(range(args.num_tags))
    actual_ids = set(tag_id_to_text)
    if actual_ids != expected_ids:
        raise ValueError(
            f"Tag mapping mismatch: loaded={len(actual_ids)}, expected={args.num_tags}, "
            f"missing={sorted(expected_ids-actual_ids)[:20]}, "
            f"extra={sorted(actual_ids-expected_ids)[:20]}"
        )

    sim_cache = SemanticSimilarityCache(
        tag_id_to_text=tag_id_to_text,
        threshold=args.sim_threshold,
        model_name=args.embedding_model,
        model_revision=args.embedding_revision or None,
        cache_path=args.embedding_cache,
        batch_size=args.embedding_batch_size,
        device=args.embedding_device,
        force_rebuild=args.force_rebuild_embeddings,
    )
    target_tags = load_target_tags(args.split_info, args.target_group)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] PHSM-PCA evaluation device:", device)
    print("[INFO] similarity method: independent pretrained hashtag-text embeddings + cosine")
    print("[INFO] threshold:", args.sim_threshold, "(provisional until calibrated)")

    dataset, loader = build_loader(
        split_pkl=args.split_pkl,
        user_history_pkl=args.user_history_pkl,
        img_feat_h5=args.img_feat_h5,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    rows = []
    if args.case_csv and os.path.exists(args.case_csv):
        os.remove(args.case_csv)

    try:
        for model_name, model_type, checkpoint_path in model_specs:
            print("\n" + "=" * 80)
            print(f"[INFO] evaluating {model_name} ({model_type})")
            model, ckpt = build_pca_model(model_type, checkpoint_path, args, device)

            results, cases = evaluate_model(
                model=model,
                loader=loader,
                device=device,
                ks=ks,
                sim_cache=sim_cache,
                target_tags=target_tags,
                max_cases=args.max_cases,
            )

            for k, result in results.items():
                row = {
                    "model": model_name,
                    "model_type": model_type,
                    "checkpoint": checkpoint_path,
                    "checkpoint_epoch": ckpt.get("epoch"),
                    "checkpoint_val_f1_at_7": ckpt.get("warm_metrics", {}).get("f1"),
                    "split_pkl": args.split_pkl,
                    "target_group": args.target_group,
                    "K": k,
                    "num_samples": result["num_samples"],
                    "avg_gt": result["avg_gt"],
                    "exact_p": result["exact_p"],
                    "exact_r": result["exact_r"],
                    "exact_f1": result["exact_f1"],
                    "semantic_p": result["sem_p"],
                    "semantic_r": result["sem_r"],
                    "semantic_f1": result["sem_f1"],
                    "semantic_ndcg": result["sem_ndcg"],
                    "semantic_gain_over_exact": result["semantic_gain"],
                    "exact_hits_total": result["exact_hits_total"],
                    "semantic_score_total": result["sem_score_total"],
                    "similarity_method": "pretrained_embedding_cosine",
                    "embedding_model": args.embedding_model,
                    "embedding_revision": args.embedding_revision or "default",
                    "similarity_threshold": args.sim_threshold,
                }
                rows.append(row)
                print(
                    f"[{model_name} @ {k}] "
                    f"ExactF1={result['exact_f1']:.4f}, "
                    f"SemanticP={result['sem_p']:.4f}, "
                    f"SemanticR={result['sem_r']:.4f}, "
                    f"SemanticF1={result['sem_f1']:.4f}, "
                    f"SemanticNDCG={result['sem_ndcg']:.4f}, "
                    f"N={result['num_samples']}"
                )

            save_cases(args.case_csv, model_name, cases, sim_cache)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        dataset.close()

    if not rows:
        raise RuntimeError("No evaluation rows were generated")
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[DONE] metrics saved to: {args.out_csv}")
    if args.case_csv:
        print(f"[DONE] cases saved to: {args.case_csv}")


if __name__ == "__main__":
    main()
