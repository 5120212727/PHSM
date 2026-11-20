"""Core components for semantic C4 hashtag evaluation.

This module is independent of the PHSM model architecture. It can be imported
by the legacy evaluator or by the PHSM-PCA evaluator.
"""

import os
import re
import json
import pickle
import hashlib
import unicodedata

import numpy as np
import torch


def normalize_semantic_tag_text(text: str) -> str:
    """Normalize hashtag text while preserving Unicode letters and symbols."""
    text = unicodedata.normalize("NFKC", str(text)).strip().lower()
    if text.startswith("#"):
        text = text[1:]
    text = text.replace("_", " ").replace("-", " ")

    chars = []
    for ch in text:
        category = unicodedata.category(ch)
        if ch.isspace() or category[0] in {"L", "N", "S"}:
            chars.append(ch)
        else:
            chars.append(" ")
    return re.sub(r"\s+", " ", "".join(chars)).strip()


def load_tag_texts(tag_meta_path: str = "", tags_file: str = ""):
    """Load the original hashtag ID -> text mapping from JSON, pickle or tags."""
    if tag_meta_path and os.path.exists(tag_meta_path):
        if str(tag_meta_path).lower().endswith(".json"):
            with open(tag_meta_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        else:
            with open(tag_meta_path, "rb") as f:
                obj = pickle.load(f)

        # Raw strings are preferred. They preserve the source label spelling;
        # semantic normalization is applied only after loading.
        if "tag_id_to_raw_text" in obj:
            source = obj["tag_id_to_raw_text"]
        elif "tag_id_to_clean_text" in obj:
            source = obj["tag_id_to_clean_text"]
        else:
            raise KeyError(
                f"{tag_meta_path} lacks tag_id_to_raw_text/tag_id_to_clean_text"
            )
        result = {int(k): normalize_semantic_tag_text(v) for k, v in source.items()}
        print(f"[INFO] loaded {len(result)} hashtag texts from {tag_meta_path}")
        return result

    if tags_file and os.path.exists(tags_file):
        result = {}
        with open(tags_file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) == 1:
                    parts = line.split(maxsplit=1)
                if len(parts) == 2 and parts[0].strip().isdigit():
                    result[int(parts[0].strip())] = normalize_semantic_tag_text(parts[1])
                elif len(parts) == 1 and parts[0].isdigit():
                    result[int(parts[0])] = f"tag {int(parts[0])}"
                else:
                    print(f"[WARN] cannot parse tags_file line {line_no}: {line[:80]}")
        print(f"[INFO] loaded {len(result)} hashtag texts from {tags_file}")
        return result

    raise FileNotFoundError(
        "Semantic C4 requires tag_meta.pkl/json or the original data/tags file."
    )


class SemanticSimilarityCache:
    """Precompute pretrained hashtag embeddings and their cosine matrix."""

    def __init__(
        self,
        tag_id_to_text,
        threshold=0.50,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_revision=None,
        cache_path="tag_embeddings_all-MiniLM-L6-v2.npz",
        batch_size=128,
        device="auto",
        force_rebuild=False,
    ):
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("Semantic cosine threshold must be between 0 and 1")

        self.tag_id_to_text = {int(k): str(v) for k, v in tag_id_to_text.items()}
        self.threshold = float(threshold)
        self.model_name = str(model_name)
        self.model_revision = str(model_revision or "default")
        self.cache_path = str(cache_path)

        self.tag_ids = sorted(self.tag_id_to_text)
        self.id_to_row = {tag_id: row for row, tag_id in enumerate(self.tag_ids)}
        self.texts = [self.tag_id_to_text[tag_id] for tag_id in self.tag_ids]

        empty_ids = [i for i, text in zip(self.tag_ids, self.texts) if not text.strip()]
        if empty_ids:
            raise ValueError(
                f"Empty normalized hashtag text for {len(empty_ids)} labels; "
                f"first IDs={empty_ids[:20]}"
            )

        text_payload = "\n".join(f"{i}\t{t}" for i, t in zip(self.tag_ids, self.texts))
        text_hash = hashlib.sha256(text_payload.encode("utf-8")).hexdigest()
        fingerprint_payload = f"{self.model_name}\n{self.model_revision}\n{text_hash}"
        self.fingerprint = hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()

        embeddings = None
        if os.path.exists(self.cache_path) and not force_rebuild:
            try:
                with np.load(self.cache_path, allow_pickle=False) as data:
                    metadata = json.loads(str(data["metadata"].item()))
                    cached_ids = data["tag_ids"].astype(np.int64)
                    cached_embeddings = data["embeddings"].astype(np.float32)
                if (
                    metadata.get("fingerprint") == self.fingerprint
                    and np.array_equal(cached_ids, np.asarray(self.tag_ids, dtype=np.int64))
                    and cached_embeddings.ndim == 2
                    and cached_embeddings.shape[0] == len(self.tag_ids)
                ):
                    embeddings = cached_embeddings
                    print(f"[INFO] loaded embedding cache: {self.cache_path}")
                else:
                    print("[WARN] embedding cache does not match model/texts; rebuilding")
            except Exception as exc:
                print(f"[WARN] failed to read embedding cache ({exc}); rebuilding")

        if embeddings is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "Semantic C4 requires sentence-transformers. Install it in the "
                    "current project environment before running this evaluator."
                ) from exc

            if device == "auto":
                encode_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                encode_device = device

            print("[INFO] embedding model:", self.model_name)
            print("[INFO] embedding revision:", self.model_revision)
            print("[INFO] embedding device:", encode_device)
            kwargs = {"revision": model_revision} if model_revision else {}
            encoder = SentenceTransformer(self.model_name, device=encode_device, **kwargs)
            embeddings = encoder.encode(
                self.texts,
                batch_size=int(batch_size),
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
            del encoder
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if embeddings.ndim != 2 or embeddings.shape[0] != len(self.tag_ids):
                raise RuntimeError(f"Unexpected embedding shape: {embeddings.shape}")
            if not np.isfinite(embeddings).all():
                raise RuntimeError("Embedding model returned NaN or infinite values")

            norms = np.linalg.norm(embeddings, axis=1)
            if np.any(norms < 1e-8):
                bad = np.where(norms < 1e-8)[0].tolist()[:20]
                raise RuntimeError(f"Zero-norm hashtag embeddings at rows: {bad}")
            embeddings = embeddings / np.maximum(norms[:, None], 1e-12)

            os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)), exist_ok=True)
            metadata = {
                "fingerprint": self.fingerprint,
                "model_name": self.model_name,
                "model_revision": self.model_revision,
                "text_sha256": text_hash,
                "num_tags": len(self.tag_ids),
                "embedding_dim": int(embeddings.shape[1]),
                "normalization": "NFKC/lowercase/remove leading #/split underscore and hyphen/preserve Unicode letters, numbers and symbols",
            }
            np.savez_compressed(
                self.cache_path,
                tag_ids=np.asarray(self.tag_ids, dtype=np.int64),
                embeddings=embeddings,
                metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
            )
            print(f"[INFO] saved embedding cache: {self.cache_path}")

        self.embeddings = embeddings
        self.sim_matrix = np.matmul(embeddings, embeddings.T).astype(np.float32, copy=False)
        np.clip(self.sim_matrix, -1.0, 1.0, out=self.sim_matrix)
        print(
            f"[INFO] cosine matrix={self.sim_matrix.shape}, "
            f"dim={self.embeddings.shape[1]}, threshold={self.threshold:.3f}"
        )

    def text(self, tag_id):
        tag_id = int(tag_id)
        return self.tag_id_to_text.get(tag_id, f"tag_{tag_id}")

    def raw_sim(self, a, b):
        a, b = int(a), int(b)
        if a == b:
            return 1.0
        if a not in self.id_to_row or b not in self.id_to_row:
            raise KeyError(f"Unknown hashtag ID: {a}, {b}")
        return float(self.sim_matrix[self.id_to_row[a], self.id_to_row[b]])

    def sim(self, a, b):
        score = self.raw_sim(a, b)
        if score < self.threshold:
            return 0.0
        return max(0.0, min(1.0, score))


def max_one_to_one_matching(pred_ids, gt_ids, sim_cache, rank_aware=False):
    """Maximum-weight one-to-one matching using the Hungarian algorithm."""
    pred_ids = list(map(int, pred_ids))
    gt_ids = list(map(int, gt_ids))
    if not pred_ids or not gt_ids:
        return 0.0, []

    weights = [1.0 / np.log2(i + 2.0) if rank_aware else 1.0 for i in range(len(pred_ids))]
    matrix = np.asarray(
        [[weights[i] * sim_cache.sim(p, g) for g in gt_ids] for i, p in enumerate(pred_ids)],
        dtype=np.float64,
    )

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise RuntimeError(
            "Semantic C4 requires scipy.optimize.linear_sum_assignment for Hungarian matching."
        ) from exc

    row_ind, col_ind = linear_sum_assignment(-matrix)
    score = 0.0
    pairs = []
    for r, c in zip(row_ind, col_ind):
        value = float(matrix[r, c])
        if value <= 0:
            continue
        raw = sim_cache.raw_sim(pred_ids[r], gt_ids[c])
        score += value
        pairs.append((r, pred_ids[r], gt_ids[c], raw, value))
    return float(score), pairs


def f1_from_pr(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)
