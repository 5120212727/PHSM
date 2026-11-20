"""Loss and label-frequency helpers extracted without behavioral changes.

Sources: train_baseline.py (compute_pos_weight),
train_matching_v1.py (build_train_mask_and_pos_weight, masked_bce_loss).
The PHSM training entry point is train_pca_v2_softmax.py; the baseline uses
weighted BCE and PHSM-T/TP/Full use multi-positive Softmax with
--v3-loss macon_softmax.
"""
import pickle
import numpy as np
import torch
import torch.nn.functional as F

def compute_pos_weight(train_pkl: str, num_tags: int = 3896, max_weight: float = 50.0):
    """
    多标签任务正负样本极度不平衡，所以给正样本加权。
    pos_weight[h] = negative_count / positive_count
    """
    _, tag_train, _, _ = pickle.load(open(train_pkl, "rb"))

    pos_counts = np.zeros(num_tags, dtype=np.float32)

    for tags in tag_train:
        for h in tags:
            h = int(h)
            if 0 <= h < num_tags:
                pos_counts[h] += 1

    n = len(tag_train)
    neg_counts = n - pos_counts

    pos_weight = np.ones(num_tags, dtype=np.float32)

    valid = pos_counts > 0
    pos_weight[valid] = neg_counts[valid] / np.maximum(pos_counts[valid], 1.0)

    # 防止极低频标签权重过大导致训练不稳定
    pos_weight = np.clip(pos_weight, 1.0, max_weight)

    print("[INFO] pos_weight stats:")
    print("  min:", float(pos_weight.min()))
    print("  max:", float(pos_weight.max()))
    print("  mean:", float(pos_weight.mean()))
    print("  zero-positive tags:", int((pos_counts == 0).sum()))

    return torch.tensor(pos_weight, dtype=torch.float32)


def build_train_mask_and_pos_weight(
    train_pkl: str,
    split_info_pkl: str,
    num_tags: int = 3896,
    max_weight: float = 50.0,
    original_split: bool = False,
):
    """
    构建训练用 label mask 和 pos_weight。

    strict split 模式：
        1. cold tags 不参与训练 loss
        2. train 中没有正样本的标签不参与训练 loss

    original_split 模式：
        1. 用 MaCon 原始 trainSet_20_2.pkl / testSet_20_2.pkl
        2. 不使用 cold-start split
        3. 3896 个标签全部参与训练
        4. pos_weight 仍然根据 train_pkl 统计
    """
    _, tag_train, _, _ = pickle.load(open(train_pkl, "rb"))

    pos_counts = np.zeros(num_tags, dtype=np.float32)

    for tags in tag_train:
        for h in tags:
            h = int(h)
            if 0 <= h < num_tags:
                pos_counts[h] += 1

    n = len(tag_train)
    neg_counts = n - pos_counts

    pos_weight = np.ones(num_tags, dtype=np.float32)
    valid = pos_counts > 0
    pos_weight[valid] = neg_counts[valid] / np.maximum(pos_counts[valid], 1.0)
    pos_weight = np.clip(pos_weight, 1.0, max_weight)

    if original_split:
        cold_tags = set()
        train_mask = np.ones(num_tags, dtype=np.float32)

        print("[INFO] train label mask: original MaCon split mode")
        print("  valid train labels:", int(train_mask.sum()))
        print("  masked labels:", int(num_tags - train_mask.sum()))
        print("  cold tags:", len(cold_tags))

    else:
        split_info = pickle.load(open(split_info_pkl, "rb"))
        cold_tags = set(map(int, split_info["cold_tags"]))

        train_mask = np.zeros(num_tags, dtype=np.float32)
        train_mask[valid] = 1.0

        for h in cold_tags:
            if 0 <= h < num_tags:
                train_mask[h] = 0.0

        print("[INFO] train label mask:")
        print("  valid train labels:", int(train_mask.sum()))
        print("  masked labels:", int(num_tags - train_mask.sum()))
        print("  cold tags:", len(cold_tags))

    print("[INFO] pos_weight stats:")
    print("  min:", float(pos_weight.min()))
    print("  max:", float(pos_weight.max()))
    print("  mean:", float(pos_weight.mean()))
    print("  zero-positive tags:", int((pos_counts == 0).sum()))

    return (
        torch.tensor(train_mask, dtype=torch.float32),
        torch.tensor(pos_weight, dtype=torch.float32),
    )


def masked_bce_loss(logits, y, train_mask, pos_weight):
    """
    logits: [B, num_tags]
    y: [B, num_tags]
    train_mask: [num_tags]
    pos_weight: [num_tags]
    """
    loss_mat = F.binary_cross_entropy_with_logits(
        logits,
        y,
        reduction="none",
        pos_weight=pos_weight,
    )

    mask = train_mask.unsqueeze(0)
    denom = (mask.sum() * logits.size(0)).clamp(min=1.0)
    loss = (loss_mat * mask).sum() / denom
    return loss

