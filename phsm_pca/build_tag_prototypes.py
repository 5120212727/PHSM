# build_tag_prototypes.py
import os
import argparse
import pickle

import numpy as np
import torch
from tqdm import tqdm

from dataset import build_loader
from model import MaConLikeBaseline, move_batch_to_device


def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


@torch.no_grad()
def build_prototypes(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)

    print("[INFO] loading baseline checkpoint:", args.ckpt)
    ckpt = safe_torch_load(args.ckpt, map_location=device)
    train_args = ckpt["args"]

    model = MaConLikeBaseline(
        num_tags=train_args.get("num_tags", args.num_tags),
        vocab_size=train_args.get("vocab_size", args.vocab_size),
        emb_dim=train_args.get("emb_dim", args.emb_dim),
        text_hidden_dim=train_args.get("text_hidden_dim", args.text_hidden_dim),
        num_hist=train_args.get("num_hist", args.num_hist),
        dropout=train_args.get("dropout", args.dropout),
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("[INFO] building train loader...")
    dataset, loader = build_loader(
        split_pkl=args.train_pkl,
        user_history_pkl=args.user_history_pkl,
        img_feat_h5=args.img_feat_h5,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    proto_sum = np.zeros((args.num_tags, args.post_dim), dtype=np.float64)
    proto_count = np.zeros((args.num_tags,), dtype=np.int64)

    for batch in tqdm(loader, desc="Build prototypes"):
        batch = move_batch_to_device(batch, device)

        post_vec = model.encode_post(batch)  # [B, 600]
        post_vec = post_vec.detach().cpu().numpy().astype(np.float64)

        tag_ids_list = batch["tag_ids"]

        for i, tag_ids in enumerate(tag_ids_list):
            tag_ids = tag_ids.detach().cpu().numpy().tolist()

            for h in tag_ids:
                h = int(h)
                if 0 <= h < args.num_tags:
                    proto_sum[h] += post_vec[i]
                    proto_count[h] += 1

    dataset.close()

    proto = np.zeros((args.num_tags, args.post_dim), dtype=np.float32)
    proto_mask = np.zeros((args.num_tags,), dtype=np.float32)

    valid = proto_count > 0
    proto[valid] = (proto_sum[valid] / proto_count[valid, None]).astype(np.float32)
    proto_mask[valid] = 1.0

    # 读取 cold tags，确保 cold-start 标签没有 prototype
    if args.split_info and os.path.exists(args.split_info):
        split_info = pickle.load(open(args.split_info, "rb"))
        cold_tags = set(map(int, split_info["cold_tags"]))

        for h in cold_tags:
            if 0 <= h < args.num_tags:
                proto[h] = 0.0
                proto_mask[h] = 0.0
                proto_count[h] = 0

        print(f"[INFO] cold tags masked from prototype: {len(cold_tags)}")

    obj = {
        "num_tags": args.num_tags,
        "post_dim": args.post_dim,
        "prototype": proto,
        "prototype_mask": proto_mask,
        "prototype_count": proto_count,
        "source_checkpoint": args.ckpt,
        "train_pkl": args.train_pkl,
    }

    pickle.dump(obj, open(args.out, "wb"))

    print(f"[OK] saved prototypes to {args.out}")
    print("[INFO] prototype stats:")
    print("  valid prototypes:", int(proto_mask.sum()))
    print("  missing prototypes:", int(args.num_tags - proto_mask.sum()))
    print("  max count:", int(proto_count.max()))
    print("  min nonzero count:", int(proto_count[proto_count > 0].min()) if (proto_count > 0).any() else 0)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_pkl", type=str, default="trainSet_strict_seen_20_2.pkl")
    parser.add_argument("--user_history_pkl", type=str, default="user_post_samples_seen_2.pkl")
    parser.add_argument("--img_feat_h5", type=str, default="/root/autodl-tmp/macon_features/insta_imgFeat_torch.h5")
    parser.add_argument("--split_info", type=str, default="coldstart_split_strict.pkl")

    parser.add_argument("--ckpt", type=str, default="/root/autodl-tmp/macon_checkpoints/baseline_best.pt")
    parser.add_argument("--out", type=str, default="tag_prototypes.pkl")

    parser.add_argument("--num_tags", type=int, default=3896)
    parser.add_argument("--vocab_size", type=int, default=212003)
    parser.add_argument("--num_hist", type=int, default=2)
    parser.add_argument("--emb_dim", type=int, default=300)
    parser.add_argument("--text_hidden_dim", type=int, default=150)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--post_dim", type=int, default=600)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()
    build_prototypes(args)


if __name__ == "__main__":
    main()