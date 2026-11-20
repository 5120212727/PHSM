# extract_imgfeat_vgg16_original.py
import os
import argparse
import pickle
from typing import Set, Dict, List

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.models as models
import torchvision.transforms as transforms


def load_ids_from_split(path: str) -> Set[str]:
    text, tags, ids, users = pickle.load(open(path, "rb"))
    return set(map(str, ids))


def load_ids_from_user_history(path: str) -> Set[str]:
    user_hist = pickle.load(open(path, "rb"))
    ids = set()

    for user_id, posts in user_hist.items():
        for inst in posts:
            # inst = (text, tags, post_id, user_id)
            ids.add(str(inst[2]))

    return ids


def collect_needed_ids(args) -> Set[str]:
    needed = set()

    split_files = [
        "trainSet_20_2.pkl",
        "testSet_20_2.pkl",
    ]

    for file in split_files:
        if os.path.exists(file):
            ids = load_ids_from_split(file)
            print(f"[INFO] loaded {len(ids)} ids from {file}")
            needed.update(ids)
        else:
            print(f"[WARN] missing split file: {file}")

    hist_file = "user_post_samples_2.pkl"
    if os.path.exists(hist_file):
        hist_ids = load_ids_from_user_history(hist_file)
        print(f"[INFO] loaded {len(hist_ids)} ids from {hist_file}")
        needed.update(hist_ids)
    else:
        print(f"[WARN] missing user history file: {hist_file}")

    print(f"[INFO] total needed image ids: {len(needed)}")

    if args.limit is not None and args.limit > 0:
        needed = set(list(sorted(needed))[: args.limit])
        print(f"[DEBUG] limit enabled, only process {len(needed)} images")

    return needed


def build_image_path_index(img_root: str, needed_ids: Set[str]) -> Dict[str, str]:
    print(f"[INFO] scanning image root: {img_root}")

    id_to_path = {}

    for root, dirs, files in os.walk(img_root):
        for name in files:
            stem, ext = os.path.splitext(name)
            if ext.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            if stem in needed_ids:
                id_to_path[stem] = os.path.join(root, name)

    print(f"[INFO] found images: {len(id_to_path)} / {len(needed_ids)}")

    missing = list(needed_ids - set(id_to_path.keys()))
    if missing:
        print(f"[WARN] missing images: {len(missing)}")
        print(f"[WARN] first 10 missing: {missing[:10]}")

    return id_to_path


def load_image(path: str, transform):
    img = Image.open(path).convert("RGB")
    return transform(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_root",
        type=str,
        default="/root/autodl-tmp/datasets/InstagramImage",
        help="Path to InstagramImage folder",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="insta_imgFeat_original_torch.h5",
        help="Output h5 feature file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N images for debugging",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Save dtype. float16 saves disk space.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")
    print(f"[INFO] torch version: {torch.__version__}")
    print(f"[INFO] cuda available: {torch.cuda.is_available()}")

    needed_ids = collect_needed_ids(args)
    id_to_path = build_image_path_index(args.img_root, needed_ids)

    # PyTorch VGG16 feature extractor
    weights = models.VGG16_Weights.IMAGENET1K_V1
    vgg = models.vgg16(weights=weights).features.to(device)
    vgg.eval()

    transform = weights.transforms()

    ids = sorted(id_to_path.keys())

    # 支持断点续跑
    with h5py.File(args.out, "a") as f:
        existing = set(f.keys())
        ids = [x for x in ids if x not in existing]
        print(f"[INFO] existing features in h5: {len(existing)}")
        print(f"[INFO] remaining images to process: {len(ids)}")

        error_log = open("extract_vgg16_errors.txt", "a", encoding="utf-8")

        for start in tqdm(range(0, len(ids), args.batch_size)):
            batch_ids = ids[start : start + args.batch_size]

            imgs = []
            valid_ids = []

            for post_id in batch_ids:
                path = id_to_path[post_id]
                try:
                    x = load_image(path, transform)
                    imgs.append(x)
                    valid_ids.append(post_id)
                except Exception as e:
                    error_log.write(f"{post_id}\t{path}\t{repr(e)}\n")
                    continue

            if len(imgs) == 0:
                continue

            x = torch.stack(imgs, dim=0).to(device)

            with torch.no_grad():
                feat = vgg(x)  # [B, 512, 7, 7]

            # 转成 MaCon/Keras 风格: [B, 7, 7, 512]
            feat = feat.permute(0, 2, 3, 1).cpu().numpy()

            if args.dtype == "float16":
                feat = feat.astype(np.float16)
            else:
                feat = feat.astype(np.float32)

            for i, post_id in enumerate(valid_ids):
                if post_id in f:
                    continue
                f.create_dataset(post_id, data=feat[i], compression="gzip")

        error_log.close()

    print(f"[DONE] saved features to {args.out}")


if __name__ == "__main__":
    main()