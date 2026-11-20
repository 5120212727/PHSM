# build_tag_text_inputs.py
import argparse
import pickle
from collections import Counter

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag_meta", type=str, default="tag_meta.pkl")
    parser.add_argument("--out", type=str, default="tag_text_inputs.pkl")
    parser.add_argument("--max_len", type=int, default=40)
    args = parser.parse_args()

    meta = pickle.load(open(args.tag_meta, "rb"))
    num_tags = meta["num_tags"]
    tag_id_to_clean_text = meta["tag_id_to_clean_text"]

    counter = Counter()
    for tag_id in range(num_tags):
        text = tag_id_to_clean_text.get(tag_id, f"tag_{tag_id}")
        for ch in text:
            counter[ch] += 1

    char2id = {"<PAD>": 0, "<UNK>": 1}
    for ch, _ in counter.most_common():
        if ch not in char2id:
            char2id[ch] = len(char2id)

    id2char = {v: k for k, v in char2id.items()}

    tag_char_ids = np.zeros((num_tags, args.max_len), dtype=np.int64)
    tag_lengths = np.zeros((num_tags,), dtype=np.int64)

    for tag_id in range(num_tags):
        text = tag_id_to_clean_text.get(tag_id, f"tag_{tag_id}")
        chars = list(text)[: args.max_len]
        tag_lengths[tag_id] = len(chars)

        for i, ch in enumerate(chars):
            tag_char_ids[tag_id, i] = char2id.get(ch, char2id["<UNK>"])

    obj = {
        "num_tags": num_tags,
        "max_len": args.max_len,
        "char2id": char2id,
        "id2char": id2char,
        "tag_char_ids": tag_char_ids,
        "tag_lengths": tag_lengths,
    }

    pickle.dump(obj, open(args.out, "wb"))

    print(f"[OK] saved {args.out}")
    print(f"[INFO] num_tags = {num_tags}")
    print(f"[INFO] char vocab size = {len(char2id)}")
    print(f"[INFO] max_len = {args.max_len}")
    print("[INFO] first 10 examples:")
    for tag_id in range(10):
        print(tag_id, tag_id_to_clean_text.get(tag_id), tag_char_ids[tag_id].tolist())


if __name__ == "__main__":
    main()