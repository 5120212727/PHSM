# split_coldstart.py
import pickle
import random
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Any, Set

import numpy as np


# ======================
# Config
# ======================
SEED = 42
NUM_HIST = 2
NUM_TAGS = 3896

# Head / tail definition after removing cold tags
HEAD_THRESHOLD = 100
TAIL_LOW = 20
TAIL_HIGH = 100

# Cold-start tag selection
MIN_COLD_TRAIN_FREQ = 100   # cold 标签在原训练集里至少出现这么多次，保证它不是极低频噪声
MIN_COLD_TEST_FREQ = 5      # cold 标签在原测试集里至少出现这么多次，保证 test_cold 有样本
COLD_RATIO = 0.05           # 从候选 cold tags 里抽 5%
MAX_COLD_TAGS = 100         # 最多抽多少个 cold 标签，防止删掉太多训练数据

random.seed(SEED)
np.random.seed(SEED)

Instance = Tuple[np.ndarray, np.ndarray, str, int]


# ======================
# Basic utilities
# ======================
def load_pkl_split(num_hist: int = NUM_HIST):
    text_train, tag_train, ids_train, user_train = pickle.load(
        open(f"trainSet_20_{num_hist}.pkl", "rb")
    )
    text_test, tag_test, ids_test, user_test = pickle.load(
        open(f"testSet_20_{num_hist}.pkl", "rb")
    )

    train = [
        (text_train[i], tag_train[i], str(ids_train[i]), int(user_train[i]))
        for i in range(len(text_train))
    ]

    test = [
        (text_test[i], tag_test[i], str(ids_test[i]), int(user_test[i]))
        for i in range(len(text_test))
    ]

    return train, test


def count_tag_freq(instances: List[Instance]) -> Counter:
    counter = Counter()
    for _, tags, _, _ in instances:
        for h in tags:
            counter[int(h)] += 1
    return counter


def has_any_tag(tags, tag_set: Set[int]) -> bool:
    return any(int(h) in tag_set for h in tags)


def contains_cold(inst: Instance, cold_tags: Set[int]) -> bool:
    return has_any_tag(inst[1], cold_tags)


def save_instances_as_pkl(instances: List[Instance], out_path: str):
    texts = [inst[0] for inst in instances]
    tags = [inst[1] for inst in instances]
    ids = [inst[2] for inst in instances]
    users = [inst[3] for inst in instances]

    pickle.dump(
        (
            np.array(texts, dtype=object),
            np.array(tags, dtype=object),
            ids,
            np.array(users, dtype=np.int32),
        ),
        open(out_path, "wb"),
    )
    print(f"[OK] Saved {out_path}, size={len(instances)}")


# ======================
# Cold tag selection
# ======================
def select_cold_tags(
    train: List[Instance],
    test: List[Instance],
    min_train_freq: int,
    min_test_freq: int,
    cold_ratio: float,
    max_cold_tags: int,
) -> Set[int]:
    train_freq = count_tag_freq(train)
    test_freq = count_tag_freq(test)

    candidates = []
    for h, c_train in train_freq.items():
        c_test = test_freq.get(h, 0)
        if c_train >= min_train_freq and c_test >= min_test_freq:
            candidates.append(int(h))

    candidates = sorted(candidates)

    if len(candidates) == 0:
        raise RuntimeError(
            "No cold-start candidates found. "
            "Try lowering MIN_COLD_TRAIN_FREQ or MIN_COLD_TEST_FREQ."
        )

    cold_num = max(1, int(len(candidates) * cold_ratio))
    cold_num = min(cold_num, max_cold_tags, len(candidates))

    cold_tags = set(random.sample(candidates, cold_num))

    print(f"[INFO] #cold candidates = {len(candidates)}")
    print(f"[INFO] #selected cold tags = {len(cold_tags)}")

    return cold_tags


# ======================
# Strict train/test construction
# ======================
def build_strict_train_seen(
    train: List[Instance],
    cold_tags: Set[int],
):
    train_seen = []
    removed_train = []

    for inst in train:
        if contains_cold(inst, cold_tags):
            removed_train.append(inst)
        else:
            train_seen.append(inst)

    return train_seen, removed_train


def split_test_sets(
    test: List[Instance],
    cold_tags: Set[int],
    tail_tags: Set[int],
):
    test_cold = []
    test_tail = []
    test_warm = []

    for inst in test:
        tags = inst[1]

        if has_any_tag(tags, cold_tags):
            test_cold.append(inst)
        elif has_any_tag(tags, tail_tags):
            test_tail.append(inst)
        else:
            test_warm.append(inst)

    return test_warm, test_tail, test_cold


# ======================
# Strict user history reconstruction
# ======================
def build_user_history_from_train_seen(
    train_seen: List[Instance],
    cold_tags: Set[int],
    num_hist: int = NUM_HIST,
):
    """
    重新构造用户历史，只允许从 train_seen 中采样。
    这样可以避免原始 user_post_samples_2.pkl 里泄漏 cold-start 标签。

    返回:
        user_post_samples_seen: dict[user_id] -> List[Instance]
    """

    user_to_posts = defaultdict(list)

    for inst in train_seen:
        text, tags, post_id, user_id = inst

        # 理论上 train_seen 已经不含 cold，这里再保险检查一次
        if contains_cold(inst, cold_tags):
            continue

        user_to_posts[user_id].append(inst)

    global_safe_posts = [
        inst for inst in train_seen if not contains_cold(inst, cold_tags)
    ]

    if len(global_safe_posts) == 0:
        raise RuntimeError("No safe posts available for user history.")

    user_post_samples_seen = {}

    for user_id, posts in user_to_posts.items():
        safe_posts = [p for p in posts if not contains_cold(p, cold_tags)]

        if len(safe_posts) >= num_hist:
            samples = random.sample(safe_posts, num_hist)
        elif len(safe_posts) > 0:
            # 用户自己的安全历史不足，则重复采样补齐
            samples = random.choices(safe_posts, k=num_hist)
        else:
            # 极端情况：该用户没有安全历史，从全局安全训练帖子里补齐
            samples = random.sample(global_safe_posts, num_hist)

        user_post_samples_seen[user_id] = samples

    return user_post_samples_seen


# ======================
# Checks
# ======================
def check_no_cold_in_instances(name: str, instances: List[Instance], cold_tags: Set[int]):
    bad = []
    for inst in instances:
        if contains_cold(inst, cold_tags):
            bad.append(inst[2])

    print(f"[CHECK] {name}: cold leakage posts = {len(bad)}")
    return bad


def check_no_cold_in_user_history(user_hist: Dict[int, List[Instance]], cold_tags: Set[int]):
    bad = []
    for user_id, posts in user_hist.items():
        for inst in posts:
            if contains_cold(inst, cold_tags):
                bad.append((user_id, inst[2]))

    print(f"[CHECK] user_history: cold leakage posts = {len(bad)}")
    return bad


def check_train_test_post_overlap(train_seen: List[Instance], test_sets: List[List[Instance]]):
    train_ids = set(inst[2] for inst in train_seen)
    test_ids = set()

    for test in test_sets:
        for inst in test:
            test_ids.add(inst[2])

    overlap = train_ids.intersection(test_ids)
    print(f"[CHECK] train_seen/test post_id overlap = {len(overlap)}")
    return overlap


# ======================
# Main
# ======================
def main():
    print("[INFO] Loading original train/test split...")
    train, test = load_pkl_split(NUM_HIST)

    print(f"[INFO] original train size = {len(train)}")
    print(f"[INFO] original test size  = {len(test)}")

    print("[INFO] Selecting cold-start tags...")
    cold_tags = select_cold_tags(
        train=train,
        test=test,
        min_train_freq=MIN_COLD_TRAIN_FREQ,
        min_test_freq=MIN_COLD_TEST_FREQ,
        cold_ratio=COLD_RATIO,
        max_cold_tags=MAX_COLD_TAGS,
    )

    print("[INFO] Building strict train_seen by removing all posts containing cold tags...")
    train_seen, removed_train = build_strict_train_seen(train, cold_tags)

    print(f"[INFO] train_seen size    = {len(train_seen)}")
    print(f"[INFO] removed_train size = {len(removed_train)}")

    print("[INFO] Re-counting tag frequencies on strict train_seen...")
    train_seen_freq = count_tag_freq(train_seen)

    head_tags = sorted([h for h, c in train_seen_freq.items() if c > HEAD_THRESHOLD])
    tail_tags = sorted([h for h, c in train_seen_freq.items() if TAIL_LOW < c <= TAIL_HIGH])

    head_tags_set = set(head_tags)
    tail_tags_set = set(tail_tags)

    print(f"[INFO] #head_tags after strict split = {len(head_tags)}")
    print(f"[INFO] #tail_tags after strict split = {len(tail_tags)}")

    print("[INFO] Splitting test into warm / tail / cold...")
    test_warm, test_tail, test_cold = split_test_sets(
        test=test,
        cold_tags=cold_tags,
        tail_tags=tail_tags_set,
    )

    print(f"[INFO] test_warm size = {len(test_warm)}")
    print(f"[INFO] test_tail size = {len(test_tail)}")
    print(f"[INFO] test_cold size = {len(test_cold)}")

    print("[INFO] Rebuilding strict user history from train_seen only...")
    user_post_samples_seen = build_user_history_from_train_seen(
        train_seen=train_seen,
        cold_tags=cold_tags,
        num_hist=NUM_HIST,
    )

    print(f"[INFO] #users in strict user history = {len(user_post_samples_seen)}")

    print("[INFO] Running leakage checks...")
    bad_train = check_no_cold_in_instances("train_seen", train_seen, cold_tags)
    bad_hist = check_no_cold_in_user_history(user_post_samples_seen, cold_tags)
    overlap = check_train_test_post_overlap(
        train_seen=train_seen,
        test_sets=[test_warm, test_tail, test_cold],
    )

    if len(bad_train) != 0:
        raise RuntimeError("Cold tag leakage found in train_seen.")

    if len(bad_hist) != 0:
        raise RuntimeError("Cold tag leakage found in user history.")

    if len(overlap) != 0:
        print("[WARN] train/test post_id overlap found. Please inspect original split.")

    # 保存完整 split 信息
    split = {
        "config": {
            "SEED": SEED,
            "NUM_HIST": NUM_HIST,
            "NUM_TAGS": NUM_TAGS,
            "HEAD_THRESHOLD": HEAD_THRESHOLD,
            "TAIL_LOW": TAIL_LOW,
            "TAIL_HIGH": TAIL_HIGH,
            "MIN_COLD_TRAIN_FREQ": MIN_COLD_TRAIN_FREQ,
            "MIN_COLD_TEST_FREQ": MIN_COLD_TEST_FREQ,
            "COLD_RATIO": COLD_RATIO,
            "MAX_COLD_TAGS": MAX_COLD_TAGS,
        },
        "cold_tags": sorted(list(cold_tags)),
        "head_tags": head_tags,
        "tail_tags": tail_tags,
        "tag_freq_train_seen": dict(train_seen_freq),
        "train_seen": train_seen,
        "removed_train": removed_train,
        "test_warm": test_warm,
        "test_tail": test_tail,
        "test_cold": test_cold,
        "user_post_samples_seen": user_post_samples_seen,
    }

    pickle.dump(split, open("coldstart_split_strict.pkl", "wb"))
    print("[OK] Saved coldstart_split_strict.pkl")

    # 额外保存成类似 MaCon 原始格式的 pkl，后续 PyTorch Dataset 更容易读取
    save_instances_as_pkl(train_seen, "trainSet_strict_seen_20_2.pkl")
    save_instances_as_pkl(test_warm, "testSet_warm_20_2.pkl")
    save_instances_as_pkl(test_tail, "testSet_tail_20_2.pkl")
    save_instances_as_pkl(test_cold, "testSet_cold_20_2.pkl")

    pickle.dump(
        user_post_samples_seen,
        open("user_post_samples_seen_2.pkl", "wb"),
    )
    print("[OK] Saved user_post_samples_seen_2.pkl")

    print("\n[DONE] Strict cold-start split finished.")
    print("[SUMMARY]")
    print(f"  train_seen: {len(train_seen)}")
    print(f"  test_warm : {len(test_warm)}")
    print(f"  test_tail : {len(test_tail)}")
    print(f"  test_cold : {len(test_cold)}")
    print(f"  cold_tags : {len(cold_tags)}")
    print(f"  head_tags : {len(head_tags)}")
    print(f"  tail_tags : {len(tail_tags)}")


if __name__ == "__main__":
    main()