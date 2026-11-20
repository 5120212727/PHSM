# dataset.py
import os
import pickle
from typing import Dict, List, Tuple, Any, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


Instance = Tuple[np.ndarray, np.ndarray, str, int]


def pad_1d_sequence(seq, max_len: int, pad_value: int = 0, shift: int = 0) -> np.ndarray:
    """
    将不定长 id 序列 padding/truncate 到固定长度。
    
    参数：
        seq: 原始 id 序列
        max_len: 最大长度
        pad_value: padding 值
        shift: 是否对非 padding id 做偏移
    
    说明：
        对 tag 序列，如果 tag_id=0 是合法标签，那么为了避免和 padding=0 冲突，
        可以设置 shift=1，使真实 tag_id 变成 tag_id+1。
    """
    seq = list(seq)
    seq = seq[:max_len]
    arr = np.full((max_len,), pad_value, dtype=np.int64)

    if len(seq) > 0:
        arr[:len(seq)] = np.array(seq, dtype=np.int64) + shift

    return arr


def make_multihot(tags, num_tags: int) -> np.ndarray:
    """
    把标签 id 列表转成 multi-hot 向量。
    """
    y = np.zeros((num_tags,), dtype=np.float32)
    for h in tags:
        h = int(h)
        if 0 <= h < num_tags:
            y[h] = 1.0
    return y


class InstagramHashtagDataset(Dataset):
    """
    MaCon / Instagram hashtag recommendation 数据集读取器。

    每条样本返回：
        text: 当前帖子文本 id，shape [seq_len]
        img: 当前帖子 VGG16 图片特征，shape [512, 7, 7]
        hist_text: 用户历史文本，shape [num_hist, seq_len]
        hist_img: 用户历史图片特征，shape [num_hist, 512, 7, 7]
        hist_tag: 用户历史标签 id，shape [num_hist, max_tag_len]
        y: 当前帖子真实标签 multi-hot，shape [num_tags]
        tag_ids: 当前帖子真实标签原始 id 列表
        post_id: 当前帖子 id
        user_id: 用户 id
    """

    def __init__(
        self,
        split_pkl: str,
        user_history_pkl: str,
        img_feat_h5: str,
        num_tags: int = 3896,
        seq_len: int = 30,
        max_tag_len: int = 48,
        num_hist: int = 2,
        img_as_float32: bool = True,
    ):
        super().__init__()

        self.split_pkl = split_pkl
        self.user_history_pkl = user_history_pkl
        self.img_feat_h5 = img_feat_h5

        self.num_tags = num_tags
        self.seq_len = seq_len
        self.max_tag_len = max_tag_len
        self.num_hist = num_hist
        self.img_as_float32 = img_as_float32

        if not os.path.exists(split_pkl):
            raise FileNotFoundError(f"Cannot find split file: {split_pkl}")

        if not os.path.exists(user_history_pkl):
            raise FileNotFoundError(f"Cannot find user history file: {user_history_pkl}")

        if not os.path.exists(img_feat_h5):
            raise FileNotFoundError(f"Cannot find image feature h5 file: {img_feat_h5}")

        text, tags, ids, users = pickle.load(open(split_pkl, "rb"))

        self.instances: List[Instance] = [
            (text[i], tags[i], str(ids[i]), int(users[i]))
            for i in range(len(text))
        ]

        self.user_history: Dict[int, List[Instance]] = pickle.load(open(user_history_pkl, "rb"))

        # h5py 文件不要在 __init__ 里打开，避免 DataLoader 多进程出问题
        self._h5: Optional[h5py.File] = None

        print(f"[INFO] Loaded dataset: {split_pkl}")
        print(f"[INFO] size = {len(self.instances)}")
        print(f"[INFO] user history users = {len(self.user_history)}")
        print(f"[INFO] image h5 = {img_feat_h5}")

    def __len__(self):
        return len(self.instances)

    def _get_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.img_feat_h5, "r")
        return self._h5

    def __getstate__(self):
        """
        防止 DataLoader 多进程 pickle h5 文件句柄。
        """
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def _load_img_feat(self, post_id: str) -> torch.Tensor:
        """
        从 h5 里读取图片特征。
        h5 中保存形状是 [7, 7, 512]。
        这里转成 PyTorch 常用形状 [512, 7, 7]。
        """
        h5 = self._get_h5()

        if post_id not in h5:
            raise KeyError(f"Image feature not found for post_id={post_id}")

        ds = h5[post_id]

        feat = np.empty(ds.shape, dtype=np.float32)
        ds.id.read(
            h5py.h5s.ALL,
            h5py.h5s.ALL,
            feat,
            mtype=h5py.h5t.NATIVE_FLOAT,
        )
        feat = np.transpose(feat, (2, 0, 1))  # [512, 7, 7]

        if self.img_as_float32:
            feat = feat.astype(np.float32)

        return torch.from_numpy(feat)

    def _get_user_histories(self, user_id: int) -> List[Instance]:
        """
        获取用户历史帖子。
        如果某个用户没有历史，理论上不应该发生；
        这里做兜底：随机使用当前数据集前 num_hist 条。
        """
        histories = self.user_history.get(user_id, None)

        if histories is None or len(histories) == 0:
            raise KeyError(f"No training-only user history found for user_id={user_id}; refusing split-local fallback")

        if len(histories) >= self.num_hist:
            histories = histories[: self.num_hist]
        else:
            # 不足 num_hist 时重复补齐
            histories = histories + [histories[-1]] * (self.num_hist - len(histories))

        return histories

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text, tags, post_id, user_id = self.instances[idx]

        # 当前帖子文本
        text_pad = pad_1d_sequence(
            text,
            max_len=self.seq_len,
            pad_value=0,
            shift=0,
        )

        # 当前帖子图片特征
        img_feat = self._load_img_feat(post_id)

        # 当前帖子标签
        tag_ids = np.array([int(h) for h in tags], dtype=np.int64)
        y = make_multihot(tag_ids, self.num_tags)

        # 用户历史
        histories = self._get_user_histories(user_id)

        hist_text_list = []
        hist_img_list = []
        hist_tag_list = []

        for hist_text, hist_tags, hist_post_id, hist_user_id in histories:
            hist_text_pad = pad_1d_sequence(
                hist_text,
                max_len=self.seq_len,
                pad_value=0,
                shift=0,
            )

            # 注意：历史 tag 序列这里 shift=1
            # 因为后续如果使用 nn.Embedding，padding=0，真实 tag_id 从 1 开始
            hist_tag_pad = pad_1d_sequence(
                hist_tags,
                max_len=self.max_tag_len,
                pad_value=0,
                shift=1,
            )

            hist_img_feat = self._load_img_feat(str(hist_post_id))

            hist_text_list.append(hist_text_pad)
            hist_tag_list.append(hist_tag_pad)
            hist_img_list.append(hist_img_feat)

        hist_text_arr = np.stack(hist_text_list, axis=0)  # [num_hist, seq_len]
        hist_tag_arr = np.stack(hist_tag_list, axis=0)    # [num_hist, max_tag_len]
        hist_img_tensor = torch.stack(hist_img_list, dim=0)  # [num_hist, 512, 7, 7]

        sample = {
            "text": torch.from_numpy(text_pad).long(),
            "img": img_feat.float(),
            "hist_text": torch.from_numpy(hist_text_arr).long(),
            "hist_img": hist_img_tensor.float(),
            "hist_tag": torch.from_numpy(hist_tag_arr).long(),
            "y": torch.from_numpy(y).float(),
            "tag_ids": torch.from_numpy(tag_ids).long(),
            "post_id": post_id,
            "user_id": torch.tensor(user_id, dtype=torch.long),
        }

        return sample

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    自定义 batch 整理函数。
    因为 tag_ids 是不定长，所以保留为 list；
    其他定长字段 stack。
    """
    out = {
        "text": torch.stack([x["text"] for x in batch], dim=0),
        "img": torch.stack([x["img"] for x in batch], dim=0),
        "hist_text": torch.stack([x["hist_text"] for x in batch], dim=0),
        "hist_img": torch.stack([x["hist_img"] for x in batch], dim=0),
        "hist_tag": torch.stack([x["hist_tag"] for x in batch], dim=0),
        "y": torch.stack([x["y"] for x in batch], dim=0),
        "tag_ids": [x["tag_ids"] for x in batch],
        "post_id": [x["post_id"] for x in batch],
        "user_id": torch.stack([x["user_id"] for x in batch], dim=0),
    }
    return out


def build_loader(
    split_pkl: str,
    user_history_pkl: str = "user_post_samples_seen_2.pkl",
    img_feat_h5: str = "/root/autodl-tmp/macon_features/insta_imgFeat_torch.h5",
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    generator=None,
):
    dataset = InstagramHashtagDataset(
        split_pkl=split_pkl,
        user_history_pkl=user_history_pkl,
        img_feat_h5=img_feat_h5,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )

    return dataset, loader


if __name__ == "__main__":
    # 默认先测试训练集读取
    dataset, loader = build_loader(
        split_pkl="trainSet_strict_seen_20_2.pkl",
        user_history_pkl="user_post_samples_seen_2.pkl",
        img_feat_h5="/root/autodl-tmp/macon_features/insta_imgFeat_torch.h5",
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(loader))

    print("\n[CHECK] batch loaded successfully")
    print("text:", batch["text"].shape)
    print("img:", batch["img"].shape)
    print("hist_text:", batch["hist_text"].shape)
    print("hist_img:", batch["hist_img"].shape)
    print("hist_tag:", batch["hist_tag"].shape)
    print("y:", batch["y"].shape)
    print("tag_ids lens:", [len(x) for x in batch["tag_ids"]])
    print("post_id:", batch["post_id"][:4])
    print("user_id:", batch["user_id"].shape)

    dataset.close()