# model_matching_v3.py
import math
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import MaConLikeBaseline, move_batch_to_device
from model_matching_v1 import HashtagCharTextEncoder


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class SemanticMatchingV3(nn.Module):
    """
    V3: text + prototype + id residual fusion.

    对 seen 标签:
        tag_vec = text_vec + proto_gate * proto_vec + id_gate * id_vec

    对 cold-start 标签:
        prototype_mask = 0
        id_mask = 0
        tag_vec = text_vec

    这样 cold 标签不会使用 prototype 或 id，避免 cold-start 泄漏。
    """

    def __init__(
        self,
        tag_text_inputs_pkl: str = "tag_text_inputs.pkl",
        tag_prototypes_pkl: str = "tag_prototypes.pkl",
        num_tags: int = 3896,
        vocab_size: int = 212003,
        emb_dim: int = 300,
        text_hidden_dim: int = 150,
        num_hist: int = 2,
        dropout: float = 0.5,
        post_dim: int = 600,
    ):
        super().__init__()

        self.num_tags = num_tags
        self.post_dim = post_dim

        self.post_encoder = MaConLikeBaseline(
            num_tags=num_tags,
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            text_hidden_dim=text_hidden_dim,
            num_hist=num_hist,
            dropout=dropout,
        )

        # hashtag text input
        tag_text_obj = pickle.load(open(tag_text_inputs_pkl, "rb"))
        tag_char_ids = torch.tensor(tag_text_obj["tag_char_ids"], dtype=torch.long)
        self.char_vocab_size = len(tag_text_obj["char2id"])
        self.register_buffer("tag_char_ids", tag_char_ids)

        self.tag_text_encoder = HashtagCharTextEncoder(
            char_vocab_size=self.char_vocab_size,
            char_emb_dim=64,
            num_filters=150,
            out_dim=post_dim,
            dropout=dropout,
        )

        # prototype
        proto_obj = pickle.load(open(tag_prototypes_pkl, "rb"))
        prototype = torch.tensor(proto_obj["prototype"], dtype=torch.float32)
        prototype_mask = torch.tensor(proto_obj["prototype_mask"], dtype=torch.float32)

        if prototype.shape[0] != num_tags:
            raise ValueError(
                f"prototype num_tags mismatch: {prototype.shape[0]} vs {num_tags}"
            )

        self.register_buffer("prototype", prototype)
        self.register_buffer("prototype_mask", prototype_mask)

        self.proto_proj = nn.Linear(post_dim, post_dim, bias=False)
        nn.init.eye_(self.proto_proj.weight)

        # id branch
        self.tag_id_embedding = nn.Embedding(num_tags, post_dim)
        nn.init.normal_(self.tag_id_embedding.weight, mean=0.0, std=0.02)

        tag_ids = torch.arange(num_tags, dtype=torch.long)
        self.register_buffer("all_tag_ids", tag_ids)

        # id mask 直接复用 prototype_mask：
        # 训练集中有 prototype 的标签，说明是 seen 标签，可以使用 id；
        # cold 标签 prototype_mask=0，因此 id_mask=0。
        self.register_buffer("id_mask", prototype_mask.clone())

        # gate 初始化
        # proto gate 会从 V2 checkpoint 继承
        self.proto_gate_logit = nn.Parameter(torch.tensor(-2.0))

        # id gate 初始更小，避免随机 id embedding 一开始破坏 V2 表示
        # sigmoid(-3) ≈ 0.047
        self.id_gate_logit = nn.Parameter(torch.tensor(-3.0))

        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def load_from_v2(self, ckpt_path: str):
        """
        从 V2 checkpoint 初始化 V3。
        会继承：
        - post_encoder
        - tag_text_encoder
        - prototype/proto_proj/proto_gate
        - logit_scale
        新增的 id embedding 和 id gate 不会加载。
        """
        print(f"[INFO] loading V2 checkpoint into V3: {ckpt_path}")
        ckpt = safe_torch_load(ckpt_path, map_location="cpu")
        v2_state = ckpt["model_state"]

        own_state = self.state_dict()
        compatible_state = {}

        for k, v in v2_state.items():
            if k in own_state and own_state[k].shape == v.shape:
                compatible_state[k] = v

        missing, unexpected = self.load_state_dict(compatible_state, strict=False)

        print("[INFO] loaded compatible V2 parameters")
        print(f"[INFO] compatible keys: {len(compatible_state)}")
        print(f"[INFO] missing keys: {len(missing)}")
        print(f"[INFO] unexpected keys: {len(unexpected)}")
        print("[INFO] V2 warm metrics:", ckpt.get("warm_metrics", {}))

    def load_from_baseline(self, ckpt_path: str):
        """
        备用：只加载 baseline 的 post encoder。
        """
        print(f"[INFO] loading baseline checkpoint into V3 post encoder: {ckpt_path}")
        ckpt = safe_torch_load(ckpt_path, map_location="cpu")
        state = ckpt["model_state"]
        missing, unexpected = self.post_encoder.load_state_dict(state, strict=False)
        print("[INFO] loaded baseline post encoder")
        print(f"[INFO] missing keys: {len(missing)}")
        print(f"[INFO] unexpected keys: {len(unexpected)}")

    def encode_all_tags(self):
        text_vec = self.tag_text_encoder(self.tag_char_ids)  # [N, 600]

        proto_vec = self.proto_proj(self.prototype)          # [N, 600]
        id_vec = self.tag_id_embedding(self.all_tag_ids)     # [N, 600]

        text_vec_norm = F.normalize(text_vec, dim=-1)
        proto_vec_norm = F.normalize(proto_vec, dim=-1)
        id_vec_norm = F.normalize(id_vec, dim=-1)

        proto_mask = self.prototype_mask.unsqueeze(-1)       # [N, 1]
        id_mask = self.id_mask.unsqueeze(-1)                 # [N, 1]

        proto_gate = torch.sigmoid(self.proto_gate_logit)
        id_gate = torch.sigmoid(self.id_gate_logit)

        tag_vec = (
            text_vec_norm
            + proto_gate * proto_mask * proto_vec_norm
            + id_gate * id_mask * id_vec_norm
        )

        return tag_vec

    def forward(self, batch):
        post_vec = self.post_encoder.encode_post(batch)
        tag_vec = self.encode_all_tags()

        post_norm = F.normalize(post_vec, dim=-1)
        tag_norm = F.normalize(tag_vec, dim=-1)

        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * torch.matmul(post_norm, tag_norm.t())

        return {
            "logits": logits,
            "post_vec": post_vec,
            "tag_vec": tag_vec,
        }


if __name__ == "__main__":
    from dataset import build_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)

    dataset, loader = build_loader(
        split_pkl="trainSet_strict_seen_20_2.pkl",
        user_history_pkl="user_post_samples_seen_2.pkl",
        img_feat_h5="/root/autodl-tmp/macon_features/insta_imgFeat_torch.h5",
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device)

    model = SemanticMatchingV3().to(device)
    output = model(batch)

    print("[CHECK] V3 forward success")
    print("logits:", output["logits"].shape)
    print("post_vec:", output["post_vec"].shape)
    print("tag_vec:", output["tag_vec"].shape)
    print("proto_gate:", float(torch.sigmoid(model.proto_gate_logit).item()))
    print("id_gate:", float(torch.sigmoid(model.id_gate_logit).item()))

    loss = F.binary_cross_entropy_with_logits(output["logits"], batch["y"])
    print("loss:", float(loss.item()))

    dataset.close()