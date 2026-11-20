# model_matching_v1.py
import math
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import MaConLikeBaseline, move_batch_to_device


class HashtagCharTextEncoder(nn.Module):
    """
    hashtag 文本编码器。
    输入 hashtag 字符序列，输出 hashtag 向量。
    """
    def __init__(
        self,
        char_vocab_size: int,
        char_emb_dim: int = 64,
        num_filters: int = 150,
        out_dim: int = 600,
        dropout: float = 0.2,
        padding_idx: int = 0,
    ):
        super().__init__()

        self.char_embedding = nn.Embedding(
            num_embeddings=char_vocab_size,
            embedding_dim=char_emb_dim,
            padding_idx=padding_idx,
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(char_emb_dim, num_filters, kernel_size=k, padding=0)
            for k in [2, 3, 4, 5]
        ])

        conv_out_dim = num_filters * len(self.convs)

        self.proj = nn.Sequential(
            nn.Linear(conv_out_dim, out_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

    def forward(self, tag_char_ids):
        """
        tag_char_ids: [N, L]
        return:
            tag_vec: [N, out_dim]
        """
        x = self.char_embedding(tag_char_ids)  # [N, L, C]
        x = x.permute(0, 2, 1)                 # [N, C, L]

        outs = []
        for conv in self.convs:
            h = F.relu(conv(x))                # [N, F, L']
            h = torch.max(h, dim=-1).values    # [N, F]
            outs.append(h)

        h = torch.cat(outs, dim=-1)            # [N, F * num_kernels]
        return self.proj(h)


class SemanticMatchingV1(nn.Module):
    """
    V1: post-to-hashtag semantic matching，只使用 hashtag 文本表示。

    post encoder:
        复用 MaCon-like backbone，得到 post_vec [B, 600]

    hashtag encoder:
        字符级 hashtag text encoder，得到 tag_vec [3896, 600]

    score:
        cosine(post_vec, tag_vec) * scale
    """
    def __init__(
        self,
        tag_text_inputs_pkl: str = "tag_text_inputs.pkl",
        num_tags: int = 3896,
        vocab_size: int = 212003,
        emb_dim: int = 300,
        text_hidden_dim: int = 150,
        num_hist: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.num_tags = num_tags
        self.post_encoder = MaConLikeBaseline(
            num_tags=num_tags,
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            text_hidden_dim=text_hidden_dim,
            num_hist=num_hist,
            dropout=dropout,
        )

        tag_text_obj = pickle.load(open(tag_text_inputs_pkl, "rb"))
        tag_char_ids = torch.tensor(tag_text_obj["tag_char_ids"], dtype=torch.long)

        self.char_vocab_size = len(tag_text_obj["char2id"])
        self.register_buffer("tag_char_ids", tag_char_ids)

        self.tag_text_encoder = HashtagCharTextEncoder(
            char_vocab_size=self.char_vocab_size,
            char_emb_dim=64,
            num_filters=150,
            out_dim=emb_dim * 2,
            dropout=dropout,
        )

        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def load_post_encoder_from_baseline(self, ckpt_path: str):
        print(f"[INFO] loading post encoder from baseline: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt["model_state"]

        missing, unexpected = self.post_encoder.load_state_dict(state, strict=False)
        print(f"[INFO] loaded baseline post encoder")
        print(f"[INFO] missing keys: {len(missing)}")
        print(f"[INFO] unexpected keys: {len(unexpected)}")

    def encode_all_tags(self):
        return self.tag_text_encoder(self.tag_char_ids)

    def forward(self, batch):
        post_vec = self.post_encoder.encode_post(batch)     # [B, 600]
        tag_vec = self.encode_all_tags()                    # [num_tags, 600]

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

    model = SemanticMatchingV1().to(device)
    output = model(batch)

    logits = output["logits"]
    print("[CHECK] forward success")
    print("logits:", logits.shape)
    print("post_vec:", output["post_vec"].shape)
    print("tag_vec:", output["tag_vec"].shape)

    loss = F.binary_cross_entropy_with_logits(logits, batch["y"])
    print("loss:", float(loss.item()))

    dataset.close()