# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMeanPooling(nn.Module):
    """
    对序列做 masked mean pooling。
    输入:
        x: [B, L, D]
        ids: [B, L]
    输出:
        pooled: [B, D]
    """
    def forward(self, x, ids):
        mask = (ids != 0).float().unsqueeze(-1)  # [B, L, 1]
        x = x * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
        return x.sum(dim=1) / denom


class TextEncoder(nn.Module):
    """
    文本编码器:
    word ids -> embedding -> BiLSTM -> sequence features
    """
    def __init__(
        self,
        vocab_size=212003,
        emb_dim=300,
        hidden_dim=150,
        dropout=0.2,
        padding_idx=0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=padding_idx,
        )

        # 双向 LSTM，hidden_dim * 2 = emb_dim
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, text_ids):
        """
        text_ids: [B, L]
        return:
            seq_feat: [B, L, D]
        """
        x = self.embedding(text_ids)
        x = self.dropout(x)
        seq_feat, _ = self.lstm(x)
        seq_feat = self.dropout(seq_feat)
        return seq_feat


class ImageEncoder(nn.Module):
    """
    图片特征编码器:
    VGG16 特征 [B, 512, 7, 7] -> region features [B, 49, D]
    """
    def __init__(self, img_in_dim=512, emb_dim=300, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(img_in_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, img):
        """
        img: [B, 512, 7, 7]
        return:
            region_feat: [B, 49, D]
        """
        b = img.size(0)
        x = img.view(b, 512, -1).permute(0, 2, 1)  # [B, 49, 512]
        x = torch.tanh(self.proj(x))               # [B, 49, D]
        x = self.dropout(x)
        return x


class ParallelCoAttention(nn.Module):
    """MaCon original parallel co-attention formulation (dim_k=100).

    C = text Wb image.T
    H_v = tanh(image Wv + C.T (text Wq))
    H_q = tanh(text Wq + C (image Wv))
    output = attention(text) + attention(image), shape [B, 300].
    The MaCon reproduction does not apply a padding mask at this level.
    """
    def __init__(self, dim=300, dim_k=100):
        super().__init__()
        self.Wb = nn.Parameter(torch.empty(dim, dim))
        self.Wq = nn.Parameter(torch.empty(dim, dim_k))
        self.Wv = nn.Parameter(torch.empty(dim, dim_k))
        self.Whq = nn.Parameter(torch.empty(dim_k, 1))
        self.Whv = nn.Parameter(torch.empty(dim_k, 1))
        for weight in (self.Wb, self.Wq, self.Wv, self.Whq, self.Whv):
            nn.init.normal_(weight, mean=0.0, std=0.05)

    def forward(self, text_feat, img_feat, text_ids=None):
        # text [B, 30, D], img [B, 49, D]
        c = torch.bmm(text_feat @ self.Wb, img_feat.transpose(1, 2))
        t_k = text_feat @ self.Wq
        v_k = img_feat @ self.Wv
        hv = torch.tanh(v_k + torch.bmm(c.transpose(1, 2), t_k))
        hq = torch.tanh(t_k + torch.bmm(c, v_k))
        av = torch.softmax((hv @ self.Whv).squeeze(-1), dim=1)
        aq = torch.softmax((hq @ self.Whq).squeeze(-1), dim=1)
        return (text_feat * aq.unsqueeze(-1)).sum(dim=1) + (img_feat * av.unsqueeze(-1)).sum(dim=1)


class ContentEncoder(nn.Module):
    """
    当前帖子或历史帖子的内容编码器:
        text + image -> content representation
    """
    def __init__(
        self,
        vocab_size=212003,
        emb_dim=300,
        text_hidden_dim=150,
        dropout=0.2,
    ):
        super().__init__()
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            hidden_dim=text_hidden_dim,
            dropout=dropout,
        )
        self.image_encoder = ImageEncoder(
            img_in_dim=512,
            emb_dim=emb_dim,
            dropout=dropout,
        )
        self.co_attention = ParallelCoAttention(dim=emb_dim, dim_k=100)

    def forward(self, text_ids, img):
        """
        text_ids: [B, L]
        img: [B, 512, 7, 7]
        return:
            content_vec: [B, D]
        """
        text_feat = self.text_encoder(text_ids)
        img_feat = self.image_encoder(img)
        content_vec = self.co_attention(text_feat, img_feat, text_ids)
        return content_vec


class HistTagEncoder(nn.Module):
    """
    用户历史标签编码器。
    dataset.py 里 hist_tag 已经做过 shift=1:
        padding = 0
        原始 tag_id h -> h + 1
    所以这里 embedding 大小是 num_tags + 1。
    """
    def __init__(self, num_tags=3896, emb_dim=300, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_tags + 1,
            embedding_dim=emb_dim,
            padding_idx=0,
        )
        self.pool = MaskedMeanPooling()
        self.dropout = nn.Dropout(dropout)

    def forward(self, hist_tag_ids):
        """
        hist_tag_ids: [B, H, T]
        return:
            hist_tag_vec: [B, H, D]
        """
        b, h, t = hist_tag_ids.shape

        flat_ids = hist_tag_ids.view(b * h, t)       # [B*H, T]
        emb = self.embedding(flat_ids)               # [B*H, T, D]
        pooled = self.pool(emb, flat_ids)            # [B*H, D]
        pooled = self.dropout(pooled)

        return pooled.view(b, h, -1)                 # [B, H, D]


class UserHabitEncoder(nn.Module):
    """
    用户历史习惯模块。

    输入:
        当前帖子表示 query_vec: [B, D]
        历史帖子表示 hist_content_vec: [B, H, D]
        历史标签表示 hist_tag_vec: [B, H, D]

    输出:
        influence_vec: [B, D]
    """
    def __init__(self, emb_dim=300, dropout=0.2):
        super().__init__()
        self.att = nn.Linear(emb_dim * 2, 1)

        self.influence_proj = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

    def forward(self, query_vec, hist_content_vec, hist_tag_vec):
        b, h, d = hist_content_vec.shape

        query_expand = query_vec.unsqueeze(1).expand(-1, h, -1)  # [B, H, D]

        # 和 MaCon 类似，当前帖子与历史帖子内容做交互
        sim = query_expand * hist_content_vec                   # [B, H, D]

        # 拼接历史标签表示
        hist_state = torch.cat([sim, hist_tag_vec], dim=-1)      # [B, H, 2D]

        score = self.att(hist_state).squeeze(-1)                 # [B, H]
        alpha = F.softmax(score, dim=1)                          # [B, H]

        influence = torch.sum(hist_state * alpha.unsqueeze(-1), dim=1)  # [B, 2D]
        influence_vec = self.influence_proj(influence)                  # [B, D]

        return influence_vec


class MaConLikeBaseline(nn.Module):
    """
    PyTorch 版 MaCon-like baseline。

    输入来自 dataset.py 的 batch:
        text: [B, 30]
        img: [B, 512, 7, 7]
        hist_text: [B, 2, 30]
        hist_img: [B, 2, 512, 7, 7]
        hist_tag: [B, 2, 48]

    输出:
        logits: [B, num_tags]
        post_vec: [B, 2D]
    """
    def __init__(
        self,
        num_tags=3896,
        vocab_size=212003,
        emb_dim=300,
        text_hidden_dim=150,
        num_hist=2,
        dropout=0.5,
    ):
        super().__init__()

        self.num_tags = num_tags
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.num_hist = num_hist

        self.content_encoder = ContentEncoder(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            text_hidden_dim=text_hidden_dim,
            dropout=dropout,
        )

        self.hist_tag_encoder = HistTagEncoder(
            num_tags=num_tags,
            emb_dim=emb_dim,
            dropout=dropout,
        )

        self.user_habit_encoder = UserHabitEncoder(
            emb_dim=emb_dim,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, num_tags),
        )

    def encode_post(self, batch):
        text = batch["text"]              # [B, L]
        img = batch["img"]                # [B, 512, 7, 7]
        hist_text = batch["hist_text"]    # [B, H, L]
        hist_img = batch["hist_img"]      # [B, H, 512, 7, 7]
        hist_tag = batch["hist_tag"]      # [B, H, T]

        b, h, l = hist_text.shape

        # 当前帖子内容表示
        query_vec = self.content_encoder(text, img)  # [B, D]

        # 历史帖子内容表示
        hist_text_flat = hist_text.view(b * h, l)
        hist_img_flat = hist_img.view(b * h, 512, 7, 7)

        hist_content_flat = self.content_encoder(hist_text_flat, hist_img_flat)  # [B*H, D]
        hist_content_vec = hist_content_flat.view(b, h, -1)                     # [B, H, D]

        # 历史标签表示
        hist_tag_vec = self.hist_tag_encoder(hist_tag)                          # [B, H, D]

        # 用户习惯影响向量
        influence_vec = self.user_habit_encoder(
            query_vec=query_vec,
            hist_content_vec=hist_content_vec,
            hist_tag_vec=hist_tag_vec,
        )                                                                        # [B, D]

        # 最终帖子表示
        post_vec = torch.cat([query_vec, influence_vec], dim=-1)                 # [B, 2D]

        return post_vec

    def forward(self, batch):
        post_vec = self.encode_post(batch)
        logits = self.classifier(post_vec)
        return {
            "logits": logits,
            "post_vec": post_vec,
        }


def move_batch_to_device(batch, device):
    """
    把 batch 里的 tensor 移到 GPU。
    post_id 和 tag_ids 这种非定长/字符串字段不移动。
    """
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


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

    model = MaConLikeBaseline().to(device)
    model.train()

    output = model(batch)
    logits = output["logits"]
    post_vec = output["post_vec"]

    print("[CHECK] forward success")
    print("logits:", logits.shape)
    print("post_vec:", post_vec.shape)

    # 测试 loss 是否能正常计算
    y = batch["y"]
    loss = F.binary_cross_entropy_with_logits(logits, y)
    print("loss:", float(loss.item()))

    dataset.close()