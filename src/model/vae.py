"""Phần VAE của Generator: Encoder, không gian tiềm ẩn, Decoder.

    ENCODER   (B, G) + (B, G)
                -> Patchify                 -> (B, P, 2p)
                -> Linear Projection 2p→d   -> (B, P, d)
                -> + Gene Embedding (học được)
                -> × L_enc khối Transformer
                -> Global Pooling            -> (B, d)

    LATENT      -> μ, log σ²  Linear d→dz    -> (B, dz)
                -> z = μ + σ ⊙ ε

    DECODER     -> Linear dz→P·d, reshape    -> (B, P, d)
                -> + Gene Embedding (học được)
                -> × L_dec khối Transformer
                -> Linear d→p, Unpatchify    -> (B, G)

Không dùng decorator. Siêu tham số đọc từ scvagan/config.py lúc khởi tạo.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .. import config
from .attention import Patchify, TransformerStack, Unpatchify


class Encoder(nn.Module):
    """Nén một tế bào thành vector ẩn (B, d) — khối "đóng gói" trong sơ đồ."""

    def __init__(self, n_genes):
        super().__init__()
        d = int(config.D_MODEL)

        self.patchify = Patchify(n_genes)
        P, p = self.patchify.n_patches, self.patchify.patch_size

        # 2p vì mỗi token mang cả giá trị lẫn mask
        self.proj = nn.Linear(2 * p, d)

        # Embedding HỌC ĐƯỢC theo chỉ số token. Không dùng sin/cos vì gene
        # không có thứ tự tự nhiên — bias theo khoảng cách sẽ là giả định sai.
        self.pos = nn.Parameter(torch.zeros(1, P, d))

        self.blocks = TransformerStack(int(config.L_ENC))
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(float(config.MODEL_DROPOUT))

        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x_masked, m, need_weights=False):
        """(B, G) + (B, G) -> (B, d), kèm ma trận attention nếu cần."""
        h = self.patchify(x_masked, m)          # (B, P, 2p)
        h = self.drop(self.proj(h) + self.pos)  # (B, P, d)
        h, weights = self.blocks(h, need_weights=need_weights)
        h = self.norm(h)
        return h.mean(dim=1), weights           # Global Pooling -> (B, d)


class VAELatent(nn.Module):
    """Đầu VAE: sinh mu, logvar và lấy mẫu z.

    Đây là RANH GIỚI tất định ↔ ngẫu nhiên của cả Generator: dòng lấy mẫu
    dưới đây là nơi duy nhất randomness đi vào mô hình.

    Khi huấn luyện thì lấy mẫu z; khi suy luận dùng z = mu để kết quả tiền
    định. Lấy mẫu nhiều lần rồi tính độ lệch chuẩn chính là ước lượng ĐỘ BẤT
    ĐỊNH cho từng giá trị bù khuyết.
    """

    def __init__(self):
        super().__init__()
        d = int(config.D_MODEL)
        dz = int(config.D_LATENT)
        self.to_mu = nn.Linear(d, dz)
        self.to_logvar = nn.Linear(d, dz)

        # logvar khởi tạo gần 0 để KL không nổ ngay ở bước đầu
        nn.init.zeros_(self.to_logvar.bias)
        nn.init.normal_(self.to_logvar.weight, std=1e-3)

    def reparameterize(self, mu, logvar, sample=True):
        """z = mu + sigma * eps. Trả thẳng mu khi sample=False."""
        if not sample:
            return mu
        std = torch.exp(0.5 * logvar.clamp(-10.0, 10.0))
        return mu + std * torch.randn_like(std)

    def forward(self, h, sample=True):
        """(B, d) -> (z, mu, logvar), tất cả shape (B, dz)."""
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return self.reparameterize(mu, logvar, sample), mu, logvar


class Decoder(nn.Module):
    """Dựng lại vector biểu hiện đầy đủ từ biểu diễn tiềm ẩn."""

    def __init__(self, n_genes):
        super().__init__()
        d = int(config.D_MODEL)
        dz = int(config.D_LATENT)

        self.unpatchify = Unpatchify(n_genes)
        P, p = self.unpatchify.n_patches, self.unpatchify.patch_size
        self.n_patches = P

        self.proj = nn.Linear(dz, P * d)
        self.pos = nn.Parameter(torch.zeros(1, P, d))
        self.blocks = TransformerStack(int(config.L_DEC))
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, p)
        self.drop = nn.Dropout(float(config.MODEL_DROPOUT))

        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, z, need_weights=False):
        """(B, dz) -> (B, G), kèm ma trận attention nếu cần."""
        h = self.proj(z).view(z.shape[0], self.n_patches, -1)   # (B, P, d)
        h = self.drop(h + self.pos)
        h, weights = self.blocks(h, need_weights=need_weights)
        h = self.out(self.norm(h))                              # (B, P, p)
        return self.unpatchify(h), weights                      # (B, G)


def kl_divergence(mu, logvar):
    """KL( q(z|x) || N(0, I) ), lấy trung bình theo batch.

    Để cùng file với VAELatent vì công thức phụ thuộc giả định phân phối
    của không gian tiềm ẩn — đổi prior thì phải đổi cả hai.
    """
    return -0.5 * torch.mean(
        torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    )
