"""Generator của scVAGAN — ghép Encoder + VAE Latent + Decoder.

File này chỉ làm nhiệm vụ LẮP RÁP. Các khối nằm ở file riêng:

    attention.py  Patchify/Unpatchify, TransformerBlock/Stack
    vae.py        Encoder, VAELatent, Decoder, kl_divergence

Toàn bộ dòng chảy:

    (B, G) + (B, G)  ->  (B, P, 2p)  ->  (B, P, d)  ->  (B, d)
                     ->  (B, dz)     ->  (B, P, d)  ->  (B, P, p)  ->  (B, G)

Không dùng decorator. Siêu tham số đọc từ scvagan/config.py lúc khởi tạo.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .. import config
from .vae import Decoder, Encoder, VAELatent


class Generator(nn.Module):
    """VAE + Multi-Head Attention sinh giá trị bù khuyết.

    Parameters
    ----------
    n_genes : số gene G của ma trận đầu vào.
    """

    def __init__(self, n_genes):
        super().__init__()
        self.n_genes = int(n_genes)

        self.encoder = Encoder(n_genes)
        self.latent = VAELatent()
        self.decoder = Decoder(n_genes)

        # thông tin hình học lấy từ Encoder cho tiện tra cứu
        self.n_patches = self.encoder.patchify.n_patches
        self.patch_size = self.encoder.patchify.patch_size
        self.n_pad = self.encoder.patchify.n_pad

    # ------------------------------------------------------------------
    def encode(self, x_masked, m, need_weights=False):
        """(B, G) -> mu, logvar (B, dz)."""
        h, weights = self.encoder(x_masked, m, need_weights)
        mu = self.latent.to_mu(h)
        logvar = self.latent.to_logvar(h)
        return mu, logvar, weights

    def reparameterize(self, mu, logvar, sample=True):
        return self.latent.reparameterize(mu, logvar, sample)

    def decode(self, z, need_weights=False):
        """(B, dz) -> (B, G)."""
        return self.decoder(z, need_weights)

    def forward(self, x_masked, m, sample=True, need_weights=False):
        """Lượt truyền xuôi đầy đủ.

        Returns
        -------
        x_gen  : (B, G) — ma trận sinh
        mu     : (B, dz)
        logvar : (B, dz)
        """
        h, w_enc = self.encoder(x_masked, m, need_weights)
        z, mu, logvar = self.latent(h, sample=sample)
        x_gen, w_dec = self.decoder(z, need_weights)
        if need_weights:
            return x_gen, mu, logvar, {"encoder": w_enc, "decoder": w_dec}
        return x_gen, mu, logvar

    # ------------------------------------------------------------------
    def impute(self, x, m, n_samples=1):
        """Bù khuyết: CHỈ điền tại các ô dropout (m = 1).

        Parameters
        ----------
        x : (B, G) — ma trận đã tiền xử lý, còn nguyên giá trị quan sát.
        m : (B, G) — mask dropout.
        n_samples : 1 thì dùng z = mu (tiền định). Lớn hơn 1 thì lấy mẫu
            nhiều lần rồi trung bình, và trả kèm ĐỘ LỆCH CHUẨN — chính là
            ước lượng độ bất định cho từng giá trị bù khuyết.

        Returns
        -------
        x_imputed : (B, G)
        std       : (B, G) hoặc None
        """
        self.eval()
        x_masked = x * (1.0 - m)
        with torch.no_grad():
            if n_samples <= 1:
                x_gen, _, _ = self.forward(x_masked, m, sample=False)
                std = None
            else:
                gens = torch.stack(
                    [self.forward(x_masked, m, sample=True)[0]
                     for _ in range(n_samples)], dim=0
                )
                x_gen = gens.mean(dim=0)
                std = gens.std(dim=0)
        return x * (1.0 - m) + x_gen * m, std

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self):
        """Bảng mô tả kiến trúc, tiện in ra khi viết luận văn."""
        return {
            "n_genes": self.n_genes,
            "n_patches": self.n_patches,
            "patch_size": self.patch_size,
            "n_pad": self.n_pad,
            "d_model": int(config.D_MODEL),
            "n_heads": int(config.N_HEADS),
            "d_ff": int(config.D_FF),
            "L_enc": len(self.encoder.blocks),
            "L_dec": len(self.decoder.blocks),
            "d_latent": int(config.D_LATENT),
            "dropout": float(config.MODEL_DROPOUT),
            "pre_norm": bool(config.PRE_NORM),
            "n_parameters": self.n_parameters(),
            "n_parameters_encoder": sum(p.numel() for p in self.encoder.parameters()),
            "n_parameters_latent": sum(p.numel() for p in self.latent.parameters()),
            "n_parameters_decoder": sum(p.numel() for p in self.decoder.parameters()),
        }


def build_generator(n_genes):
    """Tạo Generator với siêu tham số hiện hành trong config."""
    return Generator(n_genes)
