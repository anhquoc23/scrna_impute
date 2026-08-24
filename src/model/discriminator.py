"""Discriminator của scVAGAN — BẢN TẠM.

Khối 4 chưa được thiết kế bằng sơ đồ như Generator, nên đây mới là bản rút
gọn để vòng lặp đối kháng chạy được. Khi thiết kế xong, thay nội dung file
này mà không phải đụng tới phần còn lại.

Đầu vào là ma trận LAI, không phải X̂ thô:

    X̂_lai = X ⊙ (1 − M) + X̂ ⊙ M

Nếu đưa X̂ thô vào thì ma trận thật còn nguyên các số 0 tại vị trí dropout
còn X̂ thì không — Discriminator sẽ phân biệt được chỉ bằng cách đếm số 0
mà chẳng học gì về tính hợp lý sinh học.

Không dùng decorator. Siêu tham số đọc từ scvagan/config.py lúc khởi tạo.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .. import config
from .attention import Patchify, TransformerStack


class Discriminator(nn.Module):
    """Phân biệt ma trận thật / lai, có điều kiện theo nhãn cụm.

    Parameters
    ----------
    n_genes : số gene G.
    n_clusters : số cụm từ Khối 1. Bằng 0 thì bỏ phần điều kiện.
    n_layers : số khối Transformer (L_disc). None thì lấy config.L_DEC.
    """

    def __init__(self, n_genes, n_clusters=0, n_layers=None):
        super().__init__()
        d = int(config.D_MODEL)
        n_layers = int(config.L_DEC if n_layers is None else n_layers)

        # dùng lại cùng cách đóng gói với Generator để hai bên nhìn dữ liệu
        # theo cùng một cấu trúc token
        self.patchify = Patchify(n_genes)
        P, p = self.patchify.n_patches, self.patchify.patch_size

        self.proj = nn.Linear(2 * p, d)
        self.pos = nn.Parameter(torch.zeros(1, P, d))
        self.cond = nn.Embedding(int(n_clusters), d) if n_clusters > 0 else None

        self.blocks = TransformerStack(n_layers)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(float(config.MODEL_DROPOUT))
        self.fc = nn.Linear(d, 1)

        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x, m, label=None, need_weights=False):
        """(B, G) -> (B, 1) logit, CHƯA qua sigmoid.

        Dùng BCEWithLogitsLoss ở ngoài để ổn định số học.
        """
        h = self.patchify(x, m)                 # (B, P, 2p)
        h = self.proj(h) + self.pos
        if self.cond is not None and label is not None:
            # điều kiện cộng vào mọi token
            h = h + self.cond(label).unsqueeze(1)
        h = self.drop(h)
        h, weights = self.blocks(h, need_weights=need_weights)
        h = self.norm(h).mean(dim=1)            # Pooling (mean) -> (B, d)
        logit = self.fc(h)
        if need_weights:
            return logit, weights
        return logit

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_discriminator(n_genes, n_clusters=0):
    """Tạo Discriminator với siêu tham số hiện hành trong config."""
    return Discriminator(n_genes, n_clusters)
