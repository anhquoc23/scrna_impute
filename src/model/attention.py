"""Multi-Head Attention và việc đóng gói gene thành token.

Attention cần một CHUỖI token. Mỗi mẫu là một tế bào với G gene, nên phải
chia G gene thành P nhóm — mỗi nhóm là một token gồm p gene. Đây chính là
ý "Encoder đóng gói dữ liệu trước Attention" trong sơ đồ.

Nếu G không chia hết cho P thì đệm thêm ở cuối, và cắt bỏ khi ghép lại.

Cả Generator lẫn Discriminator đều dùng lại các lớp trong file này, nên
hai bên nhìn dữ liệu theo cùng một cấu trúc token.

Không dùng decorator. Siêu tham số đọc từ scvagan/config.py lúc khởi tạo.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .. import config


# =====================================================================
# ĐÓNG GÓI GENE THÀNH TOKEN
# =====================================================================

def patch_geometry(n_genes, n_patches=None):
    """Tính kích thước token và lượng đệm.

    Returns
    -------
    (n_patches, patch_size, n_pad)
    """
    P = int(config.N_PATCHES if n_patches is None else n_patches)
    if P < 2:
        raise ValueError(
            "N_PATCHES phải >= 2. Nếu chỉ có 1 token thì self-attention chỉ "
            "tự nhìn chính nó, hoàn toàn vô nghĩa."
        )
    p = int(math.ceil(int(n_genes) / P))
    return P, p, P * p - int(n_genes)


class Patchify(nn.Module):
    """(B, G) giá trị + (B, G) mask  ->  (B, P, 2p).

    Mỗi token mang CẢ giá trị biểu hiện lẫn bit mask của p gene, nên mô hình
    biết ô nào là dropout mà không cần kênh thông tin riêng.
    """

    def __init__(self, n_genes, n_patches=None):
        super().__init__()
        self.n_genes = int(n_genes)
        self.n_patches, self.patch_size, self.n_pad = patch_geometry(n_genes, n_patches)

    def forward(self, x_masked, m):
        B = x_masked.shape[0]
        if self.n_pad > 0:
            x_masked = torch.cat([x_masked, x_masked.new_zeros(B, self.n_pad)], dim=1)
            # phần đệm đánh dấu là dropout để mô hình bỏ qua
            m = torch.cat([m, m.new_ones(B, self.n_pad)], dim=1)
        xp = x_masked.view(B, self.n_patches, self.patch_size)
        mp = m.view(B, self.n_patches, self.patch_size)
        return torch.cat([xp, mp], dim=2)

    def extra_repr(self):
        return (f"n_genes={self.n_genes}, n_patches={self.n_patches}, "
                f"patch_size={self.patch_size}, n_pad={self.n_pad}")


class Unpatchify(nn.Module):
    """(B, P, p) -> (B, G), cắt bỏ phần đệm."""

    def __init__(self, n_genes, n_patches=None):
        super().__init__()
        self.n_genes = int(n_genes)
        self.n_patches, self.patch_size, self.n_pad = patch_geometry(n_genes, n_patches)

    def forward(self, h):
        out = h.reshape(h.shape[0], self.n_patches * self.patch_size)
        return out[:, : self.n_genes] if self.n_pad > 0 else out

    def extra_repr(self):
        return f"n_genes={self.n_genes}, n_patches={self.n_patches}"


# =====================================================================
# MULTI-HEAD ATTENTION
# =====================================================================

class FeedForward(nn.Module):
    """Mạng truyền thẳng hai lớp: d → d_ff → d."""

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, h):
        return self.net(h)


class TransformerBlock(nn.Module):
    """Multi-Head Self-Attention + Feed-Forward, mỗi nhánh có Add & Norm.

    Theo sơ đồ, LayerNorm đặt SAU phép cộng (post-norm, kiểu Transformer
    gốc). Đặt config.PRE_NORM = True để chuyển sang pre-norm — ổn định hơn
    khi xếp nhiều lớp, nhưng khác với hình vẽ.
    """

    def __init__(self, d_model=None, n_heads=None, d_ff=None,
                 dropout=None, pre_norm=None):
        super().__init__()
        d_model = int(config.D_MODEL if d_model is None else d_model)
        n_heads = int(config.N_HEADS if n_heads is None else n_heads)
        d_ff = int(config.D_FF if d_ff is None else d_ff)
        dropout = float(config.MODEL_DROPOUT if dropout is None else dropout)
        self.pre_norm = bool(config.PRE_NORM if pre_norm is None else pre_norm)

        if d_model % n_heads != 0:
            raise ValueError(f"D_MODEL ({d_model}) phải chia hết cho N_HEADS ({n_heads}).")

        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, need_weights=False):
        """h : (B, P, d) -> (h_mới, ma trận attention hoặc None)."""
        if self.pre_norm:
            hn = self.norm1(h)
            a, w = self.attn(hn, hn, hn, need_weights=need_weights,
                             average_attn_weights=True)
            h = h + self.drop(a)
            h = h + self.drop(self.ff(self.norm2(h)))
        else:
            a, w = self.attn(h, h, h, need_weights=need_weights,
                             average_attn_weights=True)
            h = self.norm1(h + self.drop(a))            # Add & LayerNorm
            h = self.norm2(h + self.drop(self.ff(h)))   # Add & LayerNorm
        return h, w


class TransformerStack(nn.Module):
    """Xếp chồng ``n_layers`` khối Transformer — ứng với × L_enc / × L_dec."""

    def __init__(self, n_layers, d_model=None, n_heads=None, d_ff=None,
                 dropout=None, pre_norm=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, pre_norm)
            for _ in range(int(n_layers))
        ])

    def forward(self, h, need_weights=False):
        """Trả (h, danh sách ma trận attention của từng lớp)."""
        weights = []
        for blk in self.blocks:
            h, w = blk(h, need_weights=need_weights)
            if need_weights:
                weights.append(w)
        return h, weights

    def __len__(self):
        return len(self.blocks)
