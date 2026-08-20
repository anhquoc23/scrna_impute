"""Ma trận xác suất dropout, ngưỡng quyết định và mask M.

Tham số điều chỉnh lấy mặc định từ các biến tĩnh trong scvagan/config.py.
"""
from __future__ import annotations

import numpy as np

from ..config import (
    DROPOUT_COMPONENT,
    EPS,
    REQUIRE_DROPOUT_IS_LOW,
    THRESHOLD,
    VERBOSE,
)
from .em import e_step


def dropout_prob_one_gene(x, p, dropout_component=DROPOUT_COMPONENT, eps=EPS):
    """Xác suất hậu nghiệm thuộc thành phần dropout, cho MỘT gene.

    Parameters
    ----------
    x : (C,) float64 — vector biểu hiện của gene.
    p : GammaNormalParams — tham số EM của gene đó.

    Returns
    -------
    (C,) float32 — giá trị trong [0, 1]. Trả toàn 0 nếu gene không hợp lệ.
    """
    if not p.valid:
        return np.zeros(np.asarray(x).size, dtype=np.float32)

    resp = e_step(np.asarray(x, dtype=np.float64), p, eps)   # (C, 2)
    col = 1 if dropout_component == "normal" else 0
    return resp[:, col].astype(np.float32)


def compute_dropout_matrix(
    X_em,
    zero_mask,
    params_list,
    dropout_component=DROPOUT_COMPONENT,
    eps=EPS,
    require_dropout_is_low=REQUIRE_DROPOUT_IS_LOW,
    verbose=VERBOSE,
):
    """Ma trận xác suất dropout P.

    QUAN TRỌNG: xác suất CHỈ được tính tại các ô vốn bằng 0, mọi ô khác
    gán 0.0. Vị trí các ô đó lấy từ ``zero_mask`` — do hàm
    ``inputs.detect_zero_mask`` sinh ra hoặc do bạn cung cấp sẵn — chứ không
    suy ra từ ``X_em``, vì sau tiền xử lý số 0 có thể đã thành giá trị khác.

    Parameters
    ----------
    X_em        : (G, C) float32 — ma trận đã tiền xử lý và đảm bảo dương.
    zero_mask   : (G, C) bool — True tại các ô vốn bằng 0.
    params_list : danh sách GammaNormalParams dài G.

    Returns
    -------
    P : (G, C) float32
    """
    X_em = np.asarray(X_em)
    zero_mask = np.asarray(zero_mask).astype(bool)
    if X_em.shape != zero_mask.shape:
        raise ValueError(
            f"X_em {X_em.shape} và zero_mask {zero_mask.shape} phải cùng shape."
        )
    if len(params_list) != X_em.shape[0]:
        raise ValueError(
            f"params_list dài {len(params_list)} nhưng có {X_em.shape[0]} gene."
        )

    n_genes, n_cells = X_em.shape
    P = np.zeros((n_genes, n_cells), dtype=np.float32)

    if verbose:
        print(f"[P] Tính xác suất dropout tại {int(zero_mask.sum()):,} ô bằng 0 ...")

    for g in range(n_genes):
        p = params_list[g]
        if not p.valid:
            continue
        if require_dropout_is_low and not p.dropout_is_low:
            continue                      # gene bị đảo nhãn => bỏ qua, không impute
        zc = zero_mask[g, :]
        if not np.any(zc):
            continue
        probs = dropout_prob_one_gene(
            X_em[g, :], p, dropout_component=dropout_component, eps=eps
        )
        P[g, zc] = probs[zc]

    return P


def make_dropout_mask(P, threshold=THRESHOLD):
    """Nhị phân hoá ma trận xác suất thành mask.

    Parameters
    ----------
    P : (G, C) float32
    threshold : ngưỡng quyết định, mặc định lấy từ config.THRESHOLD.

    Returns
    -------
    M : (G, C) uint8 — 1 nếu P >= threshold, ngược lại 0.
    """
    return (np.asarray(P) >= float(threshold)).astype(np.uint8)


def dropout_summary(M, zero_mask):
    """Thống kê mô tả về mask dropout (phục vụ báo cáo / luận văn)."""
    M = np.asarray(M).astype(bool)
    zero_mask = np.asarray(zero_mask).astype(bool)

    n_genes, n_cells = M.shape
    n_total = int(M.size)
    n_zero = int(zero_mask.sum())
    n_drop = int(M.sum())

    per_gene = M.sum(axis=1)
    per_cell = M.sum(axis=0)

    return {
        "n_genes": n_genes,
        "n_cells": n_cells,
        "n_entries": n_total,
        "n_zero_entries": n_zero,
        "zero_rate": float(n_zero / n_total) if n_total else 0.0,
        "n_dropout_entries": n_drop,
        "dropout_rate_overall": float(n_drop / n_total) if n_total else 0.0,
        "dropout_rate_among_zeros": float(n_drop / n_zero) if n_zero else 0.0,
        "dropout_per_gene_mean": float(per_gene.mean()),
        "dropout_per_gene_median": float(np.median(per_gene)),
        "dropout_per_cell_mean": float(per_cell.mean()),
        "dropout_per_cell_median": float(np.median(per_cell)),
        "n_genes_without_dropout": int((per_gene == 0).sum()),
        "n_cells_without_dropout": int((per_cell == 0).sum()),
    }
