#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""scVAGAN — CHẠY TOÀN BỘ ĐƯỜNG ỐNG bằng một lệnh.

Đầu vào: ma trận ĐÃ CHUẨN HOÁ (normalize + log transform) của bạn.
Đầu ra:  ma trận đã bù khuyết, ghi ra CSV.

Bốn khối, đúng như sơ đồ:

    KHỐI 1  Xác định dropout   hỗn hợp Gamma-Normal + EM cho từng gene
                               -> xác suất P, mask M (P >= THRESHOLD)
                               K-means chọn k bằng Elbow + Silhouette -> nhãn cụm
    KHỐI 2  Tiền xử lý         BẠN đã làm sẵn, đường ống này không đụng vào
    KHỐI 3  Huấn luyện         Bước 1 train VAE, Bước 2 train GAN (train.py)
    KHỐI 4  Bù khuyết          chỉ điền tại ô M = 1, ghi CSV

Chạy:
    python main.py --x X_log.csv --outdir results

ĐẦU VÀO DUY NHẤT là --x: ma trận đã chuẩn hoá. Không nhận thêm file nào khác.
Mask dropout M do EM tính ra, nhãn cụm do K-means tính ra — cả hai đều sinh
ngay trong lần chạy này, không nạp từ đâu cả.

CSV mặc định HÀNG là gene, CỘT là tế bào; nếu ngược lại thêm --cells-are-rows.

Logic huấn luyện nằm ở train.py, thuật toán Khối 1 nằm ở scvagan/dropout/.
File này chỉ điều phối. Cờ dòng lệnh áp vào config MỘT LẦN lúc khởi chạy.
Không dùng decorator.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch


from src.dropout.detector import prepare_inputs
from src.trainer import trainer
from src import config
from src.dropout.em import fit_em_all_genes
from src.dropout.dropout_prob import (
    compute_dropout_matrix, make_dropout_mask, dropout_summary,
)
from src.clustering.kmeans import cluster_cells, plot_k_selection
from src.data.dataset import (
    load_matrix_csv, build_data, split_train_val, iter_batches, n_batches,
)
from src.model.generator import build_generator
from src.model.discriminator import build_discriminator


# =====================================================================
# GHI FILE
#
# Cả đường ống chỉ đọc MỘT file CSV và ghi ra CSV. Hai hàm dưới đây là toàn
# bộ phần vào/ra — không cần module io riêng.
# =====================================================================

def write_csv(arr_cg, cell_names, gene_names, path):
    """Ghi mảng (C, G) ra CSV đúng hướng và đúng tên hàng/cột của file gốc."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(arr_cg, "detach"):
        arr_cg = arr_cg.detach().numpy()
    arr_cg = np.asarray(arr_cg)

    if bool(config.CSV_GENES_ARE_ROWS):
        body, index, columns = arr_cg.T, gene_names, cell_names
    else:
        body, index, columns = arr_cg, cell_names, gene_names
    pd.DataFrame(body, index=index, columns=columns).to_csv(path, sep=config.CSV_SEP)
    return path


def write_json(obj, path):
    """Ghi dict ra JSON. numpy không tự tuần tự hoá được nên phải ép kiểu."""
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_default),
                    encoding="utf-8")
    return path


# =====================================================================
# KHỐI 1 — XÁC ĐỊNH DROPOUT
# =====================================================================

def detect_dropout(X_proc, cell_names, gene_names, outdir):
    """Xác định dropout bằng hỗn hợp Gamma-Normal + EM, trên ma trận (G, C).

    Giai đoạn A  prepare_inputs        tìm vị trí ô vốn bằng 0, tịnh tiến để
                                       mật độ Gamma không suy biến
    Giai đoạn B  fit_em_all_genes      EM ước lượng hỗn hợp cho TỪNG gene
                                       (song song bằng joblib)
    Giai đoạn C  compute_dropout_matrix  xác suất hậu nghiệm P tại các ô = 0
                 make_dropout_mask     M = 1 khi P >= THRESHOLD

    Returns
    -------
    M : (G, C) mask dropout. info : dict thống kê.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X_em, zero_mask, shift = prepare_inputs(X_proc)
    params_list = fit_em_all_genes(X_em)

    P = compute_dropout_matrix(X_em, zero_mask, params_list)
    M = make_dropout_mask(P)
    info = dropout_summary(M, zero_mask)
    if config.VERBOSE:
        print(f"[K1] Dropout: {info['n_dropout_entries']:,} ô "
              f"({info['dropout_rate_overall']*100:.2f}% toàn ma trận, "
              f"{info['dropout_rate_among_zeros']*100:.2f}% trong số các ô bằng 0)")

    write_csv(P.T, cell_names, gene_names, outdir / "P.csv")
    write_csv(M.T.astype(np.uint8), cell_names, gene_names, outdir / "M.csv")

    info.update({
        "em_shift_applied": float(shift),
        "n_em_converged": int(sum(p.converged for p in params_list)),
        "n_em_invalid": int(sum(not p.valid for p in params_list)),
        "n_label_switched": int(sum(p.valid and not p.dropout_is_low for p in params_list)),
    })
    return M, info


def cluster_by_kmeans(X_proc, cell_names, outdir):
    """Phân cụm tế bào bằng K-means — nguồn DUY NHẤT sinh ra nhãn cụm.

    PCA xuống N_PCS chiều rồi quét k từ K_MIN đến K_MAX, chọn k bằng Elbow
    (điểm gãy) kết hợp Silhouette. Không có đường nào nhận nhãn từ bên ngoài.

    Returns
    -------
    labels : (C,) int. k_best : int. report : dict để vẽ đồ thị chọn k.
    """
    labels, k_best, report = cluster_cells(X_proc)
    plot_k_selection(report, Path(outdir) / "k_selection.png")
    pd.DataFrame({"cell": cell_names, "cluster": labels.astype(int)}).to_csv(
        Path(outdir) / "cluster_labels.csv", index=False, sep=config.CSV_SEP)
    return labels, k_best, report


# =====================================================================
# KHỐI 3 — DỰNG MÔ HÌNH
# =====================================================================

def build_models(data):
    """Tạo Generator và Discriminator khớp kích thước dữ liệu."""
    gen = build_generator(data["n_genes"])
    disc = build_discriminator(data["n_genes"], n_clusters=max(data["n_clusters"], 1))
    return gen, disc


def build_optimizers(gen, disc):
    """Hai optimizer RIÊNG với tốc độ học riêng.

    LR_D nên NHỎ HƠN LR_G. Discriminator có bài toán dễ hơn nhiều nên nếu học
    cùng tốc độ nó sẽ thắng áp đảo trong vài epoch, acc_D leo lên gần 1.0 và
    gradient đối kháng về G bão hoà — GAN đứng im.
    """
    opt_g = torch.optim.Adam(gen.parameters(), lr=float(config.LR_G))
    opt_d = torch.optim.Adam(disc.parameters(), lr=float(config.LR_D))
    return opt_g, opt_d


def print_model_info(gen, disc):
    """In bảng kiến trúc, tiện chép vào luận văn."""
    d = gen.describe()
    print("\nGenerator (VAE + Multi-Head Attention)")
    print(f"  G={d['n_genes']} -> P={d['n_patches']} token x p={d['patch_size']} gene "
          f"(đệm {d['n_pad']})")
    print(f"  d_model={d['d_model']} | heads={d['n_heads']} | d_ff={d['d_ff']} | "
          f"L_enc={d['L_enc']} | L_dec={d['L_dec']} | dz={d['d_latent']}")
    print(f"  tham số: {d['n_parameters']:,} "
          f"(encoder {d['n_parameters_encoder']:,} | latent {d['n_parameters_latent']:,} "
          f"| decoder {d['n_parameters_decoder']:,})")
    print(f"Discriminator: {disc.n_parameters():,} tham số")


# =====================================================================
# KHỐI 4 — BÙ KHUYẾT VÀ GHI CSV
# =====================================================================

def impute_matrix(gen, data):
    """Bù khuyết TOÀN BỘ ma trận.

    Ô quan sát (M = 0) giữ NGUYÊN không đổi một bit; chỉ ô M = 1 bị thay bằng
    giá trị sinh ra — đúng như bước Impute trong sơ đồ.

    IMPUTE_SAMPLES > 1 thì lấy mẫu z nhiều lần rồi trung bình, và trả kèm độ
    lệch chuẩn — chính là ước lượng ĐỘ BẤT ĐỊNH của từng giá trị điền vào.

    Returns
    -------
    x_out : (C, G) tensor. std_out : (C, G) tensor hoặc None.
    """
    n_samples = int(config.IMPUTE_SAMPLES)
    idx_all = torch.arange(data["n_cells"], dtype=torch.long)

    x_out = torch.empty_like(data["X"])
    std_out = torch.empty_like(data["X"]) if n_samples > 1 else None

    for b in iter_batches(data, idx_all, epoch=0, shuffle=False, m_sim=None):
        x_imp, std = gen.impute(b["x"], b["m"], n_samples=n_samples)
        x_out[b["idx"]] = x_imp
        if std_out is not None:
            std_out[b["idx"]] = std

    return x_out, std_out


def save_history(history, path):
    """Ghi lịch sử loss từng epoch ra CSV để vẽ đồ thị hội tụ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(path, index=False)
    return path


def save_checkpoint(gen, disc, path):
    """Lưu trọng số G, D kèm bản chụp toàn bộ tham số của lần chạy."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"generator": gen.state_dict(),
                "discriminator": disc.state_dict(),
                "config": config.as_dict()}, path)
    return path


# =====================================================================
# ĐIỀU PHỐI
# =====================================================================

def apply_cli_to_config(args):
    """Áp cờ dòng lệnh vào config MỘT LẦN, ngay lúc khởi chạy.

    Sau lời gọi này không chỗ nào được sửa config nữa — mọi hàm đọc thẳng từ
    module, nên bộ tham số của cả lần chạy là nhất quán.
    """
    overrides = {
        # Khối 1
        "ZERO_DETECTION": args.zero_detection,
        "EM_SHIFT": args.em_shift,
        "MAX_ITER": args.max_iter,
        "N_RESTARTS": args.n_restarts,
        "DROPOUT_COMPONENT": args.dropout_component,
        "THRESHOLD": args.threshold,
        "K_MIN": args.kmin,
        "K_MAX": args.kmax,
        "N_PCS": args.n_pcs,
        "N_JOBS": args.n_jobs,
        # Khối 3
        "BATCH_SIZE": args.batch_size,
        "VAL_FRACTION": args.val_fraction,
        "SIM_MASK_RATE": args.sim_mask_rate,
        "VAE_EPOCHS": args.vae_epochs,
        "GAN_EPOCHS": args.gan_epochs,
        "LR_G": args.lr_g,
        "LR_D": args.lr_d,
        "KL_WEIGHT": args.kl_weight,
        "ADV_WEIGHT": args.adv_weight,
        # Khối 4
        "IMPUTE_SAMPLES": args.impute_samples,
        # hệ thống
        "RANDOM_STATE": args.seed,
        "CSV_GENES_ARE_ROWS": (not args.cells_are_rows),
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if args.quiet:
        overrides["VERBOSE"] = False
    config.apply(overrides)
    config.validate()
    return overrides


def main(args):
    t0 = time.time()
    apply_cli_to_config(args)
    torch.manual_seed(int(config.RANDOM_STATE))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Nạp ma trận ĐÃ CHUẨN HOÁ -----------------------------------
    print(f"\n########## KHỐI 1: Xác định dropout ##########")
    X_cg, cell_names, gene_names = load_matrix_csv(args.x)   # (C, G)
    X_proc = X_cg.T                                          # Khối 1 dùng (G, C)
    print(f"[K1] Ma trận: {X_proc.shape[0]} gene x {X_proc.shape[1]} tế bào "
          f"| min={X_proc.min():.4g} max={X_proc.max():.4g}")

    # ---- KHỐI 1: dropout, rồi K-means sinh nhãn cụm ------------------
    M, info_k1 = detect_dropout(X_proc, cell_names, gene_names, outdir)
    labels, k_best, report = cluster_by_kmeans(X_proc, cell_names, outdir)
    info_k1.update({"k_best": int(k_best), "k_selection": report})

    # Khối 1 làm việc với (gene, cell); Khối 3 cần (cell, gene) -> chuyển vị
    data = build_data(X_cg, M.T, labels, cell_names, gene_names)
    idx_train, idx_val = split_train_val(data["labels"])
    print(f"[data] train {idx_train.numel()} tế bào ({n_batches(idx_train)} batch) | "
          f"val {idx_val.numel()} tế bào")

    # ---- KHỐI 3: huấn luyện -----------------------------------------
    print(f"\n########## KHỐI 3: Huấn luyện ##########")
    gen, disc = build_models(data)
    opt_g, opt_d = build_optimizers(gen, disc)
    print_model_info(gen, disc)

    history = trainer.train_vae(gen, opt_g, data, idx_train, idx_val)
    history += trainer.train_gan(gen, disc, opt_g, opt_d, data, idx_train, idx_val,
                               epoch_offset=int(config.VAE_EPOCHS))

    # ---- KHỐI 4: bù khuyết ------------------------------------------
    print(f"\n########## KHỐI 4: Bù khuyết ##########")
    x_out, std_out = impute_matrix(gen, data)

    path_x = write_csv(x_out, cell_names, gene_names, outdir / "X_imputed.csv")
    print(f"  Đã điền {int(data['M'].sum()):,} ô dropout, giữ nguyên toàn bộ ô quan sát")
    print(f"  Ghi: {path_x}")

    if std_out is not None:
        path_std = write_csv(std_out, cell_names, gene_names,
                             outdir / "X_imputed_std.csv")
        mean_std = float((std_out * data["M"]).sum()) / max(float(data["M"].sum()), 1.0)
        print(f"  Ghi: {path_std} (độ bất định, trung bình {mean_std:.4f})")

    print(f"  Ghi: {save_history(history, outdir / 'history.csv')}")
    if args.save_model:
        print(f"  Ghi: {save_checkpoint(gen, disc, outdir / 'scvagan_model.pt')}")

    # ---- tóm tắt cả lần chạy ----------------------------------------
    summary = dict(info_k1)
    summary.update({
        "n_genes": data["n_genes"],
        "n_cells": data["n_cells"],
        "n_clusters": data["n_clusters"],
        "final": history[-1] if history else {},
        "elapsed_sec": round(time.time() - t0, 2),
        "params": config.as_dict(),
    })
    write_json(summary, outdir / "summary.json")
    print(f"  Ghi: {outdir / 'summary.json'}")
    print(f"\nXong sau {summary['elapsed_sec']}s.")
    return summary


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="scVAGAN — chạy toàn bộ: xác định dropout, K-means, "
                    "huấn luyện VAE + GAN, bù khuyết ra CSV.")

    # --- đầu vào / đầu ra ---------------------------------------------
    ap.add_argument("--x", required=True,
                    help="File CSV ma trận ĐÃ CHUẨN HOÁ — đầu vào DUY NHẤT")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--cells-are-rows", action="store_true",
                    help="File có HÀNG là tế bào (mặc định: hàng là gene)")

    # --- Khối 1 --------------------------------------------------------
    g1 = ap.add_argument_group("Khối 1 — xác định dropout")
    # "from_file" bị loại: nó cần một file mask bên ngoài, mà đường ống này
    # chỉ nhận đúng một file đầu vào là --x.
    g1.add_argument("--zero-detection", choices=("exact_zero", "min_value"),
                    default=None,
                    help=f"exact_zero khi bạn dùng log1p; min_value khi dùng "
                         f"log(x + c) với c > 1. Mặc định {config.ZERO_DETECTION}")
    g1.add_argument("--em-shift", type=float, default=None)
    g1.add_argument("--max-iter", type=int, default=None)
    g1.add_argument("--n-restarts", type=int, default=None)
    g1.add_argument("--dropout-component", choices=list(config.VALID_DROPOUT_COMPONENT),
                    default=None)
    g1.add_argument("--threshold", type=float, default=None,
                    help=f"ngưỡng P để gắn cờ dropout, mặc định {config.THRESHOLD}")
    g1.add_argument("--kmin", type=int, default=None)
    g1.add_argument("--kmax", type=int, default=None)
    g1.add_argument("--n-pcs", type=int, default=None)
    g1.add_argument("--n-jobs", type=int, default=None)

    # --- Khối 3 --------------------------------------------------------
    g3 = ap.add_argument_group("Khối 3 — huấn luyện")
    g3.add_argument("--batch-size", type=int, default=None)
    g3.add_argument("--val-fraction", type=float, default=None)
    g3.add_argument("--sim-mask-rate", type=float, default=None,
                    help="Tỉ lệ ô quan sát bị che nhân tạo để đo RMSE thật")
    g3.add_argument("--vae-epochs", type=int, default=None, help="Bước 1")
    g3.add_argument("--gan-epochs", type=int, default=None, help="Bước 2")
    g3.add_argument("--lr-g", type=float, default=None)
    g3.add_argument("--lr-d", type=float, default=None,
                    help="Nên NHỎ HƠN --lr-g để D không thắng áp đảo")
    g3.add_argument("--kl-weight", type=float, default=None)
    g3.add_argument("--adv-weight", type=float, default=None)

    # --- Khối 4 và hệ thống -------------------------------------------
    g4 = ap.add_argument_group("Khối 4 và hệ thống")
    g4.add_argument("--impute-samples", type=int, default=None,
                    help="1 = z=mu; >1 = lấy mẫu nhiều lần để ước lượng độ bất định")
    g4.add_argument("--save-model", action="store_true")
    g4.add_argument("--seed", type=int, default=None)
    g4.add_argument("--quiet", action="store_true")

    return ap.parse_args(argv)


if __name__ == "__main__":
    sys.exit(0 if main(parse_args()) else 1)