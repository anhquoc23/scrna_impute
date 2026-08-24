#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""scVAGAN — CÁC HÀM HUẤN LUYỆN.

File này CHỈ chứa logic huấn luyện. Không có CLI, không đọc file, không ghi
file. Phần điều phối (nạp CSV, dựng mô hình, impute, ghi kết quả) nằm ở
main.py.

Bố cục
------
    1. HÀM LOSS      masked_mse, reconstruction_loss, make_hybrid, kl_anneal_weight
    2. BƯỚC 1 — VAE  vae_step, vae_epoch, train_vae
    3. BƯỚC 2 — GAN  disc_step, gen_step, gan_epoch, train_gan
    4. ĐÁNH GIÁ      evaluate

Mỗi bước có BA MỨC, dùng được độc lập:

    *_step   một batch  — muốn tự viết vòng lặp thì gọi mức này
    *_epoch  một epoch
    train_*  cả giai đoạn, trả về history để vẽ đồ thị hội tụ

Mọi hàm chỉ nhận DỮ LIỆU; siêu tham số đọc thẳng từ scvagan/config.py tại
thời điểm gọi. Không dùng decorator.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn

from src import config
from src.data.dataset import simulate_mask, iter_batches
from src.model.vae import kl_divergence


# =====================================================================
# 1. HÀM LOSS
# =====================================================================

def masked_mse(pred, target, mask):
    """MSE chỉ tính trên các ô mà mask = 1.
 
    Chia cho SỐ Ô ĐƯỢC TÍNH, không phải tổng số ô — nếu chia cho tổng thì giá
    trị loss sẽ phụ thuộc tỉ lệ dropout của từng batch, không so sánh được
    giữa các batch. Trả 0 khi mask rỗng nhờ clamp mẫu số về tối thiểu 1.
    """
    denom = mask.sum().clamp(min=1.0)
    return ((pred - target) ** 2 * mask).sum() / denom
 
 
def reconstruction_loss(x_gen, x, m, m_sim):
    """Loss tái tạo, gồm hai phần.
 
    Phần 1 — ô QUAN SÁT (m = 0 và m_sim = 0): nơi duy nhất có đáp án thật, nên
    là tín hiệu học chính. KHÔNG thể tính tại m = 1 vì ở đó dữ liệu gốc chỉ là
    số 0 do lỗi kỹ thuật; lấy nó làm đích sẽ dạy mô hình điền số 0.
 
    Phần 2 — ô CHE NHÂN TẠO (m_sim = 1): ta cố tình giấu những ô vốn quan sát
    được, nên vẫn giữ đáp án. Đây là phần dạy mô hình ĐIỀN giá trị vào chỗ
    trống, sát nhiệm vụ thật hơn hẳn phần 1. Chỉ có khi SIM_MASK_RATE > 0.
    """
    observed = (1.0 - m) * (1.0 - m_sim)
    loss = masked_mse(x_gen, x, observed)
    if m_sim.sum() > 0:
        loss = loss + masked_mse(x_gen, x, m_sim)
    return loss
 
 
def make_hybrid(x, m, x_gen):
    """Ma trận LAI đưa vào Discriminator: X ⊙ (1−M) + X̂ ⊙ M.
 
    QUAN TRỌNG: KHÔNG đưa x_gen thô vào D. Ma trận thật còn nguyên các số 0
    tại vị trí dropout, còn x_gen thì không — D sẽ phân biệt được chỉ bằng
    cách đếm số 0 mà chẳng học gì về tính hợp lý sinh học.
    """
    return x * (1.0 - m) + x_gen * m
 
 
def kl_anneal_weight(epoch, n_epochs):
    """Trọng số KL tăng tuyến tính trong NỬA ĐẦU quá trình rồi giữ nguyên.
 
    Chống posterior collapse: nếu áp đủ trọng số KL ngay từ epoch đầu, khi
    Decoder còn kém thì cách rẻ nhất để giảm loss là đẩy q(z|x) về đúng prior
    — latent không mang thông tin gì nữa và thường không hồi phục.
    """
    full = float(config.KL_WEIGHT)
    warmup = max(n_epochs * 0.5, 1.0)
    return full * min(1.0, (epoch + 1) / warmup)
 
 
# =====================================================================
# 2. BƯỚC 1 — HUẤN LUYỆN VAE
#
# Chỉ Generator học, Discriminator chưa tham gia. Mục tiêu là đưa VAE tới chỗ
# tái tạo được dữ liệu trước. Nhảy thẳng vào GAN khi Generator còn sinh nhiễu
# thì D thắng ngay từ đầu và G không học được gì.
# =====================================================================
 
def vae_step(gen, opt_g, batch, kl_weight):
    """Một BƯỚC cập nhật VAE trên một batch.
 
    Returns
    -------
    dict: rec, kl, loss (float) và bs (kích thước batch).
    """
    x, m, xm, m_sim = batch["x"], batch["m"], batch["x_masked"], batch["m_sim"]
 
    opt_g.zero_grad()
    x_gen, mu, logvar = gen(xm, m)
    loss_rec = reconstruction_loss(x_gen, x, m, m_sim)
    loss_kl = kl_divergence(mu, logvar)
    loss = loss_rec + kl_weight * loss_kl
    loss.backward()
    opt_g.step()
 
    return {
        "rec": loss_rec.detach().item(),
        "kl": loss_kl.detach().item(),
        "loss": loss.detach().item(),
        "bs": int(x.shape[0]),
    }
 
 
def vae_epoch(gen, opt_g, data, idx, epoch, m_sim, kl_weight):
    """Một EPOCH huấn luyện VAE. Trả trung bình có trọng số theo kích thước batch."""
    gen.train()
    tot = {"rec": 0.0, "kl": 0.0, "loss": 0.0}
    n = 0
    for b in iter_batches(data, idx, epoch=epoch, m_sim=m_sim):
        out = vae_step(gen, opt_g, b, kl_weight)
        for k in tot:
            tot[k] += out[k] * out["bs"]
        n += out["bs"]
    return {k: v / max(n, 1) for k, v in tot.items()}
 
 
def train_vae(gen, opt_g, data, idx_train, idx_val, n_epochs=None, verbose=None):
    """BƯỚC 1 ĐẦY ĐỦ — huấn luyện VAE qua nhiều epoch.
 
    Parameters
    ----------
    gen : Generator.
    opt_g : optimizer của Generator.
    data : dict từ load_training_data.
    idx_train, idx_val : tensor chỉ số tế bào.
    n_epochs : None thì lấy config.VAE_EPOCHS.
    verbose : None thì lấy config.VERBOSE.
 
    Returns
    -------
    history : list[dict], mỗi phần tử là một epoch.
    """
    n_epochs = int(config.VAE_EPOCHS if n_epochs is None else n_epochs)
    verbose = bool(config.VERBOSE if verbose is None else verbose)
    history = []
 
    if verbose:
        print(f"\n=== Bước 1: Train VAE ({n_epochs} epoch) ===")
 
    for ep in range(n_epochs):
        # mẫu che nhân tạo đổi theo epoch -> mô hình không học thuộc một mẫu che
        m_sim = simulate_mask(data["M"], ep)
        kl_w = kl_anneal_weight(ep, n_epochs)
 
        t = time.perf_counter()
        out = vae_epoch(gen, opt_g, data, idx_train, ep, m_sim, kl_w)
        out.update({"phase": "vae", "epoch": ep + 1, "kl_weight": kl_w,
                    "seconds": time.perf_counter() - t})
        history.append(out)
 
        if verbose:
            print(f"  epoch {ep+1}/{n_epochs} | rec {out['rec']:.4f} | "
                  f"kl {out['kl']:.4f} | kl_w {kl_w:.3f} | {out['seconds']:.1f}s")
 
    ev = evaluate(gen, data, idx_val)
    if ev:
        if verbose:
            print("  [val] " + " | ".join(f"{k} {v:.4f}" for k, v in ev.items()))
        if history:
            history[-1].update({f"val_{k}": v for k, v in ev.items()})
    return history
 
 
# =====================================================================
# 3. BƯỚC 2 — HUẤN LUYỆN GAN
#
# Luân phiên trên TỪNG batch: cập nhật D một lần rồi G một lần. Thứ tự D
# trước là quy ước chuẩn — G cần một D đã nhìn qua batch hiện tại thì tín
# hiệu đối kháng mới có nghĩa.
# =====================================================================
 
def disc_step(gen, disc, opt_d, batch, bce):
    """Một BƯỚC cập nhật Discriminator. Generator bị ĐÓNG BĂNG.
 
    x_gen sinh trong no_grad nên đồ thị tính toán của G không bị dựng lên —
    tiết kiệm bộ nhớ và chắc chắn không gradient nào rò sang G.
    """
    x, m, xm, label = batch["x"], batch["m"], batch["x_masked"], batch["label"]
    bs = x.shape[0]
 
    opt_d.zero_grad()
    with torch.no_grad():
        x_gen, _, _ = gen(xm, m)
    x_fake = make_hybrid(x, m, x_gen)
 
    logit_real = disc(x, m, label)
    logit_fake = disc(x_fake, m, label)
    # ones_like/zeros_like -> nhãn tự nằm cùng thiết bị với logit, không
    # phải truyền device xuống tận đây.
    loss_d = (bce(logit_real, torch.ones_like(logit_real))
              + bce(logit_fake, torch.zeros_like(logit_fake)))
    loss_d.backward()
    opt_d.step()
 
    # acc_D là chỉ báo quan trọng nhất khi gỡ lỗi GAN. ~0.5 là cân bằng lành
    # mạnh; > 0.95 nghĩa là D thắng áp đảo, gradient đối kháng về G bão hoà
    # và G không còn học được gì từ D nữa.
    acc = 0.5 * ((logit_real > 0).float().mean() + (logit_fake <= 0).float().mean())
    return {"loss_d": loss_d.detach().item(), "acc_d": float(acc), "bs": int(bs)}
 
 
def gen_step(gen, disc, opt_g, batch, bce):
    """Một BƯỚC cập nhật Generator. Discriminator bị ĐÓNG BĂNG.
 
    Loss của G gồm ba phần:
        rec  giữ giá trị sinh ra bám dữ liệu thật              (trọng số 1)
        adv  đánh lừa D — kéo phân phối giá trị điền về giống thật (ADV_WEIGHT)
        kl   ràng buộc latent về prior                          (KL_WEIGHT)
 
    Dùng non-saturating loss: tối thiểu BCE(D(fake), 1) thay vì tối đa
    −BCE(D(fake), 0). Hai cách cùng điểm tối ưu nhưng cách sau có gradient
    gần 0 đúng lúc G đang yếu, tức đúng lúc cần gradient nhất.
    """
    x, m, xm, m_sim, label = (batch["x"], batch["m"], batch["x_masked"],
                              batch["m_sim"], batch["label"])
    bs = x.shape[0]
    adv_weight = float(config.ADV_WEIGHT)
    kl_weight = float(config.KL_WEIGHT)
 
    opt_g.zero_grad()
    x_gen, mu, logvar = gen(xm, m)
    x_fake = make_hybrid(x, m, x_gen)
 
    loss_rec = reconstruction_loss(x_gen, x, m, m_sim)
    logit_fake = disc(x_fake, m, label)
    loss_adv = bce(logit_fake, torch.ones_like(logit_fake))
    loss_kl = kl_divergence(mu, logvar)
 
    loss_g = loss_rec + adv_weight * loss_adv + kl_weight * loss_kl
    loss_g.backward()
    opt_g.step()
 
    return {
        "loss_g": loss_g.detach().item(),
        "rec": loss_rec.detach().item(),
        "adv": loss_adv.detach().item(),
        "kl": loss_kl.detach().item(),
        "bs": int(bs),
    }
 
 
def gan_epoch(gen, disc, opt_g, opt_d, data, idx, epoch, m_sim):
    """Một EPOCH đối kháng: mỗi batch chạy disc_step rồi gen_step."""
    gen.train()
    disc.train()
    bce = nn.BCEWithLogitsLoss()
    tot = {"loss_d": 0.0, "acc_d": 0.0, "loss_g": 0.0, "rec": 0.0, "adv": 0.0, "kl": 0.0}
    n = 0
 
    for b in iter_batches(data, idx, epoch=epoch, m_sim=m_sim):
        d_out = disc_step(gen, disc, opt_d, b, bce)
        g_out = gen_step(gen, disc, opt_g, b, bce)
        bs = d_out["bs"]
        for k in ("loss_d", "acc_d"):
            tot[k] += d_out[k] * bs
        for k in ("loss_g", "rec", "adv", "kl"):
            tot[k] += g_out[k] * bs
        n += bs
 
    return {k: v / max(n, 1) for k, v in tot.items()}
 
 
def train_gan(gen, disc, opt_g, opt_d, data, idx_train, idx_val,
              n_epochs=None, epoch_offset=0, verbose=None):
    """BƯỚC 2 ĐẦY ĐỦ — huấn luyện đối kháng qua nhiều epoch.
 
    Parameters
    ----------
    epoch_offset : số epoch đã chạy ở Bước 1. Cộng vào để mẫu che nhân tạo và
        thứ tự xáo trộn KHÔNG lặp lại y hệt Bước 1.
 
    Returns
    -------
    history : list[dict], mỗi phần tử là một epoch.
    """
    n_epochs = int(config.GAN_EPOCHS if n_epochs is None else n_epochs)
    verbose = bool(config.VERBOSE if verbose is None else verbose)
    history = []
 
    if verbose:
        print(f"\n=== Bước 2: Train GAN ({n_epochs} epoch) ===")
 
    for ep in range(n_epochs):
        e = int(epoch_offset) + ep
        m_sim = simulate_mask(data["M"], e)
 
        t = time.perf_counter()
        out = gan_epoch(gen, disc, opt_g, opt_d, data, idx_train, e, m_sim)
        out.update({"phase": "gan", "epoch": ep + 1,
                    "seconds": time.perf_counter() - t})
        history.append(out)
 
        if verbose:
            flag = "  <-- D thắng áp đảo, hãy giảm LR_D" if out["acc_d"] > 0.95 else ""
            print(f"  epoch {ep+1}/{n_epochs} | loss_D {out['loss_d']:.4f} | "
                  f"loss_G {out['loss_g']:.4f} | rec {out['rec']:.4f} | "
                  f"adv {out['adv']:.4f} | acc_D {out['acc_d']:.3f} | "
                  f"{out['seconds']:.1f}s{flag}")
 
    ev = evaluate(gen, data, idx_val)
    if ev:
        if verbose:
            print("  [val] " + " | ".join(f"{k} {v:.4f}" for k, v in ev.items()))
        if history:
            history[-1].update({f"val_{k}": v for k, v in ev.items()})
    return history
 
 
# =====================================================================
# 4. ĐÁNH GIÁ
# =====================================================================
 
def evaluate(gen, data, idx, m_sim=None):
    """Đánh giá trên tập kiểm định. Trả None nếu tập rỗng.
 
    rmse_observed   sai số tại ô quan sát — chỉ đo khả năng TÁI TẠO, luôn lạc
                    quan vì mô hình đã nhìn thấy chính những giá trị đó.
    rmse_sim_masked sai số tại ô che nhân tạo — mô hình KHÔNG nhìn thấy giá
                    trị nhưng ta biết đáp án. Đây mới là con số đo chất lượng
                    bù khuyết thật sự; chỉ có khi SIM_MASK_RATE > 0.
 
    Không có cách nào đo trực tiếp sai số tại ô dropout THẬT vì ở đó không tồn
    tại đáp án — đó chính là lý do phải che nhân tạo.
    """
    if idx.numel() == 0:
        return None
    if m_sim is None:
        m_sim = simulate_mask(data["M"], 0)
 
    gen.eval()
    se_obs = n_obs = se_sim = n_sim = 0.0
    with torch.no_grad():
        for b in iter_batches(data, idx, epoch=0, shuffle=False, m_sim=m_sim):
            x, m, xm, ms = b["x"], b["m"], b["x_masked"], b["m_sim"]
            x_gen, _, _ = gen(xm, m, sample=False)      # z = mu, tiền định
            obs = (1.0 - m) * (1.0 - ms)
            se_obs += float((((x_gen - x) ** 2) * obs).sum()); n_obs += float(obs.sum())
            se_sim += float((((x_gen - x) ** 2) * ms).sum()); n_sim += float(ms.sum())
 
    out = {"rmse_observed": (se_obs / max(n_obs, 1)) ** 0.5}
    if n_sim > 0:
        out["rmse_sim_masked"] = (se_sim / n_sim) ** 0.5
    return out