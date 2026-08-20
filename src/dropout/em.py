"""Thuật toán Kỳ vọng - Cực đại (EM) cho hỗn hợp Gamma-Normal.

Tham số điều chỉnh lấy mặc định từ các biến tĩnh trong scvagan/config.py,
và có thể truyền thẳng vào hàm để ghi đè cho từng lần gọi.
"""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from scipy.special import digamma, polygamma

from ..config import (
    DROPOUT_COMPONENT,
    EPS,
    MAX_ITER,
    N_JOBS,
    N_RESTARTS,
    RANDOM_STATE,
    TOL,
    VERBOSE,
)
from .gamma_normal import (
    GammaNormalParams,
    init_params,
    log_component_matrix,
    log_likelihood,
)


# ------------------------------------------------------------- E-STEP ----


def e_step(x, p, eps=EPS):
    """Bước E: tính trách nhiệm hậu nghiệm của hai thành phần.

    Parameters
    ----------
    x : (n,) float64
    p : GammaNormalParams — tham số hiện tại.
    eps : hằng số chống chia 0.

    Returns
    -------
    resp : (n, 2) float64 — cột 0 = Gamma, cột 1 = Normal.
           Mỗi hàng có tổng bằng 1.
    """
    logc = log_component_matrix(x, p, eps)                  # (n, 2)
    m = np.max(logc, axis=1, keepdims=True)
    e = np.exp(logc - m)
    s = np.sum(e, axis=1, keepdims=True)
    resp = e / np.maximum(s, eps)
    # hàng nào bị NaN (cả hai thành phần cùng underflow) thì chia đều
    bad = ~np.isfinite(resp).all(axis=1)
    if np.any(bad):
        resp[bad, :] = 0.5
    return resp


# ------------------------------------------------------------- M-STEP ----


def _update_gamma_mle(x, w, eps=EPS, n_newton=20):
    """Ước lượng hợp lý cực đại CÓ TRỌNG SỐ cho Gamma.

    Giải phương trình  log(k) - digamma(k) = s  bằng Newton, với
    s = log(mean_w(x)) - mean_w(log x). Sau đó theta = mean_w(x) / k.

    Parameters
    ----------
    x : (n,) float64 — dữ liệu (đã đảm bảo > 0).
    w : (n,) float64 — trọng số, chính là resp[:, 0].

    Returns
    -------
    (k, theta)
    """
    W = float(np.sum(w)) + eps
    xs = np.maximum(x, eps)
    xbar = float(np.sum(w * xs) / W)
    mean_log = float(np.sum(w * np.log(xs)) / W)
    s = np.log(max(xbar, eps)) - mean_log
    s = float(np.clip(s, eps, 1e3))

    # khởi tạo theo xấp xỉ Minka
    k = (3.0 - s + np.sqrt((s - 3.0) ** 2 + 24.0 * s)) / (12.0 * s)
    k = float(np.clip(k, 1e-3, 1e4))

    for _ in range(n_newton):
        num = np.log(k) - digamma(k) - s
        den = 1.0 / k - polygamma(1, k)
        if not np.isfinite(den) or abs(den) < eps:
            break
        k_new = k - num / den
        if not np.isfinite(k_new) or k_new <= 0.0:
            k_new = k / 2.0
        if abs(k_new - k) < 1e-8:
            k = k_new
            break
        k = k_new

    k = float(np.clip(k, 1e-3, 1e4))
    theta = float(np.clip(xbar / k, eps, 1e4))
    return k, theta


def _update_normal_mle(x, w, eps=EPS):
    """Ước lượng hợp lý cực đại CÓ TRỌNG SỐ cho Normal.

    Returns (mu, sigma).
    """
    W = float(np.sum(w)) + eps
    mu = float(np.sum(w * x) / W)
    var = float(np.sum(w * (x - mu) ** 2) / W)
    sigma = float(np.sqrt(max(var, 1e-6)))
    return mu, sigma


def m_step(x, resp, eps=EPS):
    """Bước M: cập nhật {pi, k, theta, mu, sigma} từ trách nhiệm.

    Parameters
    ----------
    x : (n,) float64
    resp : (n, 2) float64

    Returns
    -------
    GammaNormalParams — tham số mới.
    """
    w_gamma = resp[:, 0]
    w_normal = resp[:, 1]
    pi = float(np.clip(np.mean(w_gamma), 1e-4, 1.0 - 1e-4))
    k, theta = _update_gamma_mle(x, w_gamma, eps)
    mu, sigma = _update_normal_mle(x, w_normal, eps)
    return GammaNormalParams(pi=pi, k=k, theta=theta, mu=mu, sigma=sigma)


# --------------------------------------------------------------- FIT -----


def dropout_is_low(p, dropout_component=DROPOUT_COMPONENT):
    """Thành phần dropout có kỳ vọng thấp hơn thành phần kia hay không.

    Dropout tạo ra các giá trị gần 0, nên bình thường điều này phải đúng.
    Nếu sai thì EM đã đảo nhãn hai thành phần trên gene đó.
    """
    if not p.valid:
        return False
    if dropout_component == "normal":
        return p.normal_mean() <= p.gamma_mean()
    return p.gamma_mean() <= p.normal_mean()


def fit_em_one_gene(
    x,
    max_iter=MAX_ITER,
    tol=TOL,
    n_restarts=N_RESTARTS,
    eps=EPS,
    dropout_component=DROPOUT_COMPONENT,
    random_state=RANDOM_STATE,
):
    """Chạy EM cho MỘT gene, có nhiều lần khởi tạo lại.

    Parameters
    ----------
    x : (C,) — vector biểu hiện đã tiền xử lý của gene, mọi giá trị > 0.
    max_iter, tol, n_restarts, eps, dropout_component, random_state :
        mặc định lấy từ scvagan/config.py.

    Returns
    -------
    GammaNormalParams — nghiệm có log-likelihood cao nhất.
        ``valid=False`` nếu gene suy biến (gần như hằng số) — khi đó gene
        này sẽ không được impute.
    """
    x = np.asarray(x, dtype=np.float64).ravel()

    # gene hằng số => không tách được hai thành phần
    if x.size < 4 or float(np.std(x)) < 1e-8:
        return GammaNormalParams(valid=False, converged=False)

    best = None
    for r in range(n_restarts):
        rng = np.random.default_rng(random_state + 1000 * r)
        p = init_params(
            x, rng,
            dropout_component=dropout_component,
            jitter=0.0 if r == 0 else 0.15,
            eps=eps,
        )
        ll_prev = -np.inf
        n_iter = 0
        converged = False

        for it in range(1, max_iter + 1):
            n_iter = it
            resp = e_step(x, p, eps)
            p_new = m_step(x, resp, eps)
            if not p_new.is_valid():
                break
            ll = log_likelihood(x, p_new, eps)
            if not np.isfinite(ll):
                break
            p = p_new
            if ll - ll_prev < tol:
                converged = True
                ll_prev = ll
                break
            ll_prev = ll

        p.loglik = float(ll_prev)
        p.n_iter = n_iter
        p.converged = converged
        p.valid = p.is_valid()

        if best is None or (p.valid and p.loglik > best.loglik):
            best = p

    best.dropout_is_low = dropout_is_low(best, dropout_component)
    return best


def fit_em_all_genes(
    X_em,
    max_iter=MAX_ITER,
    tol=TOL,
    n_restarts=N_RESTARTS,
    eps=EPS,
    dropout_component=DROPOUT_COMPONENT,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,
    verbose=VERBOSE,
):
    """Chạy EM cho toàn bộ gene, song song hoá theo n_jobs.

    Parameters
    ----------
    X_em : (G, C) float32 — ma trận đã tiền xử lý và đảm bảo dương.

    Returns
    -------
    list[GammaNormalParams] độ dài G.
    """
    X_em = np.asarray(X_em)
    n_genes = X_em.shape[0]

    if verbose:
        print(f"[EM] Ước lượng hỗn hợp Gamma-Normal cho {n_genes} gene ...")

    params_list = Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(
        delayed(fit_em_one_gene)(
            X_em[g, :],
            max_iter=max_iter,
            tol=tol,
            n_restarts=n_restarts,
            eps=eps,
            dropout_component=dropout_component,
            random_state=random_state,
        )
        for g in range(n_genes)
    )
    params_list = list(params_list)

    if verbose:
        n_conv = sum(p.converged for p in params_list)
        n_bad = sum(not p.valid for p in params_list)
        n_switch = sum(p.valid and not p.dropout_is_low for p in params_list)
        print(
            f"[EM] Hội tụ {n_conv}/{n_genes} | suy biến {n_bad} | "
            f"đảo nhãn thành phần {n_switch}"
        )
    return params_list
