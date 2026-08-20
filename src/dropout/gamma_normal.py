
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.special import gammaln, logsumexp

from ..config import DROPOUT_COMPONENT, EPS

LOG_2PI = float(np.log(2.0 * np.pi))


class GammaNormalParams(object):
    """Tham số của hỗn hợp Gamma-Normal cho MỘT gene."""

    def __init__(
        self,
        pi: float = 0.5,          # trọng số thành phần Gamma
        k: float = 1.0,           # shape của Gamma
        theta: float = 1.0,       # scale của Gamma
        mu: float = 0.0,          # kỳ vọng của Normal
        sigma: float = 1.0,       # độ lệch chuẩn của Normal
        loglik: float = -np.inf,
        n_iter: int = 0,
        converged: bool = False,
        valid: bool = True,           # False nếu gene suy biến (phương sai ~ 0)
        dropout_is_low: bool = True,  # thành phần dropout có kỳ vọng thấp hơn không
    ):
        self.pi = float(pi)
        self.k = float(k)
        self.theta = float(theta)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.loglik = float(loglik)
        self.n_iter = int(n_iter)
        self.converged = bool(converged)
        self.valid = bool(valid)
        self.dropout_is_low = bool(dropout_is_low)

    def as_tuple(self) -> Tuple[float, float, float, float, float]:
        return (self.pi, self.k, self.theta, self.mu, self.sigma)

    def is_valid(self) -> bool:
        vals = np.array(self.as_tuple(), dtype=np.float64)
        return bool(
            self.valid
            and np.all(np.isfinite(vals))
            and 0.0 <= self.pi <= 1.0
            and self.k > 0.0
            and self.theta > 0.0
            and self.sigma > 0.0
        )

    def gamma_mean(self) -> float:
        """Kỳ vọng của thành phần Gamma: k * theta."""
        return float(self.k * self.theta)

    def normal_mean(self) -> float:
        """Kỳ vọng của thành phần Normal: mu."""
        return float(self.mu)

    def __repr__(self) -> str:
        return (
            "GammaNormalParams(pi={p:.3f}, k={k:.3f}, theta={t:.3f}, "
            "mu={m:.3f}, sigma={s:.3f}, loglik={l:.2f}, converged={c})".format(
                p=self.pi, k=self.k, t=self.theta, m=self.mu,
                s=self.sigma, l=self.loglik, c=self.converged,
            )
        )


# ------------------------------------------------------------ MẬT ĐỘ -----


def log_gamma_pdf(x, k, theta, eps=EPS):
    """log f_Gamma(x; k, theta). x phải > 0. Trả về (n,) float64."""
    x = np.asarray(x, dtype=np.float64)
    xs = np.maximum(x, eps)
    k = max(float(k), eps)
    theta = max(float(theta), eps)
    return (k - 1.0) * np.log(xs) - xs / theta - gammaln(k) - k * np.log(theta)


def log_normal_pdf(x, mu, sigma, eps=EPS):
    """log f_Normal(x; mu, sigma). Trả về (n,) float64."""
    x = np.asarray(x, dtype=np.float64)
    sigma = max(float(sigma), eps)
    z = (x - float(mu)) / sigma
    return -0.5 * LOG_2PI - np.log(sigma) - 0.5 * z * z


def gamma_pdf(x, k, theta, eps=EPS):
    """Mật độ Gamma (n,) float64."""
    return np.exp(log_gamma_pdf(x, k, theta, eps))


def normal_pdf(x, mu, sigma, eps=EPS):
    """Mật độ Normal (n,) float64."""
    return np.exp(log_normal_pdf(x, mu, sigma, eps))


def log_component_matrix(x, p, eps=EPS):
    """Ma trận log của hai thành phần ĐÃ nhân trọng số.

    Returns (n, 2) float64 — cột 0 = Gamma, cột 1 = Normal.
    """
    pi = float(np.clip(p.pi, eps, 1.0 - eps))
    lg = np.log(pi) + log_gamma_pdf(x, p.k, p.theta, eps)
    ln = np.log(1.0 - pi) + log_normal_pdf(x, p.mu, p.sigma, eps)
    return np.stack([lg, ln], axis=1)


def mixture_pdf(x, p, eps=EPS):
    """Mật độ hỗn hợp (n,) float64."""
    return np.exp(logsumexp(log_component_matrix(x, p, eps), axis=1))


def log_likelihood(x, p, eps=EPS):
    """Tổng log-likelihood — tiêu chí dừng của EM. Trả về một số vô hướng."""
    val = float(np.sum(logsumexp(log_component_matrix(x, p, eps), axis=1)))
    return val if np.isfinite(val) else -np.inf


# -------------------------------------------------------- KHỞI TẠO -------


def moment_match_gamma(v, eps=EPS):
    """Khớp moment cho Gamma: k = m^2 / s^2, theta = s^2 / m."""
    m = float(np.mean(v))
    s2 = float(np.var(v))
    m = max(m, eps)
    s2 = max(s2, eps)
    k = float(np.clip(m * m / s2, 1e-3, 1e4))
    theta = float(np.clip(s2 / m, eps, 1e4))
    return k, theta


def init_params(
    x,
    rng,
    dropout_component=DROPOUT_COMPONENT,
    jitter=0.0,
    eps=EPS,
):
    """Khởi tạo tham số bằng cách cắt đôi dữ liệu tại trung vị.

    Nửa giá trị THẤP được gán cho thành phần dropout (vì dropout tạo ra các
    giá trị gần 0), nửa CAO cho thành phần còn lại. Cách này giữ đúng ngữ
    nghĩa trong sơ đồ và làm giảm khả năng EM đảo nhãn hai thành phần.

    Parameters
    ----------
    x : (n,) float64 — vector biểu hiện đã log của một gene.
    rng : bộ sinh ngẫu nhiên, để mỗi lần restart cho khởi tạo khác nhau.
    jitter : biên độ nhiễu tương đối thêm vào (0 = không nhiễu).
    """
    x = np.asarray(x, dtype=np.float64)
    med = float(np.median(x))
    low = x[x <= med]
    high = x[x > med]
    if low.size < 2:
        low = x
    if high.size < 2:
        high = x

    if dropout_component == "normal":
        mu = float(np.mean(low))
        sigma = max(float(np.std(low)), 1e-3)
        k, theta = moment_match_gamma(high, eps)
    else:
        k, theta = moment_match_gamma(low, eps)
        mu = float(np.mean(high))
        sigma = max(float(np.std(high)), 1e-3)

    pi = 0.5
    if jitter > 0.0:
        pi = float(np.clip(pi + rng.normal(0.0, jitter * 0.3), 0.05, 0.95))
        k = float(max(k * (1.0 + rng.normal(0.0, jitter)), 1e-3))
        theta = float(max(theta * (1.0 + rng.normal(0.0, jitter)), eps))
        mu = float(mu + rng.normal(0.0, jitter * max(sigma, 1e-3)))
        sigma = float(max(sigma * (1.0 + rng.normal(0.0, jitter)), 1e-3))

    return GammaNormalParams(pi=pi, k=k, theta=theta, mu=mu, sigma=sigma)
