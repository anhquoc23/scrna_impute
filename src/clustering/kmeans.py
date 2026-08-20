"""Phân cụm tế bào bằng K-means, chọn k bằng Elbow + Silhouette.

Tham số điều chỉnh lấy mặc định từ các biến tĩnh trong scvagan/config.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from ..config import (
    K_MAX,
    K_MIN,
    KMEANS_N_INIT,
    N_PCS,
    RANDOM_STATE,
    SILHOUETTE_MAX_CELLS,
    VERBOSE,
)


def prepare_features(X_proc, n_pcs=N_PCS, random_state=RANDOM_STATE):
    """Chuyển vị sang chiều tế bào rồi giảm chiều bằng PCA.

    LƯU Ý CHIỀU: X_proc là (gene, cell) nhưng K-means phân cụm trên TẾ BÀO,
    nên phải chuyển vị trước. Đây là chỗ rất dễ nhầm.

    Parameters
    ----------
    X_proc : (G, C) float32

    Returns
    -------
    Z : (C, n_pcs_eff) float32
    """
    Xc = np.asarray(X_proc, dtype=np.float32).T          # (C, G)
    Xc = StandardScaler(with_mean=True, with_std=True).fit_transform(Xc)
    n_comp = int(min(n_pcs, Xc.shape[0] - 1, Xc.shape[1]))
    n_comp = max(n_comp, 2)
    Z = PCA(n_components=n_comp, random_state=random_state).fit_transform(Xc)
    return np.ascontiguousarray(Z, dtype=np.float32)


def run_kmeans(Z, k, kmeans_n_init=KMEANS_N_INIT, random_state=RANDOM_STATE):
    """Chạy K-means với k cụm.

    Returns
    -------
    labels : (C,) int32 — giá trị trong [0, k-1]
    inertia : float — tổng bình phương khoảng cách trong cụm
    """
    km = KMeans(
        n_clusters=int(k),
        n_init=kmeans_n_init,
        random_state=random_state,
    ).fit(Z)
    return km.labels_.astype(np.int32), float(km.inertia_)


def elbow_scores(
    Z, k_min=K_MIN, k_max=K_MAX, kmeans_n_init=KMEANS_N_INIT, random_state=RANDOM_STATE
):
    """Inertia theo từng giá trị k (đường Elbow).

    Returns
    -------
    dict[int, float] — ánh xạ k -> inertia.
    """
    out = {}
    for k in range(k_min, k_max + 1):
        if k >= Z.shape[0]:
            break
        _, inertia = run_kmeans(Z, k, kmeans_n_init, random_state)
        out[k] = inertia
    return out


def silhouette_scores(
    Z,
    k_min=K_MIN,
    k_max=K_MAX,
    kmeans_n_init=KMEANS_N_INIT,
    random_state=RANDOM_STATE,
    silhouette_max_cells=SILHOUETTE_MAX_CELLS,
):
    """Hệ số Silhouette theo từng k, giá trị trong [-1, 1].

    Với dữ liệu lớn, Silhouette là O(n^2) nên lấy mẫu con
    ``silhouette_max_cells`` tế bào để tính.

    Returns
    -------
    dict[int, float] — ánh xạ k -> silhouette.
    """
    out = {}
    n = Z.shape[0]
    sample_size = None if n <= silhouette_max_cells else int(silhouette_max_cells)

    for k in range(k_min, k_max + 1):
        if k >= n:
            break
        labels, _ = run_kmeans(Z, k, kmeans_n_init, random_state)
        if len(np.unique(labels)) < 2:
            out[k] = -1.0
            continue
        out[k] = float(
            silhouette_score(
                Z, labels, sample_size=sample_size, random_state=random_state
            )
        )
    return out


def _knee_point(ks, vals):
    """Tìm điểm gãy của đường Elbow.

    Dùng phương pháp khoảng cách lớn nhất tới đường thẳng nối điểm đầu và
    điểm cuối của đường cong (kiểu Kneedle).
    """
    if ks.size < 3:
        return int(ks[0])
    v = (vals - vals.min()) / max(vals.max() - vals.min(), 1e-12)
    kk = (ks - ks.min()) / max(ks.max() - ks.min(), 1e-12)
    x1, y1 = kk[0], v[0]
    x2, y2 = kk[-1], v[-1]
    num = np.abs((y2 - y1) * kk - (x2 - x1) * v + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) + 1e-12
    return int(ks[int(np.argmax(num / den))])


def select_k(inertia, sil):
    """Chọn số cụm k từ hai tiêu chí Elbow và Silhouette.

    Quy tắc: lấy k cực đại Silhouette làm chính (đo trực tiếp chất lượng
    tách cụm), dùng điểm gãy Elbow để đối chiếu. Nếu hai tiêu chí lệch nhau
    quá 1 đơn vị thì vẫn chọn theo Silhouette nhưng ghi nhận bất đồng vào
    report để người dùng tự xem lại hình.

    Returns
    -------
    (k_best, report)
    """
    if not inertia or not sil:
        raise ValueError("inertia và sil không được rỗng.")

    ks = np.array(sorted(inertia.keys()), dtype=int)
    iv = np.array([inertia[k] for k in ks], dtype=float)
    k_elbow = _knee_point(ks, iv)

    ks_s = np.array(sorted(sil.keys()), dtype=int)
    sv = np.array([sil[k] for k in ks_s], dtype=float)
    k_sil = int(ks_s[int(np.argmax(sv))])

    agree = abs(k_elbow - k_sil) <= 1
    k_best = k_sil

    report = {
        "k_values": ks.tolist(),
        "inertia": iv.tolist(),
        "silhouette": [float(sil[k]) for k in ks_s],
        "silhouette_k_values": ks_s.tolist(),
        "k_elbow": int(k_elbow),
        "k_silhouette": int(k_sil),
        "k_best": int(k_best),
        "criteria_agree": bool(agree),
        "rule": "ưu tiên Silhouette; Elbow dùng để đối chiếu",
    }
    return int(k_best), report


def cluster_cells(
    X_proc,
    k_min=K_MIN,
    k_max=K_MAX,
    n_pcs=N_PCS,
    kmeans_n_init=KMEANS_N_INIT,
    random_state=RANDOM_STATE,
    silhouette_max_cells=SILHOUETTE_MAX_CELLS,
    verbose=VERBOSE,
):
    """Toàn bộ chuỗi phân cụm tế bào.

    Returns
    -------
    labels : (C,) int32
    k_best : int
    report : dict
    """
    if verbose:
        print("[Cluster] PCA + quét k cho K-means ...")
    Z = prepare_features(X_proc, n_pcs, random_state)
    inertia = elbow_scores(Z, k_min, k_max, kmeans_n_init, random_state)
    sil = silhouette_scores(
        Z, k_min, k_max, kmeans_n_init, random_state, silhouette_max_cells
    )
    k_best, report = select_k(inertia, sil)
    labels, _ = run_kmeans(Z, k_best, kmeans_n_init, random_state)
    report["n_pcs_used"] = int(Z.shape[1])
    report["cluster_sizes"] = np.bincount(labels, minlength=k_best).tolist()
    if verbose:
        print(
            f"[Cluster] k_elbow={report['k_elbow']} | "
            f"k_silhouette={report['k_silhouette']} | chọn k={k_best}"
        )
    return labels, int(k_best), report


def plot_k_selection(report, out_path):
    """Vẽ đường Elbow và Silhouette trên cùng một hình."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = report["k_values"]
    iv = report["inertia"]
    ks_s = report["silhouette_k_values"]
    sv = report["silhouette"]
    k_best = report["k_best"]

    fig, ax1 = plt.subplots(figsize=(7.5, 4.4), dpi=140)
    ax1.plot(ks, iv, "o-", color="#2563eb", label="Inertia (Elbow)")
    ax1.set_xlabel("Số cụm k")
    ax1.set_ylabel("Inertia", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")

    ax2 = ax1.twinx()
    ax2.plot(ks_s, sv, "s--", color="#dc2626", label="Silhouette")
    ax2.set_ylabel("Hệ số Silhouette", color="#dc2626")
    ax2.tick_params(axis="y", labelcolor="#dc2626")

    ax1.axvline(k_best, color="#16a34a", lw=2, ls=":", label=f"k được chọn = {k_best}")
    ax1.axvline(report["k_elbow"], color="#94a3b8", lw=1.2, ls="-.", alpha=0.8)

    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="best", fontsize=9)
    ax1.set_title("Chọn số cụm k: Elbow và Silhouette", fontweight="bold")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
