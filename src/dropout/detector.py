"""Chuẩn bị đầu vào cho Khối 1.

Khối 1 nhận ma trận ĐÃ TIỀN XỬ LÝ (người dùng tự chuẩn hoá / log transform).
Hai việc phải làm trước khi chạy EM:

1. Xác định các ô "vốn bằng 0" — vì chỉ những ô đó mới được tính xác suất
   dropout. Sau tiền xử lý, số 0 có thể đã biến thành một giá trị khác.
2. Đảm bảo mọi giá trị đều > 0 — vì mật độ Gamma chỉ xác định trên x > 0.
"""
from __future__ import annotations

import numpy as np

from ..config import (
    EM_SHIFT,
    VALID_ZERO_DETECTION,
    VERBOSE,
    ZERO_DETECTION,
    ZERO_TOL,
)


def detect_zero_mask(
    X,
    zero_detection=ZERO_DETECTION,
    zero_tol=ZERO_TOL,
    mask_path=None,
    verbose=VERBOSE,
):
    """Xác định các ô vốn bằng 0 trong ma trận đã tiền xử lý.

    Parameters
    ----------
    X : (G, C) — ma trận đã tiền xử lý.
    zero_detection : "exact_zero" | "min_value" | "from_file".
    zero_tol : sai số khi so sánh bằng.
    mask_path : bắt buộc khi zero_detection="from_file". File .npy chứa mảng
        0/1 cùng shape với X, giá trị 1 nghĩa là ô đó vốn bằng 0.
    verbose : in thông tin ra màn hình.

    Returns
    -------
    zero_mask : (G, C) bool — True tại các ô vốn bằng 0.
    """
    X = np.asarray(X)

    if zero_detection not in VALID_ZERO_DETECTION:
        raise ValueError(
            f"zero_detection phải thuộc {VALID_ZERO_DETECTION}, nhận '{zero_detection}'."
        )

    if zero_detection == "from_file":
        if mask_path is None:
            raise ValueError('zero_detection="from_file" nhưng chưa truyền mask_path.')
        mask = np.load(mask_path)
        if mask.shape != X.shape:
            raise ValueError(f"Mask shape {mask.shape} không khớp ma trận {X.shape}.")
        zero_mask = mask.astype(bool)

    elif zero_detection == "exact_zero":
        zero_mask = np.abs(X) <= zero_tol

    else:  # "min_value"
        mn = float(np.min(X))
        zero_mask = np.abs(X - mn) <= zero_tol

    rate = float(zero_mask.mean())
    if verbose:
        print(
            f"[Input] Chiến lược '{zero_detection}': tìm thấy "
            f"{int(zero_mask.sum()):,} ô vốn bằng 0 ({rate*100:.2f}% ma trận)"
        )
    if rate == 0.0:
        raise ValueError(
            f"Không tìm thấy ô nào bằng 0 với chiến lược '{zero_detection}'. "
            "Nếu bạn dùng log(x + c) với c > 1 thì đặt zero_detection='min_value'; "
            "hoặc cung cấp mask sẵn với zero_detection='from_file'."
        )
    if rate > 0.98:
        print(
            f"[Input] CẢNH BÁO: {rate*100:.1f}% ma trận bị coi là bằng 0. "
            "Nhiều khả năng chiến lược xác định ô 0 đang sai."
        )
    return zero_mask


def ensure_positive(X, em_shift=EM_SHIFT, verbose=VERBOSE):
    """Tịnh tiến ma trận để mọi giá trị đều > 0 (yêu cầu của mật độ Gamma).

    Nếu giá trị nhỏ nhất đã > 0 thì không làm gì. Ngược lại cộng thêm một
    lượng để giá trị nhỏ nhất trở thành đúng ``em_shift``.

    Phép tịnh tiến này chỉ dùng NỘI BỘ cho EM, không ảnh hưởng ma trận mà
    bạn bàn giao sang các khối sau.

    Returns
    -------
    X_pos : (G, C) float32
    shift : lượng đã cộng (0.0 nếu không cần tịnh tiến).
    """
    if em_shift <= 0.0:
        raise ValueError("em_shift phải > 0.")

    X = np.asarray(X, dtype=np.float32)
    mn = float(np.min(X))
    if mn > 0.0:
        return X, 0.0

    shift = float(em_shift - mn)
    if verbose:
        print(
            f"[Input] Giá trị nhỏ nhất là {mn:.6g} <= 0, tịnh tiến +{shift:.6g} "
            "để mật độ Gamma không suy biến"
        )
    return (X + shift).astype(np.float32), shift


def prepare_inputs(
    X_proc,
    zero_detection=ZERO_DETECTION,
    zero_tol=ZERO_TOL,
    em_shift=EM_SHIFT,
    mask_path=None,
    verbose=VERBOSE,
):
    """Gộp hai bước chuẩn bị đầu vào.

    Thứ tự quan trọng: xác định mask 0 TRƯỚC khi tịnh tiến, vì tịnh tiến
    sẽ làm các giá trị 0 không còn bằng 0 nữa.

    Returns
    -------
    X_em      : (G, C) float32 — ma trận đã đảm bảo dương, dùng cho EM.
    zero_mask : (G, C) bool
    shift     : lượng đã tịnh tiến.
    """
    zero_mask = detect_zero_mask(
        X_proc,
        zero_detection=zero_detection,
        zero_tol=zero_tol,
        mask_path=mask_path,
        verbose=verbose,
    )
    X_em, shift = ensure_positive(X_proc, em_shift=em_shift, verbose=verbose)
    return X_em, zero_mask, shift
