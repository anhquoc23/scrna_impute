"""Tham số của Khối 1 — khai báo bằng BIẾN TĨNH cấp module.

Không có class cấu hình, không đọc file cấu hình.

Các hàm trong package KHÔNG nhận tham số cấu hình qua đối số. Chúng chỉ
nhận dữ liệu, còn tham số thì đọc thẳng từ module này TẠI THỜI ĐIỂM GỌI.

Có hai cách đổi tham số:

1. Sửa trực tiếp giá trị trong file này.
2. Gán lại lúc khởi chạy, TRƯỚC khi gọi hàm:

    from scvagan import config
    config.MAX_ITER = 500
    config.THRESHOLD = 0.7

   Vì hàm đọc giá trị lúc gọi (không phải lúc nạp module), cách này có
   hiệu lực ngay.
"""

# =====================================================================
# XÁC ĐỊNH VỊ TRÍ CÁC Ô VỐN BẰNG 0
#
# Xác suất dropout chỉ tính tại những ô mà giá trị đếm GỐC bằng 0. Vì ma
# trận đưa vào đã qua tiền xử lý, phải cho biết các ô đó giờ trông ra sao.
#   "exact_zero" : ô bằng đúng 0        -> dùng khi bạn log1p
#   "min_value"  : ô bằng giá trị nhỏ nhất -> dùng khi bạn log(x + c), c > 1
#   "from_file"  : nạp mask 0/1 bạn cung cấp sẵn (an toàn nhất)
# =====================================================================
ZERO_DETECTION = "exact_zero"
ZERO_TOL = 1e-8
ZERO_MASK_PATH = None   # đường dẫn .npy, chỉ dùng khi ZERO_DETECTION = "from_file"

# =====================================================================
# CHUẨN BỊ ĐẦU VÀO CHO EM
#
# Mật độ Gamma chỉ xác định trên miền x > 0. Nếu ma trận có giá trị <= 0
# thì tịnh tiến sao cho giá trị nhỏ nhất bằng EM_SHIFT. Chỉ dùng nội bộ
# cho EM, không ảnh hưởng ma trận bàn giao sang khối sau.
# =====================================================================
EM_SHIFT = 1e-3

# =====================================================================
# THUẬT TOÁN EM
# =====================================================================
MAX_ITER = 200          # số vòng lặp E-M tối đa cho một gene
TOL = 1e-4              # dừng khi log-likelihood tăng ít hơn giá trị này
N_RESTARTS = 3          # số lần khởi tạo lại; nên tăng 5-10 khi chạy bản cuối
EPS = 1e-10             # hằng số chống chia 0 / log 0
GAMMA_NEWTON_ITER = 20  # số vòng Newton khi ước lượng shape k của Gamma

# "normal" -> theo sơ đồ scVAGAN (Gamma = biểu hiện, Normal = nhiễu/0)
# "gamma"  -> theo quy ước scImpute (Gamma = dropout, Normal = biểu hiện)
DROPOUT_COMPONENT = "normal"

# True  -> gene bị EM đảo nhãn thành phần sẽ KHÔNG được impute
# False -> vẫn tính bình thường, chỉ ghi nhận vào summary.json
REQUIRE_DROPOUT_IS_LOW = False

# =====================================================================
# NGƯỠNG QUYẾT ĐỊNH
#
# Ô có P >= THRESHOLD thì gắn cờ dropout (M = 1).
# THAM SỐ CẦN PHÂN TÍCH ĐỘ NHẠY: chạy lại với 0.3 / 0.5 / 0.7 / 0.9.
# =====================================================================
THRESHOLD = 0.5

# =====================================================================
# PHÂN CỤM TẾ BÀO
# =====================================================================
K_MIN = 2                    # k nhỏ nhất khi quét Elbow / Silhouette
K_MAX = 15                   # k lớn nhất khi quét
N_PCS = 50                   # số thành phần chính trước khi K-means
KMEANS_N_INIT = 10           # số lần khởi tạo K-means cho mỗi k
SILHOUETTE_MAX_CELLS = 5000  # lấy mẫu con khi tính Silhouette (O(n^2))

# =====================================================================
# ĐỌC FILE CSV
#
# Ma trận huấn luyện nạp thẳng từ CSV, không qua DataLoader.
#   CSV_GENES_ARE_ROWS = True  -> hàng là gene, cột là tế bào (quy ước Khối 1)
#   CSV_INDEX_COL      = 0     -> cột đầu tiên là tên hàng; None nếu không có
# =====================================================================
CSV_SEP = ","
CSV_INDEX_COL = 0
CSV_GENES_ARE_ROWS = True

# =====================================================================
# HUẤN LUYỆN (Khối 3 — GAN)
#
# Mỗi MẪU là MỘT TẾ BÀO; vector đặc trưng là toàn bộ gene.
# Batch có shape (BATCH_SIZE, số gene) — khớp với lớp
# "Linear Projection (Genes -> d_model)" ở đầu Encoder.
# Batch được tạo bằng cách CẮT CHỈ SỐ trên ma trận đã nằm trong RAM.
# =====================================================================
# 8-16 là vùng hợp lý cho scRNA-seq: số tế bào thường chỉ vài nghìn, batch to
# thì mỗi epoch chỉ có dăm bước cập nhật, mô hình gần như không học được gì.
# Mạng chỉ dùng LayerNorm (không có BatchNorm) nên batch nhỏ hoàn toàn an toàn.
BATCH_SIZE = 16
VAL_FRACTION = 0.1        # tỉ lệ tế bào tách ra làm tập kiểm định
SHUFFLE = True            # xáo trộn tập huấn luyện mỗi epoch
DROP_LAST = False         # bỏ batch cuối nếu thiếu; bật khi dùng BatchNorm
STRATIFY_BY_CLUSTER = True  # chia train/val cân bằng theo nhãn cụm

# Che nhân tạo thêm một phần các ô ĐÃ QUAN SÁT để có đáp án mà đo RMSE.
# 0.0 = tắt (giữ đúng phương pháp gốc). Đặt 0.1-0.2 khi cần đánh giá.
SIM_MASK_RATE = 0.0

# =====================================================================
# GENERATOR (Khối 3) — VAE + Multi-Head Attention
#
# Mỗi mẫu là MỘT TẾ BÀO, vector đặc trưng dài G gene. Attention cần một
# CHUỖI token, nên G gene được chia thành N_PATCHES nhóm; mỗi nhóm là một
# token gồm p = ceil(G / N_PATCHES) gene. Đây chính là bước "đóng gói dữ
# liệu trước Attention" trong sơ đồ.
# =====================================================================
N_PATCHES = 64          # P — số token. Chi phí attention là O(P²·d)
D_MODEL = 128           # d — chiều ẩn; phải chia hết cho N_HEADS
N_HEADS = 4             # H — số head
D_FF = 512              # chiều Feed-Forward, thường 2-4 lần D_MODEL
L_ENC = 2               # số lớp Transformer trong Encoder
L_DEC = 2               # số lớp Transformer trong Decoder
D_LATENT = 32           # dz — chiều biểu diễn tiềm ẩn
MODEL_DROPOUT = 0.1     # dropout của MẠNG (khác hẳn dropout sinh học)

# False -> Add & LayerNorm ĐẶT SAU (đúng như sơ đồ, kiểu Transformer gốc)
# True  -> LayerNorm đặt TRƯỚC (pre-norm), huấn luyện ổn định hơn khi mạng sâu
PRE_NORM = False

# =====================================================================
# VÒNG LẶP HUẤN LUYỆN (Khối 3)
#
# Đúng chiến lược 3 bước trong sơ đồ: train VAE trước, rồi mới train GAN.
# =====================================================================
VAE_EPOCHS = 3          # số epoch của Bước 1 (chỉ Reconstruction + KL)
GAN_EPOCHS = 3          # số epoch của Bước 2 (đối kháng)
LR_G = 1e-3             # tốc độ học của Generator
LR_D = 1e-4             # tốc độ học của Discriminator; để nhỏ hơn LR_G vì D
                        # rất dễ thắng áp đảo làm gradient đối kháng biến mất
KL_WEIGHT = 0.1         # trọng số KL sau khi annealing xong
ADV_WEIGHT = 0.01       # trọng số loss đối kháng trong tổng loss của G
IMPUTE_SAMPLES = 8      # 1 = dùng z = mu (tiền định); >1 = lấy mẫu để ước
                        # lượng độ bất định của từng giá trị bù khuyết

# =====================================================================
# THIẾT BỊ TÍNH TOÁN
#
# "auto" -> có GPU NVIDIA thì dùng, không thì chạy CPU. Đây là mặc định nên
#           đổi máy không phải sửa gì.
# "cuda" -> BẮT BUỘC dùng GPU; không có thì báo lỗi ngay thay vì âm thầm
#           chạy CPU rồi bạn ngồi chờ vài tiếng mới biết.
# "cpu"  -> ép chạy CPU dù có GPU.
#
# Cả ma trận X và M được đưa lên GPU MỘT LẦN lúc khởi động; sau đó mỗi batch
# chỉ là phép cắt chỉ số ngay trên GPU, không có lần copy nào giữa các epoch.
# Đây chính là cái lợi của việc bỏ DataLoader.
# =====================================================================
DEVICE = "auto"

# =====================================================================
# HỆ THỐNG
# =====================================================================
RANDOM_STATE = 42       # giữ cố định để kết quả tái lập được
N_JOBS = -1             # số tiến trình song song; -1 = dùng hết CPU
VERBOSE = True          # in tiến trình ra màn hình

# =====================================================================
# GIÁ TRỊ HỢP LỆ
# =====================================================================
VALID_ZERO_DETECTION = ("exact_zero", "min_value", "from_file")
VALID_DROPOUT_COMPONENT = ("normal", "gamma")


def as_dict():
    """Trả về toàn bộ biến tĩnh của module dưới dạng dict.

    Dùng để ghi lại tham số của lần chạy vào summary.json.
    """
    import sys

    module = sys.modules[__name__]
    return {
        name: getattr(module, name)
        for name in sorted(dir(module))
        if name.isupper() and not name.startswith("VALID_")
    }


def apply(params):
    """Gán lại các biến tĩnh của module từ một dict.

    Dùng ở hai chỗ: (1) run_step1.py áp cờ dòng lệnh lúc khởi chạy,
    (2) tiến trình con của joblib nhận lại tham số của tiến trình cha —
    vì tiến trình con nạp module mới nên không thấy thay đổi ở tiến trình cha.

    Khoá lạ bị bỏ qua. Chỉ nhận khoá viết HOA.
    """
    import sys

    module = sys.modules[__name__]
    for name, value in params.items():
        if name.isupper() and not name.startswith("VALID_") and hasattr(module, name):
            setattr(module, name, value)


def resolve_device():
    """Đổi DEVICE thành một torch.device cụ thể. Ném lỗi nếu ép cuda mà không có."""
    import torch

    want = str(DEVICE).lower()
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if want.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f'DEVICE = "{DEVICE}" nhưng PyTorch không thấy GPU nào. '
            "Kiểm tra driver NVIDIA và bản PyTorch đã cài (cpuonly hay pytorch-cuda). "
            'Đặt DEVICE = "auto" để tự lùi về CPU.'
        )
    return torch.device(want)


def validate():
    """Kiểm tra tính hợp lệ của bộ tham số HIỆN HÀNH. Ném ValueError nếu sai."""
    if ZERO_DETECTION not in VALID_ZERO_DETECTION:
        raise ValueError(
            f"ZERO_DETECTION phải thuộc {VALID_ZERO_DETECTION}, nhận '{ZERO_DETECTION}'."
        )
    if ZERO_DETECTION == "from_file" and not ZERO_MASK_PATH:
        raise ValueError('ZERO_DETECTION = "from_file" nhưng ZERO_MASK_PATH đang trống.')
    if ZERO_TOL < 0.0:
        raise ValueError("ZERO_TOL phải >= 0.")
    if EM_SHIFT <= 0.0:
        raise ValueError(
            "EM_SHIFT phải > 0. Mật độ Gamma chỉ xác định trên miền x > 0, "
            "nên nếu ma trận có giá trị <= 0 thì phải tịnh tiến lên."
        )
    if not (0.0 < THRESHOLD < 1.0):
        raise ValueError("THRESHOLD phải nằm trong khoảng (0, 1).")
    if DROPOUT_COMPONENT not in VALID_DROPOUT_COMPONENT:
        raise ValueError(
            f"DROPOUT_COMPONENT phải thuộc {VALID_DROPOUT_COMPONENT}, "
            f"nhận '{DROPOUT_COMPONENT}'."
        )
    if K_MIN < 2 or K_MAX <= K_MIN:
        raise ValueError("Phải có K_MIN >= 2 và K_MAX > K_MIN.")
    if MAX_ITER < 1 or N_RESTARTS < 1:
        raise ValueError("MAX_ITER và N_RESTARTS phải >= 1.")
    if N_PCS < 2:
        raise ValueError("N_PCS phải >= 2.")