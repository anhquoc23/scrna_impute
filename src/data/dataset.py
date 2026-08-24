from __future__ import annotations
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse as sp
import torch
from pathlib import Path
from src import config


# Read data CSV
def loader_data(path:str) -> pd.DataFrame: 
    return pd.read_csv(path, header=0, index_col=0)

# Write data CSV
def write_data(data:pd.DataFrame, path:str):
    data.to_csv(path)


# Normnalize and log data
def load_real_data(file_path, min_cells=1, round_input=False):
    
    # --- Load file ---
    df = pd.read_csv(file_path, index_col=0)
    X_raw = df.values
    print(f"[INFO] Loaded data: {X_raw.shape[0]} × {X_raw.shape[1]} (before filtering)")

    # --- Detect orientation ---
    # We expect rows=cells, columns=genes. If not, transpose automatically.
    if X_raw.shape[0] < X_raw.shape[1]:
        print("[INFO] Detected genes as columns → assuming cells × genes format.")
        X = X_raw
    else:
        print("[INFO] Detected genes as rows → transposing to cells × genes.")
        X = X_raw.T
        df_temp = pd.DataFrame(X)
        df_temp.index=df.columns
        df_temp.columns=df.index
        df=df_temp

    # --- Optional rounding ---
    if round_input:
        print("[INFO] Rounding input to nearest integer (for raw counts).")
        X = np.round(X).clip(min=0)

    # --- Filter out genes not expressed in any cell ---
    adata = sc.AnnData(X)
    gene_counts = np.sum(adata.X > 0, axis=0)
    genes_to_keep = gene_counts >= min_cells
    X_filtered = X[:, genes_to_keep].astype(np.float32)
    df_filtered = df.loc[:, genes_to_keep]
    
    print(f"[INFO] Filtered genes: {np.sum(genes_to_keep)} kept out of {len(genes_to_keep)}")

    print(df_filtered.columns)
    print(df_filtered.index)

    return X_filtered, df_filtered


def load_and_filter_data(dropout_rate):
    """Load and filter data for a specific dropout rate"""
    # Load the counts data
    counts_file = f"./dataset/sim.Tung/sim.Tung.drop{dropout_rate}/SplatDrop_counts.csv"
    df3 = pd.read_csv(counts_file)
    Xmiss_original = df3.iloc[:, 1:].values.T

    # Load true counts
    truecounts_file = f"./dataset/sim.Tung/sim.Tung.drop{dropout_rate}/SplatDrop_TrueCounts.csv"
    df1 = pd.read_csv(truecounts_file)
    X_original = df1.iloc[:, 1:].values.T
    
    # Apply gene filtering
    adata = sc.AnnData(Xmiss_original)
    gene_counts = np.sum(adata.X > 0, axis=0)
    genes_to_keep_mask = gene_counts >= 10
    
    # Filter all matrices consistently
    Xmiss = Xmiss_original[:, genes_to_keep_mask].astype('float32')
    X = X_original[:, genes_to_keep_mask]

    return X, Xmiss

def load_and_filter_data_n(dropout_rate):
    """Load and filter data for a specific dropout rate"""
    # Load the counts data
    counts_file = f"sim.Tung/sim.Tung.drop{dropout_rate}/SplatDrop_counts.csv"
    df3 = pd.read_csv(counts_file)
    Xmiss_original = df3.iloc[:, 1:].values.T

    # Load true counts
    truecounts_file = f"sim.Tung/sim.Tung.drop{dropout_rate}/SplatDrop_TrueCounts.csv"
    df1 = pd.read_csv(truecounts_file)
    X_original = df1.iloc[:, 1:].values.T
    
    # Load mask
    mask_file = f"sim.Tung/sim.Tung.drop{dropout_rate}/SplatDrop_Dropout.csv"
    df2 = pd.read_csv(mask_file)
    Xmask_original = df2.iloc[:, 1:].values.T
    
    # Apply gene filtering
    adata = sc.AnnData(Xmiss_original)
    gene_counts = np.sum(adata.X > 0, axis=0)
    genes_to_keep_mask = gene_counts >= 10
    
    # Filter all matrices consistently
    Xmiss = Xmiss_original[:, genes_to_keep_mask].astype('float32')
    X = X_original[:, genes_to_keep_mask]
    Xmask = Xmask_original[:, genes_to_keep_mask]

    return X, Xmiss, Xmask

def normalize_and_log_single(X, do_normalize=True, do_log=True, target_sum=1e4, copy=True):
    """
    Chuẩn hoá và log-transform cho 1 ma trận biểu hiện X (cells × genes).

    - do_normalize: có chuẩn hoá tổng mỗi cell về target_sum không.
    - do_log: có áp dụng log1p không.
    - target_sum: giá trị chuẩn hoá tổng mỗi cell (mặc định 1e4).
    - copy: True → không thay đổi X gốc.
    Trả về:
        X_processed (numpy.ndarray hoặc scipy.sparse)
    """

    adata = ad.AnnData(X.copy() if copy else X)

    if do_normalize:
        sc.pp.normalize_total(adata, target_sum=target_sum, exclude_highly_expressed=False)

    if do_log:
        sc.pp.log1p(adata)

    X_processed = adata.X
    # Ép kiểu an toàn
    X_processed = X_processed.astype(np.float32) if sp.issparse(X_processed) else np.asarray(X_processed, dtype=np.float32)

    return X_processed

def normalize_and_log(X, X_imp, do_normalize=True, do_log=True, target_sum=1e4, copy=True):
    """
    Chuẩn hóa X bằng scanpy và áp dụng CHÍNH XÁC cùng phép biến đổi lên X_imp.
    - target_sum: giống sc.pp.normalize_total (mặc định 1e4).
    - Yêu cầu X và X_imp có cùng số hàng (cell).
    """

    # Tổng theo cell trên X (trước chuẩn hóa)
    if sp.issparse(X):
        total_counts_X = np.asarray(X.sum(axis=1)).ravel()
    else:
        total_counts_X = X.sum(axis=1).ravel()

    # Tránh chia 0
    safe_total = total_counts_X.copy()
    safe_total[safe_total == 0] = 1.0


    adata = ad.AnnData(X.copy() if copy else X)
    if do_normalize:
        sc.pp.normalize_total(adata, target_sum=target_sum, exclude_highly_expressed=False)
    if do_log:
        sc.pp.log1p(adata)
    X_norm = adata.X

    #    factor_i = total_counts_X[i] / target_sum  -> X_imp / factor_i
    if do_normalize:
        factors = safe_total / float(target_sum)
        if sp.issparse(X_imp):
            X_imp_scaled = X_imp.multiply(1.0 / factors[:, None])
        else:
            X_imp_scaled = X_imp / factors[:, None]
    else:
        X_imp_scaled = X_imp.copy()

    # 3) Log1p nếu cần
    if do_log:
        if sp.issparse(X_imp_scaled):
            coo = X_imp_scaled.tocoo()
            coo.data = np.log1p(coo.data)
            X_imp_scaled = coo.tocsr()
        else:
            X_imp_scaled = np.log1p(X_imp_scaled)

    X_norm = X_norm.astype(np.float32) if sp.issparse(X_norm) else np.asarray(X_norm, dtype=np.float32)
    X_imp_scaled = X_imp_scaled.astype(np.float32) if sp.issparse(X_imp_scaled) else np.asarray(X_imp_scaled, dtype=np.float32)

    return X_norm, X_imp_scaled



def filter_data(X):
    # Apply gene filtering
    adata = sc.AnnData(X)
    gene_counts = np.sum(adata.X > 0, axis=0)
    genes_to_keep_mask = gene_counts >= 10
     
    X = X[:, genes_to_keep_mask]
  
    return X

def filter_data_mask(X):
    # Apply gene filtering
    adata = sc.AnnData(X)
    gene_counts = np.sum(adata.X == False, axis=0)
    genes_to_keep_mask = gene_counts >= 1
     
    X = X.values[:, genes_to_keep_mask]
  
    return X

def _read_csv(path):
    """Đọc một file CSV về DataFrame theo quy ước trong config."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    index_col = config.CSV_INDEX_COL
    return pd.read_csv(path, sep=config.CSV_SEP,
                       index_col=(None if index_col is None else int(index_col)))
 
 
def _orient(df):
    """Đưa DataFrame về (tế bào, gene) và trả kèm tên hàng/cột gốc."""
    if bool(config.CSV_GENES_ARE_ROWS):
        gene_names = [str(v) for v in df.index]
        cell_names = [str(v) for v in df.columns]
        arr = df.to_numpy()
        return arr.T, cell_names, gene_names
    cell_names = [str(v) for v in df.index]
    gene_names = [str(v) for v in df.columns]
    return df.to_numpy(), cell_names, gene_names
 
 
def load_matrix_csv(path):
    """Nạp ma trận biểu hiện ĐÃ TIỀN XỬ LÝ.
 
    Returns
    -------
    X : (C, G) float32 — mỗi hàng là một tế bào.
    cell_names : list[str] độ dài C.
    gene_names : list[str] độ dài G.
    """
    X, cell_names, gene_names = _orient(_read_csv(path))
    X = np.asarray(X, dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError(
            f"{Path(path).name} chứa NaN/Inf. Hãy làm sạch trước khi huấn luyện."
        )
    return X, cell_names, gene_names
 
 
def load_mask_csv(path, expected_shape=None):
    """Nạp mask dropout M (giá trị 0/1) do Khối 1 sinh ra.
 
    Returns
    -------
    M : (C, G) float32 — 1 nghĩa là ô đó bị dropout, cần bù khuyết.
    """
    M, _, _ = _orient(_read_csv(path))
    M = np.asarray(M, dtype=np.float32)
    uniq = np.unique(M)
    if not np.isin(uniq, (0.0, 1.0)).all():
        raise ValueError(
            f"{Path(path).name} phải chỉ chứa 0 và 1, nhận được {uniq[:5]}..."
        )
    if expected_shape is not None and M.shape != expected_shape:
        raise ValueError(
            f"Mask có shape {M.shape} nhưng ma trận biểu hiện là {expected_shape}. "
            "Nhiều khả năng hai file khác hướng — kiểm tra config.CSV_GENES_ARE_ROWS."
        )
    return M
 
 
def load_labels_csv(path, n_cells=None):
    """Nạp nhãn cụm từ file hai cột `cell, cluster` do K-means sinh ra.
 
    KHÔNG có giá trị mặc định. Trước đây path=None thì trả về toàn số 0, và
    đó là một lỗi âm thầm nguy hiểm: mọi tế bào rơi vào cùng một cụm, n_clusters
    = 1, embedding điều kiện của Discriminator thành vô nghĩa — mà chương trình
    vẫn chạy hết, không báo gì. Nhãn cụm chỉ được phép đến từ K-means.
    """
    if path is None:
        raise ValueError(
            "load_labels_csv cần đường dẫn file nhãn. Nhãn cụm phải do K-means "
            "sinh ra (cluster_by_kmeans trong main.py), không có mặc định."
        )
    df = pd.read_csv(path, sep=config.CSV_SEP)
    if "cluster" not in df.columns:
        raise ValueError(f"{Path(path).name} phải có cột 'cluster'.")
    labels = df["cluster"].to_numpy().astype(np.int64)
    if n_cells is not None and labels.shape[0] != n_cells:
        raise ValueError(
            f"File nhãn có {labels.shape[0]} dòng nhưng ma trận có {n_cells} tế bào."
        )
    return labels
 
 
def build_data(X, M, labels, cell_names, gene_names, device=None, verbose=None):
    """Đóng gói mảng numpy đã có sẵn trong bộ nhớ thành dict huấn luyện.
 
    Dùng khi Khối 1 vừa chạy xong ngay trong cùng tiến trình — không phải ghi
    ra CSV rồi đọc lại. load_training_data() gọi chính hàm này sau khi đọc file.
 
    Parameters
    ----------
    X, M : (C, G) — CHÚ Ý hướng: mỗi hàng là một tế bào. Khối 1 làm việc với
        (G, C) nên nhớ chuyển vị trước khi gọi.
    labels : (C,) nhãn cụm.
    cell_names, gene_names : list[str].
    device : None thì lấy từ config.DEVICE. Cả X và M được đưa lên thiết bị
        MỘT LẦN ở đây; sau đó mỗi batch chỉ là phép cắt chỉ số ngay trên đó,
        không có lần copy nào giữa các epoch.
    """
    X = np.asarray(X, dtype=np.float32)
    M = np.asarray(M, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if M.shape != X.shape:
        raise ValueError(f"M có shape {M.shape} nhưng X là {X.shape}.")
    if labels.shape[0] != X.shape[0]:
        raise ValueError(
            f"Có {labels.shape[0]} nhãn nhưng {X.shape[0]} tế bào. "
            "Nhiều khả năng quên chuyển vị ma trận."
        )
 
    dev = config.resolve_device() if device is None else torch.device(device)
    data = {
        "X": torch.from_numpy(X).to(dev),
        "M": torch.from_numpy(M).to(dev),
        "labels": torch.from_numpy(labels).to(dev),
        "device": dev,
        "n_cells": int(X.shape[0]),
        "n_genes": int(X.shape[1]),
        "n_clusters": int(labels.max()) + 1 if labels.size else 1,
        "cell_names": list(cell_names),
        "gene_names": list(gene_names),
    }
    if data["n_clusters"] < 2:
        print("[data] CẢNH BÁO: chỉ có 1 cụm — mọi tế bào cùng nhãn 0. "
              "Kiểm tra lại K-means; embedding điều kiện của Discriminator "
              "sẽ vô nghĩa.")
    if bool(config.VERBOSE if verbose is None else verbose):
        print(f"[data] {data['n_cells']} tế bào x {data['n_genes']} gene | "
              f"dropout {float(M.mean())*100:.2f}% | {data['n_clusters']} cụm "
              f"| thiết bị {dev}")
    return data
 
 
def load_training_data(x_csv, m_csv, label_csv):
    """Nạp trọn bộ dữ liệu huấn luyện và chuyển sang tensor.
 
    Parameters
    ----------
    x_csv : ma trận biểu hiện đã tiền xử lý (CSV).
    m_csv : mask dropout M của Khối 1 (CSV).
    label_csv : nhãn cụm (CSV) do K-means sinh ra. BẮT BUỘC.
 
    Returns
    -------
    dict với các khoá:
        X : (C, G) float32 tensor — giá trị quan sát, còn nguyên.
        M : (C, G) float32 tensor — mask dropout.
        labels : (C,) int64 tensor.
        n_cells, n_genes, n_clusters : int.
        cell_names, gene_names : list[str].
    """
    X, cell_names, gene_names = load_matrix_csv(x_csv)
    M = load_mask_csv(m_csv, expected_shape=X.shape)
    labels = load_labels_csv(label_csv, n_cells=X.shape[0])
    return build_data(X, M, labels, cell_names, gene_names)
 
 
# =====================================================================
# CHIA TRAIN / VAL
# =====================================================================
 
def split_train_val(labels):
    """Chia chỉ số tế bào thành tập huấn luyện và kiểm định.
 
    Đọc config.VAL_FRACTION, config.STRATIFY_BY_CLUSTER, config.RANDOM_STATE.
    Phân tầng theo cụm để mỗi cụm đều có mặt ở cả hai tập.
 
    Returns
    -------
    idx_train, idx_val : tensor int64. idx_val có thể rỗng.
    """
    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
    n = labels_np.shape[0]
    frac = float(config.VAL_FRACTION)
    rng = np.random.default_rng(int(config.RANDOM_STATE))
 
    if frac <= 0.0:
        return torch.arange(n, dtype=torch.long), torch.empty(0, dtype=torch.long)
 
    if bool(config.STRATIFY_BY_CLUSTER) and labels_np.max() > 0:
        val_parts = []
        for c in np.unique(labels_np):
            idx_c = np.where(labels_np == c)[0]
            rng.shuffle(idx_c)
            n_val = max(1, int(round(len(idx_c) * frac)))
            val_parts.append(idx_c[:n_val])
        idx_val = np.sort(np.concatenate(val_parts))
    else:
        perm = rng.permutation(n)
        idx_val = np.sort(perm[: max(1, int(round(n * frac)))])
 
    mask = np.ones(n, dtype=bool)
    mask[idx_val] = False
    idx_train = np.where(mask)[0]
    return (torch.from_numpy(idx_train).long(), torch.from_numpy(idx_val).long())
 
 
# =====================================================================
# CHE NHÂN TẠO (để có đáp án mà đo RMSE)
# =====================================================================
 
def simulate_mask(M, epoch):
    """Che thêm một phần các ô ĐÃ QUAN SÁT, tạo tập kiểm chứng có đáp án.
 
    Chỉ che ở nơi M = 0 (ô thật sự quan sát được), nên đáp án vẫn nằm trong X.
    Tỉ lệ lấy từ config.SIM_MASK_RATE; 0.0 thì trả về ma trận 0 (tắt hẳn).
 
    Sinh bằng generator gieo hạt theo epoch nên TÁI LẬP ĐƯỢC: cùng epoch cho
    cùng kết quả, khác epoch cho mẫu che khác.
 
    Returns
    -------
    m_sim : (C, G) float32 — cùng shape với M, không giao với M.
    """
    rate = float(config.SIM_MASK_RATE)
    if rate <= 0.0:
        return torch.zeros_like(M)
    # Generator gieo hạt của PyTorch là generator CPU, nên sinh số ngẫu nhiên
    # trên CPU rồi mới chuyển sang thiết bị của M. Nhờ vậy mẫu che nhân tạo
    # GIỐNG HỆT nhau dù chạy CPU hay GPU — kết quả tái lập được giữa hai máy.
    g = torch.Generator().manual_seed(int(config.RANDOM_STATE) * 100003 + int(epoch))
    u = torch.rand(M.shape, generator=g).to(M.device)
    return ((u < rate) & (M < 0.5)).float()
 
 
# =====================================================================
# LẶP BATCH — thay thế DataLoader
# =====================================================================
 
def iter_batches(data, idx, epoch=0, shuffle=None, m_sim=None):
    """Sinh từng batch bằng cách cắt chỉ số trên ma trận đã nằm sẵn trong RAM.
 
    Parameters
    ----------
    data : dict trả về từ load_training_data.
    idx : tensor int64 — chỉ số các tế bào thuộc tập này (train hoặc val).
    epoch : dùng để gieo hạt xáo trộn, đảm bảo tái lập được.
    shuffle : None thì lấy config.SHUFFLE.
    m_sim : ma trận che nhân tạo TOÀN CỤC (C, G) hoặc None.
 
    Yields
    ------
    dict với x, m, x_masked, m_sim, label, idx — cùng bộ khoá như bản DataLoader
    cũ, nên phần huấn luyện không phải sửa gì.
    """
    if shuffle is None:
        shuffle = bool(config.SHUFFLE)
    batch_size = int(config.BATCH_SIZE)
    n = idx.numel()
    if n == 0:
        return
 
    X, M, labels = data["X"], data["M"], data["labels"]
 
    # cùng lý do như simulate_mask: hoán vị sinh trên CPU cho tái lập được,
    # rồi chuyển sang thiết bị của X để phép cắt chỉ số chạy ngay tại chỗ.
    order = idx.to(X.device)
    if shuffle:
        g = torch.Generator().manual_seed(int(config.RANDOM_STATE) * 7919 + int(epoch))
        order = order[torch.randperm(n, generator=g).to(X.device)]
 
    drop_last = bool(config.DROP_LAST)
 
    for start in range(0, n, batch_size):
        rows = order[start : start + batch_size]
        if drop_last and rows.numel() < batch_size:
            break
        x = X[rows]
        m = M[rows]
        ms = m_sim[rows] if m_sim is not None else torch.zeros_like(m)
        # ô đưa vào mô hình bị làm rỗng ở cả dropout thật lẫn dropout nhân tạo
        keep = (1.0 - m) * (1.0 - ms)
        yield {
            "x": x,
            "m": m,
            "x_masked": x * keep,
            "m_sim": ms,
            "label": labels[rows],
            "idx": rows,
        }
 
 
def n_batches(idx):
    """Số batch của một epoch, để in tiến trình."""
    n = int(idx.numel())
    bs = int(config.BATCH_SIZE)
    if bool(config.DROP_LAST):
        return n // bs
    return (n + bs - 1) // bs

if __name__ == '__main__':
    dropout_rate = 20
    # path = './dataset/sim.Tung/sim.Tung.drop20/SplatDrop_counts.csv'
    # load and filter raw data 
    x, x_miss = load_and_filter_data(dropout_rate=dropout_rate)
    print(x[5], x_miss[5])
    # normalize and log tranform
    x_miss_norm, x_norm = normalize_and_log(x_miss, x, do_normalize=True, do_log=True)
    pd.DataFrame(x_miss_norm).to_csv(f'./dataset/sim.Tung/sim.Tung.drop{dropout_rate}/SplatDrop_norm.csv')
    pd.DataFrame(x_norm).to_csv(f'./dataset/sim.Tung/sim.Tung.drop{dropout_rate}/SplatDrop_truenorm.csv')