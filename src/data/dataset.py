import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse as sp


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