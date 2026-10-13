# utils.py

import logging
import pickle

import numpy as np
import torch
# from fastdtw import fastdtw
from dtaidistance import dtw

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(data_path, dist_path, data_name):
    """
    Loads traffic data and distance matrix.
    """
    logging.info(f"Loading data from {data_path} and {dist_path}...")
    # data_file_path = f'Data/{data_name}/data.dat'
    data_file_path = data_path
    if data_name == "PEMS08":
        disc = [17856,170,3]
    elif data_name == "PEMS04":
        disc = [16992,307,3]
    elif data_name == "PEMS03":
        disc = [26208,358,3]
    elif data_name == "PEMS07":
        disc = [28224,883,3]
    #-----------------------------------------

    try:
        data = np.memmap(data_file_path, dtype='float32', mode='r', shape=tuple(disc))
    except (FileNotFoundError, ValueError) as e:
        raise ValueError(f'Error loading data file: {data_file_path}') from e
    #-----------------------------------------
    traffic_data = data[:, :, 0]
    num_nodes = traffic_data.shape[1]

    matrix_file_path = dist_path
    try:
        with open(matrix_file_path, 'rb') as f:
            dist_mx = pickle.load(f)  # 读出来就是 (N, N) 的距离矩阵
    except FileNotFoundError:
        logging.error(f"Error: {matrix_file_path} not found.")
        exit()
    #----------------------------------------------

    logging.info(f"Traffic data shape: {traffic_data.shape}")
    logging.info(f"Distance matrix constructed with shape: {dist_mx.shape}")
    return traffic_data, dist_mx



def normalize_data(data):
    """
    Normalizes data using Z-score normalization.
    """
    mean = np.mean(data)
    std = np.std(data)
    logging.info(f"Data normalized with mean: {mean:.4f}, std: {std:.4f}")
    return (data - mean) / std, mean, std


def generate_time_features(num_samples):
    """
    Generates time-of-day and day-of-week features.
    """
    logging.info("Generating time features...")
    time_ind = np.arange(num_samples)
    time_of_day = time_ind % 288
    day_of_week = (time_ind // 288) % 7
    time_features = np.stack((day_of_week, time_of_day), axis=-1)
    logging.info(f"Time features generated with shape: {time_features.shape}")
    return time_features


def create_sequences(data, time_features, seq_len, pred_len):
    """
    Creates input sequences and corresponding labels for model training.
    """
    logging.info(f"Creating sequences with input length {seq_len} and prediction length {pred_len}...")
    xs, ys, xs_time = [], [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        x = data[i:i + seq_len]
        y = data[i + seq_len:i + seq_len + pred_len]
        xt = time_features[i:i + seq_len]
        xs.append(x)
        ys.append(y)
        xs_time.append(xt)
    logging.info(f"Number of sequences created: {len(xs)}")
    return np.array(xs), np.array(ys), np.array(xs_time)


def construct_spatial_hypergraph(dist_mx, k):
    """
    Constructs a spatial hypergraph based on k-nearest neighbors.
    """
    logging.info(f"Constructing spatial hypergraph with k={k}...")
    num_nodes = dist_mx.shape[0]
    hyperedges = []
    for i in range(num_nodes):
        neighbors = np.argsort(dist_mx[i, :])[:k + 1]
        hyperedges.append(tuple(sorted(neighbors)))

    hyperedges = sorted(list(set(hyperedges)))
    H_spatial = np.zeros((num_nodes, len(hyperedges)))
    for j, edge in enumerate(hyperedges):
        for i in edge:
            H_spatial[i, j] = 1
    logging.info(f"Spatial hypergraph constructed with {len(hyperedges)} hyperedges.")
    return H_spatial



def construct_temporal_hypergraph(data, k):
    window = 50
    normalize = True
    logging.info(f"Constructing temporal hypergraph using dtaidistance DTW (k={k}, window={window})...")
    num_nodes = data.shape[1]

    sample_data = data[-24 * 7 * 4:, :].T  # (N, T_sample)
    sample_data = np.ascontiguousarray(sample_data, dtype=np.double)  # 必须是连续double数组

    if normalize:
        mean = sample_data.mean(axis=1, keepdims=True)
        std = sample_data.std(axis=1, keepdims=True)
        sample_data = (sample_data - mean) / (std + 1e-8)

    # 核心：直接计算完整距离矩阵，底层C实现+可选并行，速度极快
    logging.info("Calculating pairwise DTW distance matrix (C-optimized)...")
    dtw_matrix = dtw.distance_matrix(
        sample_data,
        window=window,       # Sakoe-Chiba窗口，和fastdtw的radius类似
        parallel=True,       # 开启并行计算（多核加速）
        use_c=True
    )

    # 对角线设为无穷大，避免选自己
    np.fill_diagonal(dtw_matrix, np.inf)

    # 后面构建超边的逻辑完全不变
    hyperedges = []
    for i in range(num_nodes):
        neighbors = np.argsort(dtw_matrix[i, :])[:k]
        edge = tuple(sorted(list(set([i] + list(neighbors)))))
        hyperedges.append(edge)

    hyperedges = sorted(list(set(hyperedges)))
    H_temporal = np.zeros((num_nodes, len(hyperedges)))
    for j, edge in enumerate(hyperedges):
        for i in edge:
            H_temporal[i, j] = 1

    logging.info(f"Temporal hypergraph (DTW-C) constructed with {len(hyperedges)} hyperedges.")
    return H_temporal



def calculate_periodicity_features(data):
    """
    Calculates daily and weekly periodicity features for each node using FFT.
    """
    logging.info("Calculating periodicity features for each node using FFT...")
    num_nodes = data.shape[1]
    num_samples = data.shape[0]
    points_per_day = 288

    fft_coeffs = np.fft.rfft(data, axis=0)

    num_days = num_samples // points_per_day
    daily_strength = np.abs(fft_coeffs[num_days, :]) if num_days > 0 and fft_coeffs.shape[0] > num_days else np.zeros(
        num_nodes)

    num_weeks = num_days // 7
    weekly_strength = np.abs(fft_coeffs[num_weeks, :]) if num_weeks > 0 and fft_coeffs.shape[
        0] > num_weeks else np.zeros(num_nodes)

    # Normalize features to [0, 1] range
    if daily_strength.max() > 0:
        daily_strength = (daily_strength - daily_strength.min()) / (daily_strength.max() - daily_strength.min())
    if weekly_strength.max() > 0:
        weekly_strength = (weekly_strength - weekly_strength.min()) / (weekly_strength.max() - weekly_strength.min())

    periodicity_features = np.stack((daily_strength, weekly_strength), axis=-1)
    logging.info("Periodicity features calculated and normalized.")
    return torch.FloatTensor(periodicity_features)


def create_initial_tensor(H_spatial, H_temporal):
    """
    Merges spatial and temporal hypergraphs to create the initial tensor T0.
    """
    logging.info("Creating initial tensor T0 by merging hypergraphs...")

    def get_edges_from_H(H):
        edges = []
        for j in range(H.shape[1]):
            edge = tuple(sorted(np.where(H[:, j] == 1)[0]))
            if len(edge) > 1:
                edges.append(edge)
        return edges

    edges_spatial = get_edges_from_H(H_spatial)
    edges_temporal = get_edges_from_H(H_temporal)
    all_edges = sorted(list(set(edges_spatial + edges_temporal)))


    T0_edges = all_edges
    T0_weights = torch.ones(len(all_edges))
    logging.info(f"Initial tensor T0 created with {len(T0_edges)} hyperedges (cliques).")
    return T0_edges, T0_weights


# Evaluation Metrics
def masked_mae(preds, labels, null_val=0.0):
    mask = (labels != null_val).float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask) | torch.isinf(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss) | torch.isinf(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=0.0):
    mask = (labels != null_val).float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask) | torch.isinf(mask), torch.zeros_like(mask), mask)
    labels_safe = torch.clamp(labels, min=1.0)
    loss = torch.abs((preds - labels) / labels_safe)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss) | torch.isinf(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss) * 100


def masked_rmse(preds, labels, null_val=0.0):
    return torch.sqrt(masked_mse(preds, labels, null_val))


def masked_mse(preds, labels, null_val=0.0):
    mask = (labels != null_val).float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask) | torch.isinf(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss) | torch.isinf(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)