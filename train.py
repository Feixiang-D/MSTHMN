# train.py

import argparse
import time
import logging
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils import (load_data, normalize_data, generate_time_features,
                   create_sequences, construct_spatial_hypergraph,
                   construct_temporal_hypergraph, create_initial_tensor,
                   calculate_periodicity_features, masked_mae, masked_mape, masked_rmse)
from model import MLP_Enhanced_ATDHGN


def main(args):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    data_name = args.dataset_name

    data_path = f'Data/{data_name}/data.dat'
    dist_path = f'Data/{data_name}/adj_mx.pkl'


    # --- 1. Load and preprocess data ---
    traffic_data, dist_mx = load_data(data_path, dist_path,data_name)
    norm_data, mean, std = normalize_data(traffic_data)
    time_features = generate_time_features(traffic_data.shape[0])
    X, Y, X_time = create_sequences(norm_data, time_features, args.seq_len, args.pred_len)

    periodicity_features = calculate_periodicity_features(traffic_data).to(device)

    # Split data
    train_size = int(len(X) * 0.6)
    val_size = int(len(X) * 0.2)
    train_X, train_Y, train_X_time = X[:train_size], Y[:train_size], X_time[:train_size]
    val_X, val_Y, val_X_time = X[train_size:train_size + val_size], Y[train_size:train_size + val_size], X_time[
                                                                                                         train_size:train_size + val_size]
    test_X, test_Y, test_X_time = X[train_size + val_size:], Y[train_size + val_size:], X_time[train_size + val_size:]

    # Create DataLoaders
    train_dataset = TensorDataset(torch.FloatTensor(train_X), torch.FloatTensor(train_Y),
                                  torch.LongTensor(train_X_time))
    val_dataset = TensorDataset(torch.FloatTensor(val_X), torch.FloatTensor(val_Y), torch.LongTensor(val_X_time))
    test_dataset = TensorDataset(torch.FloatTensor(test_X), torch.FloatTensor(test_Y), torch.LongTensor(test_X_time))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # --- 2. Construct initial hypergraphs ---
    num_nodes = traffic_data.shape[1]
    H_spatial = construct_spatial_hypergraph(dist_mx, args.k)
    H_temporal = construct_temporal_hypergraph(traffic_data, args.k)
    T0_edges, T0_weights = create_initial_tensor(H_spatial, H_temporal)
    T0_weights = T0_weights.to(device)

    # --- 3. Initialize model, optimizer, loss function ---
    model = MLP_Enhanced_ATDHGN(
        num_nodes=num_nodes, T0_edges=T0_edges, T0_weights=T0_weights,
        seq_len=args.seq_len, pred_len=args.pred_len,
        periodicity_features=periodicity_features,
        embed_dim=64, hid_dim=128
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=False)

    def joint_loss_fn(preds, labels, T_weights, T0_weights, mu_consist):
        pred_loss = masked_mae(preds, labels)
        consistency_loss = F.mse_loss(T_weights, T0_weights) if mu_consist > 0 else torch.tensor(0.0).to(preds.device)
        return pred_loss + mu_consist * consistency_loss, pred_loss

    # --- 4. Training and validation loop ---

    best_model_path = f'best_model_{data_name}_k{args.k}.pth'
    logging.info(f"Model will be saved to: {best_model_path}")

    best_val_mae = float('inf')
    for epoch in range(args.epochs):
        model.train()
        epoch_start_time = time.time()
        total_pred_loss = 0
        for x_batch, y_batch, x_time_batch in train_loader:
            x_batch = x_batch.to(device).unsqueeze(-1)
            y_batch = y_batch.to(device).unsqueeze(-1)
            x_time_batch = x_time_batch.to(device)

            optimizer.zero_grad()
            preds = model(x_batch, x_time_batch)
            loss, pred_loss = joint_loss_fn(preds, y_batch, model.T_weights, model.T0_weights, args.mu_consist)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total_pred_loss += pred_loss.item()

        # Validation
        model.eval()
        val_maes, val_mapes, val_rmses = [], [], []
        with torch.no_grad():
            for x_batch, y_batch, x_time_batch in val_loader:
                x_batch = x_batch.to(device).unsqueeze(-1)
                y_batch_real = y_batch * std + mean
                x_time_batch = x_time_batch.to(device)

                preds = model(x_batch, x_time_batch).squeeze()
                preds_real = preds.cpu() * std + mean

                val_maes.append(masked_mae(preds_real, y_batch_real).item())
                val_mapes.append(masked_mape(preds_real, y_batch_real).item())
                val_rmses.append(masked_rmse(preds_real, y_batch_real).item())

        avg_val_mae = np.mean(val_maes)
        epoch_time = time.time() - epoch_start_time
        print(
            f"Epoch {epoch + 1:2d}/{args.epochs} ({epoch_time:.2f}s) | Val MAE: {avg_val_mae:.4f} | Train MAE Loss: {total_pred_loss / len(train_loader):.4f}")

        scheduler.step(avg_val_mae)
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Best model saved with MAE: {best_val_mae:.4f}")

    # --- 5. Final testing ---
    logging.info("==================== FINAL TEST ====================")
    logging.info(f"Loading best model from: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    test_maes, test_mapes, test_rmses = [], [], []
    with torch.no_grad():
        for x_batch, y_batch, x_time_batch in test_loader:
            x_batch = x_batch.to(device).unsqueeze(-1)
            y_batch_real = y_batch * std + mean
            x_time_batch = x_time_batch.to(device)

            preds = model(x_batch, x_time_batch).squeeze()
            preds_real = preds.cpu() * std + mean

            test_maes.append(masked_mae(preds_real, y_batch_real).item())
            test_mapes.append(masked_mape(preds_real, y_batch_real).item())
            test_rmses.append(masked_rmse(preds_real, y_batch_real).item())

    avg_test_mae = np.mean(test_maes)
    avg_test_mape = np.mean(test_mapes)
    avg_test_rmse = np.mean(test_rmses)

    final_results = f"Final Test Results -- MAE: {avg_test_mae:.4f}, MAPE: {avg_test_mape:.2f}%, RMSE: {avg_test_rmse:.4f}"
    print(final_results)
    logging.info(final_results)


# 数据集划分 6 2 2
# python train.py --device 0 --data_path ./Data/PEMS08/PEMS08.npz --dist_path ./Data/PEMS07/PEMS07.csv
# python train.py --device 0 --dataset_name PEMS08

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')  #做效率实验换成64
    parser.add_argument('--seq_len', type=int, default=12, help='Input sequence length.')
    parser.add_argument('--pred_len', type=int, default=12, help='Prediction sequence length.')
    parser.add_argument('--k', type=int, default=8, help='k for k-NN in hypergraph construction.')
    parser.add_argument('--mu_consist', type=float, default=0.01, help='Weight for consistency loss.')
    parser.add_argument('--device', type=int, default=0, help='CUDA device index.')
    parser.add_argument('--data_path', type=str, help='Path to data file.')
    parser.add_argument('--dist_path', type=str, help='Path to distance matrix file.')
    parser.add_argument('--dataset_name', type=str, default='PEMS08')
    args = parser.parse_args()
    main(args)


