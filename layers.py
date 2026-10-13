# layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F



class EnhancedHypergraphConv(nn.Module):
    """
    Enhanced Hypergraph Convolutional Layer.
    """

    def __init__(self, in_channels, out_channels, T_edges, num_nodes):
        super(EnhancedHypergraphConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.norm = nn.LayerNorm(out_channels)

        # Construct sparse incidence matrix H from edge list
        indices = []
        num_edges = len(T_edges)
        for edge_idx, edge in enumerate(T_edges):
            for node_idx in edge:
                indices.append([node_idx, edge_idx])

        H = torch.sparse_coo_tensor(
            torch.tensor(indices).t(), torch.ones(len(indices)),
            (num_nodes, num_edges), dtype=torch.float32
        )
        self.register_buffer('H', H)

        edge_degrees = torch.tensor([len(edge) for edge in T_edges], dtype=torch.float32)
        D_e_inv = 1.0 / (edge_degrees + 1e-9)
        self.register_buffer('D_e_inv', D_e_inv)

        self.edge_meta_bank = nn.Parameter(
            torch.randn(16, in_channels) * 0.01
        )


        self.W_E = nn.Linear(in_channels,in_channels,bias=False)
        # Memory temperature
        self.edge_mem_temperature = 0.2
        # Memory residual normalization
        self.edge_mem_norm = nn.LayerNorm(in_channels)

    def forward(self, x, T_weights):
        batch_size = x.shape[0]

        # Reshape for sparse matrix multiplication
        x_reshaped = x.permute(1, 0, 2).reshape(self.H.shape[0], -1)

        # Node to Edge aggregation
        edge_features = torch.sparse.mm(self.H.t(), x_reshaped)
        edge_features = edge_features.reshape(self.H.shape[1], batch_size, -1).permute(1, 0, 2)

        # Apply normalization and edge weights
        edge_features = edge_features * self.D_e_inv.unsqueeze(0).unsqueeze(-1)

        edge_query_norm = F.normalize(edge_features,p=2,dim=-1)

        edge_bank_norm = F.normalize(self.edge_meta_bank,p=2,dim=-1)

        edge_mem_sim = torch.matmul(edge_query_norm,edge_bank_norm.t()) / self.edge_mem_temperature

        edge_mem_attn = F.softmax(edge_mem_sim,dim=-1)

        edge_mem_read = torch.matmul(edge_mem_attn,self.edge_meta_bank)
        edge_features_mem = (edge_features + edge_mem_read)

        edge_features_mem = self.edge_mem_norm(edge_features_mem)
        #mem

        weighted_edge_features = edge_features_mem * T_weights.unsqueeze(0).unsqueeze(-1)

        # Reshape for back-propagation to nodes
        edge_features_reshaped = weighted_edge_features.permute(1, 0, 2).reshape(self.H.shape[1], -1)

        # Edge to Node aggregation
        aggregated_neighbors = torch.sparse.mm(self.H, edge_features_reshaped)
        aggregated_neighbors = aggregated_neighbors.reshape(self.H.shape[0], batch_size, -1).permute(1, 0, 2)

        # Update node features
        combined = torch.cat([x, aggregated_neighbors], dim=-1)
        updated_features = self.mlp(combined)

        # Residual connection and normalization
        if x.shape[-1] == updated_features.shape[-1]:
            x_out = self.norm(x + updated_features)
        else:
            x_out = self.norm(updated_features)

        return x_out