import torch
import torch.nn as nn
import numpy as np
from layers import EnhancedHypergraphConv
import torch.nn.functional as F

class TemporalAggregationConv(nn.Module):
    def __init__(self, features, seq_len, dropout=0.1):
        super().__init__()
        layers = []
        current_len = seq_len + 1
        while current_len > 1:
            kernel_size = min(4, current_len)
            conv = nn.Conv2d(features, features, (1, kernel_size))
            relu = nn.ReLU()
            drop = nn.Dropout(dropout)
            layers.append(nn.Sequential(conv, relu, drop))
            current_len = current_len - kernel_size + 1
        self.tcn = nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(2)
        x = F.pad(x, (1, 0, 0, 0))
        out = self.tcn(x)
        out = out + x[..., -1].unsqueeze(-1)
        out = out.squeeze(-1).squeeze(-1)
        return out


class GLU(nn.Module):
    def __init__(self, features, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, (1, 1))
        self.conv2 = nn.Conv2d(features, features, (1, 1))
        self.conv3 = nn.Conv2d(features, features, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, C, N, 1]  Conv2d格式
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = x1 * torch.sigmoid(x2)  # 门控线性单元
        out = self.dropout(out)
        out = self.conv3(out)
        return out



class DualSpatialConv(nn.Module):
    def __init__(self, in_channels, out_channels, T_edges, num_nodes, short_mem_num=24):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels

        # 分支1：静态超图卷积
        self.static_hgc = EnhancedHypergraphConv(
            in_channels, out_channels,
            T_edges=T_edges, num_nodes=num_nodes
        )

        self.W_K = nn.Linear(in_channels, in_channels, bias=False)
        self.meta_bank = nn.Parameter(torch.randn(short_mem_num, in_channels) * 0.01)
        self.q_fuse_gate = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
        )

        self.dynamic_mlp = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.dynamic_norm = nn.LayerNorm(out_channels)

        # 两路空间特征门控融合
        self.fusion_gate = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.Sigmoid()
        )
        self.final_norm = nn.LayerNorm(out_channels)
        self.mem_norm = nn.LayerNorm(out_channels)
        self.bias_alpha = nn.Parameter(torch.tensor(0.1))
        self.mem_temperature = 0.2
        self.psi_m = nn.Parameter(torch.randn(num_nodes, in_channels) * 0.01)
        self.E1 = nn.Parameter(torch.randn(num_nodes, 10) * 0.01)
        self.E2 = nn.Parameter(torch.randn(num_nodes, 10) * 0.01)
        self.concatLinear = nn.Linear(2 * out_channels, out_channels)
        self.concatLayer = nn.LayerNorm(out_channels)

        self.glu = GLU(in_channels)

    def forward(self, x, static_weights):
        B, N, D = x.shape

        K = self.W_K(x)

        K_norm = F.normalize(K, p=2, dim=-1)
        bank_norm = F.normalize(self.meta_bank, p=2, dim=-1)
        mem_sim = torch.matmul(K_norm, bank_norm.t()) / self.mem_temperature
        mem_attn = F.softmax(mem_sim, dim=-1)
        mem_read = torch.matmul(mem_attn, self.meta_bank)

        K_mem = mem_read + K
        K_mem = self.mem_norm(K_mem)

        fused = self.static_hgc(K_mem, static_weights)


        smatt_2d = fused.permute(0, 2, 1).unsqueeze(-1)  # [B, D, N, 1]
        glu_out = self.glu(smatt_2d) + smatt_2d
        # 转回 [B, N, D]
        fused_feat = glu_out.squeeze(-1).permute(0, 2, 1)

        if x.shape[-1] == fused_feat.shape[-1]:
            out = self.final_norm(x + fused_feat)
        else:
            out = self.final_norm(fused_feat)
        return out


class MLP_Enhanced_ATDHGN(nn.Module):
    def __init__(self, num_nodes, T0_edges, T0_weights, seq_len, pred_len,
                 periodicity_features,
                 in_dim=1, embed_dim=64, hid_dim=128,
                 time_embed_dim=16, period_embed_dim=16,
                 num_times_of_day=288, num_days_of_week=7,
                 trans_nhead=4, trans_layers=2,
                 num_hgc_layers=2, short_mem_num=24):
        super().__init__()
        self.num_nodes = num_nodes
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.T0_edges = T0_edges

        self.register_buffer('T0_weights', T0_weights.clone())
        self.T_weights = nn.Parameter(T0_weights.clone())
        self.register_buffer('periodicity_features', periodicity_features)

        self.time_of_day_embedding = nn.Embedding(num_times_of_day, time_embed_dim)
        self.day_of_week_embedding = nn.Embedding(num_days_of_week, time_embed_dim)
        self.periodicity_embedding = nn.Linear(2, period_embed_dim)

        total_in_dim = 1 + 2 * time_embed_dim + period_embed_dim
        self.start_conv = nn.Conv2d(in_channels=total_in_dim, out_channels=embed_dim, kernel_size=(1, 1))

        self.node_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.feature_proj = nn.Linear(embed_dim, hid_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hid_dim, nhead=trans_nhead,
            dim_feedforward=hid_dim * 2,
            batch_first=False, dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=trans_layers)

        self.temporal_agg = TemporalAggregationConv(features=hid_dim, seq_len=seq_len)

        self.num_hgc_layers = num_hgc_layers
        self.hgc_layers = nn.ModuleList([
            DualSpatialConv(hid_dim, hid_dim, T_edges=self.T0_edges,
                            num_nodes=self.num_nodes, short_mem_num=short_mem_num)
            for _ in range(num_hgc_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            nn.ReLU(),
            nn.Linear(hid_dim * 2, pred_len)
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim * 2, hid_dim),
        )

        # 两层 LayerNorm（对齐参考文件）
        self.ln1 = nn.LayerNorm(hid_dim)
        self.ln2 = nn.LayerNorm(hid_dim)

        # 两层 Dropout
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x_seq, time_features):
        batch_size, seq_len, _, _ = x_seq.shape

        tod_embed = self.time_of_day_embedding(time_features[..., 1])
        dow_embed = self.day_of_week_embedding(time_features[..., 0])
        time_embed = torch.cat([dow_embed, tod_embed], dim=-1)
        time_embed_expanded = time_embed.unsqueeze(2).expand(-1, -1, self.num_nodes, -1)

        period_feat = self.periodicity_features.view(1, 1, self.num_nodes, 2).expand(batch_size, seq_len, -1, -1)
        period_embed = self.periodicity_embedding(period_feat)

        x_with_features = torch.cat([x_seq, time_embed_expanded, period_embed], dim=-1)
        x_embed = self.start_conv(x_with_features.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        x_enhanced_embed = self.node_mlp(x_embed)
        x_proj = self.feature_proj(x_enhanced_embed)

        temporal_input = x_proj.permute(0, 2, 1, 3).reshape(batch_size * self.num_nodes, seq_len, -1)

        trans_input = temporal_input.permute(1, 0, 2)

        trans_out_seq = self.transformer_encoder(trans_input)

        residual11 = trans_out_seq

        trans_out_seq = self.feed_forward(trans_out_seq)

        trans_out_seq = self.ln1(residual11 + trans_out_seq)

        trans_out_seq = trans_out_seq.permute(1, 0, 2)

        node_temporal_feat = self.temporal_agg(trans_out_seq)
        node_feat = node_temporal_feat.reshape(batch_size, self.num_nodes, -1)

        hgc_out = node_feat
        for hgc in self.hgc_layers:
            hgc_out = hgc(hgc_out, self.T_weights)

        output = self.output_proj(hgc_out)
        return output.permute(0, 2, 1).unsqueeze(-1)


