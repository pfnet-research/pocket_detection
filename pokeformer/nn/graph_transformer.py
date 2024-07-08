import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from .utils import pyg_softmax


class MultiHeadAttention2Layer(nn.Module):

    def __init__(self, gamma, in_dim, out_dim, num_heads, use_bias):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.gamma = nn.Parameter(torch.tensor(0.5, dtype=float), requires_grad=True)

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.full_graph = False

    def propagate_attention(self, edge_index, K_h, Q_h, E, V_h):
        src = K_h[edge_index[0]]
        dest = Q_h[edge_index[1]]
        score = torch.mul(src, dest)

        score = score / np.sqrt(self.out_dim)
        score = torch.mul(score, E)

        score = pyg_softmax(score.sum(-1, keepdim=True), edge_index[1])

        msg = V_h[edge_index[0]] * score
        wV = torch.zeros_like(V_h)
        scatter(msg, edge_index[1], dim=0, out=wV, reduce="add")
        return wV

    def forward(self, x, edge_attr, edge_index):
        Q_h = self.Q(x)
        K_h = self.K(x)
        E = self.E(edge_attr)

        V_h = self.V(x)

        Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        K_h = K_h.view(-1, self.num_heads, self.out_dim)
        E = E.view(-1, self.num_heads, self.out_dim)

        V_h = V_h.view(-1, self.num_heads, self.out_dim)

        return self.propagate_attention(edge_index, K_h, Q_h, E, V_h)


class GraphTrfmLayer(nn.Module):
    def __init__(
        self,
        gamma,
        in_dim,
        out_dim,
        num_heads,
        dropout=0.0,
        layer_norm=False,
        batch_norm=True,
        residual=True,
        use_bias=False,
        use_gate=True,
    ):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.attention = MultiHeadAttention2Layer(
            gamma=gamma,
            in_dim=in_dim,
            out_dim=out_dim // num_heads,
            num_heads=num_heads,
            use_bias=use_bias,
        )

        self.O_h = nn.Linear(out_dim, out_dim)

        self.use_gate = use_gate
        if self.use_gate:
            self.G_h = nn.Linear(in_dim, out_dim)
            self.sigmoid = nn.Sigmoid()

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)

        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)

    def forward(self, x, edge_attr, edge_index):
        h = x
        h_in1 = h

        h_attn_out = self.attention(x, edge_attr, edge_index)

        h = h_attn_out.view(-1, self.out_channels)

        h = F.dropout(h, self.dropout, training=self.training)

        if self.use_gate:
            g = self.sigmoid(self.G_h(x))
            h = h * g

        h = self.O_h(h)

        if self.residual:
            h = h_in1 + h

        if self.layer_norm:
            h = self.layer_norm1_h(h)

        if self.batch_norm:
            h = self.batch_norm1_h(h)

        h_in2 = h

        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)

        return h


class GraphTransformer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fake_edge_emb = nn.Embedding(1, config.in_dim)

        layers = []
        for _ in range(config.num_layers):
            layers.append(
                GraphTrfmLayer(
                    gamma=config.gamma,
                    in_dim=config.in_dim,
                    out_dim=config.out_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    layer_norm=config.layer_norm,
                    batch_norm=config.batch_norm,
                    residual=config.residual,
                    use_bias=config.use_bias,
                    use_gate=config.use_gate,
                )
            )
        self.trf_layers = torch.nn.Sequential(*layers)

    def forward(self, x, edge_index, edge_attr):
        h = x
        for module in self.trf_layers:
            h = module(x=h, edge_index=edge_index, edge_attr=edge_attr)
        return h
