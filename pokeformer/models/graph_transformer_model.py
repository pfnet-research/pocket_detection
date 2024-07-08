import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
from pokeformer.nn.dist_embeds import GaussianEmbedding
from pokeformer.nn.graph_head import GraphHead
from pokeformer.nn.graph_transformer import GraphTransformer
from torch_geometric.nn import radius_graph

logger = logging.getLogger(__name__)


class DistEmbedType(Enum):
    gaussian = "gaussian"


@dataclass
class GraphTransformerModelConfig:
    num_layers: int = 1
    gamma: float = 1
    in_dim: int = 1
    out_dim: int = 1
    num_heads: int = 4
    dropout: float = 0.0
    layer_norm: bool = False
    batch_norm: bool = True
    residual: bool = True
    use_bias: bool = False
    use_gate: bool = False

    vocab_size: int = 100

    cutoff: float = 8.0
    cuton: float = 3.0
    max_num_neighbors: int = 32
    dist_embed_type: DistEmbedType = DistEmbedType.gaussian
    dist_resoln: Optional[float] = None

    # residue SASA
    sasa_rbf_dim: Optional[int] = None
    sasa_max: float = 200.0

    pooling_type: str = "add"


class GraphTransformerModel(nn.Module):
    @classmethod
    def get_config_class(cls):
        return GraphTransformerModelConfig

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def __init__(self, config):
        super().__init__()

        self.config = config
        hidden_size = config.in_dim

        self.fake_edge_emb = nn.Embedding(1, hidden_size)

        self.net = GraphTransformer(config)

        self.sasa_embedding = None
        if config.sasa_rbf_dim is not None:
            logger.info("using AA + SASA embedding")
            self.node_embed = nn.Embedding(config.vocab_size, hidden_size)
            self.sasa_embedding = GaussianEmbedding(
                out_dim=config.sasa_rbf_dim,
                min_x=0.0,
                max_x=config.sasa_max,
            )
            self.input_linear = nn.Linear(
                hidden_size + config.sasa_rbf_dim, hidden_size
            )
        else:
            logger.info("using AA embedding")
            self.node_embed = nn.Embedding(config.vocab_size, hidden_size)

        self.cutoff = config.cutoff

        self.cuton = config.cuton
        if config.dist_embed_type == DistEmbedType.gaussian:
            self.dist_embed = GaussianEmbedding(
                out_dim=hidden_size,
                dist_resoln=self.config.dist_resoln,
                min_x=self.cuton,
                max_x=self.cutoff,
            )

        self.max_num_neighbors = config.max_num_neighbors
        logger.info(f"using head pooling type: {config.pooling_type=}")
        self.readout = GraphHead(hidden_size, 1, pooling_type=config.pooling_type)

    def forward(self, batch):
        pos = batch.pos
        edge_index = radius_graph(
            pos,
            r=self.cutoff,
            batch=batch.batch,
            max_num_neighbors=self.max_num_neighbors,
        )
        row, col = edge_index
        edge_len = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.dist_embed(edge_len)

        if self.sasa_embedding is not None:
            h_sasa = self.sasa_embedding(batch.sasa)
            h = self.node_embed(batch.x)
            h = self.input_linear(torch.cat([h, h_sasa], axis=1))
        else:
            h = self.node_embed(batch.x)

        h = self.net(x=h, edge_index=edge_index, edge_attr=edge_attr)
        h = self.readout(x=h, batch=batch.batch)
        h = h.squeeze(1)
        return h
