import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.graphgym.register as register


class GraphHead(nn.Module):
    def __init__(self, dim_in, dim_out, L=2, pooling_type="add"):
        super().__init__()
        self.pooling_fun = register.pooling_dict[pooling_type]
        list_FC_layers = [
            nn.Linear(dim_in // 2 ** _l, dim_in // 2 ** (_l + 1), bias=True)
            for _l in range(L)
        ]
        list_FC_layers.append(nn.Linear(dim_in // 2 ** L, dim_out, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        self.activation = F.relu

    def forward(self, x, batch):
        graph_emb = self.pooling_fun(x, batch)
        for _l in range(self.L):
            graph_emb = self.FC_layers[_l](graph_emb)
            graph_emb = self.activation(graph_emb)
        graph_emb = self.FC_layers[self.L](graph_emb)
        return graph_emb
