import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.nn import GraphConv, ChebConv
from torch_geometric.nn import GatedGraphConv
from torch_geometric.nn import global_mean_pool, global_add_pool
import math

class PseudoIdentity(nn.Module):
    def __init__(self, in_channels, out_channels, bias):
        super().__init__()
        assert out_channels >= in_channels
        self.partial_linear = nn.Linear(in_channels, out_channels - in_channels, bias = bias)
    def forward(self,x):
        return torch.cat([self.partial_linear(x), x], dim = -1)
    
        

class ResGraphModule(nn.Module):
    def __init__(self, in_channels, out_channels, edge_channels, residual=False):
        super(ResGraphModule, self).__init__()

        self.conv = GatedGraphConv(out_channels= out_channels, num_layers= 1, aggr = 'add')
    
        self.relu = nn.ReLU()
        self.residual = residual

        self.edge_lin = nn.Linear(edge_channels, in_channels, bias=False)
        self.alpha = 0.5
        
        if residual:
            if in_channels == out_channels:
                self.res_lin = nn.Identity()
            else:
                self.res_lin = PseudoIdentity(in_channels, out_channels, bias = False)

    def forward(self, x, edge_index, edge_attr):
        x_ = x
        edge_attr = self.edge_lin(edge_attr)
        # print(x.shape, edge_index.shape, edge_attr.shape)
        x = self.conv(x, edge_index, edge_attr)
        x = self.relu(x)

        if self.residual:
            return (1 - self.alpha) * x + self.alpha * self.res_lin(x_)
        return x


class GatedGCN(nn.Module):
    def __init__(
        self,
        hidden_channels,
        grow_size=1.5,
        n_layers=5,
        n_ff=512,
        residual=True,
        dropout=0.5,
        name="GatedGCN",
        n_ydim=1,
    ):
        super(GatedGCN, self).__init__()

        def n_width(n):
            return math.floor(pow(grow_size, n) + 1e-2)

        self.vert_emb = nn.Linear(13, hidden_channels, bias=False)
        self.edge_emb = nn.Linear(4, hidden_channels, bias=False)

        self.main = gnn.Sequential(
            "x, edge_index, edge_attr",
            [
                (
                    ResGraphModule(
                        n_width(i) * hidden_channels,
                        n_width(i + 1) * hidden_channels,
                        edge_channels=hidden_channels,
                        residual=residual,
                    ),
                    "x, edge_index, edge_attr -> x",
                )
                for i in range(n_layers)
            ],
        )

        self.head = nn.Sequential(
            nn.Linear(n_width(n_layers) * hidden_channels, n_ff),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(n_ff, n_ydim),
        )

    def forward(self, x, edge_index, edge_attr, batch):

        x = self.vert_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        x = self.main(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)
        x = self.head(x)

        return x
