import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.nn import GraphConv, ChebConv
from torch_geometric.nn import GCNConv, GatedGraphConv, GINConv, GINEConv
from torch_geometric.nn import global_mean_pool, global_add_pool
import math

MAX_ITEM = 300


class ResGraphModule(nn.Module):
    def __init__(self, in_channels, out_channels, edge_channels, residual=False):
        super(ResGraphModule, self).__init__()

        self.conv = GINEConv(nn.Linear(in_channels, out_channels), eps=1e-5)
        self.relu = nn.ReLU()
        self.residual = residual
        # self.bn = nn.BatchNorm1d(out_channels)

        self.edge_lin = nn.Linear(edge_channels, in_channels, bias=False)

        if residual:
            self.res_lin = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index, edge_attr):
        x_ = x
        edge_attr = self.edge_lin(edge_attr)
        # print(x.shape, edge_index.shape, edge_attr.shape)
        x = self.conv(x, edge_index, edge_attr)
        x = self.relu(x)
        # x = self.bn(x)

        if self.residual:
            return x + self.res_lin(x_)
        return x


class GINEEmb(nn.Module):
    def __init__(
        self,
        hidden_channels,
        grow_size=1.5,
        n_layers=5,
        n_ff=512,
        residual=True,
        dropout=0.5,
        name="GINEEmb",
        n_ydim=1,
        positional=True,
    ):
        super(GINEEmb, self).__init__()

        def n_width(n):
            return math.floor(pow(grow_size, n) + 1e-2)

        # self.vert_emb = nn.Linear(13, hidden_channels, bias=False)
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
        EMB_OFFSET = 0
        if positional:
            self.pos_emb = nn.Linear(3, 32, bias=False)
            EMB_OFFSET = 32
        self.positional = positional

        self.vert_emb = nn.Embedding(
            MAX_ITEM + 1, hidden_channels - EMB_OFFSET, padding_idx=MAX_ITEM
        )

        self.head = nn.Sequential(
            nn.Linear(n_width(n_layers) * hidden_channels, n_ff),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(n_ff, n_ydim),
        )

    def forward(self, x, edge_index, edge_attr, batch, pos=None):

        # print(x.shape)
        x = self.vert_emb(x)
        if self.positional:
            pos = self.pos_emb(pos)
            x = torch.cat([x, pos], dim=-1)
        # print(x.shape)
        edge_attr = self.edge_emb(edge_attr)
        x = self.main(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)
        x = self.head(x)

        return x
