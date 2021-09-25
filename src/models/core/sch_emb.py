import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.nn import GraphConv, ChebConv
from torch_geometric.nn import GCNConv, GatedGraphConv, GINConv, GINEConv
from torch_geometric.nn import global_mean_pool, global_add_pool
import math
from torch_geometric.nn.conv import MessagePassing
from .custom_layers.sch import SchNetInteraction

MAX_ITEM = 300


class ResGraphModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        edge_channels,
        residual=False,
        dense_layer=False,
        layer_idx=0,
        growth_rate=32,
        edge_mlp=False,
    ):
        super(ResGraphModule, self).__init__()

        if dense_layer:
            in_channels = in_channels + growth_rate * layer_idx
            out_channels = growth_rate

        self.conv = SchNetInteraction(
            in_channels=in_channels,
            out_channels=out_channels,
            num_filters=256,
        )
        self.relu = nn.ReLU()
        self.residual = residual
        # self.bn = nn.BatchNorm1d(out_channels)
        self.dense_layer = dense_layer
        self.edge_mlp = edge_mlp

        self.edge_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(edge_channels + in_channels, in_channels),
            nn.Tanh(),
        )

    def forward(self, x, edge_index, edge_attr, x_pos):
        x_ = x

        # print(x.shape, edge_index.shape, edge_attr.shape)
        x = self.conv(x, edge_index, edge_attr, x_pos)
        x = self.relu(x)

        if self.dense_layer:
            x = torch.cat([x, x_], dim=-1)
        else:
            x = x + x_
        # print(x.shape)
        if self.edge_mlp:
            row, col = edge_index
            edge_attr = (
                self.edge_mlp(torch.cat([edge_attr, x[row] + x[col]], dim=-1))
                + edge_attr
            )
            return x, edge_attr
        else:
            return x


class SchEmb(nn.Module):
    def __init__(
        self,
        hidden_channels,
        grow_size=1.5,
        n_layers=5,
        n_ff=512,
        residual=True,
        dropout=0.5,
        name="SchEmb",
        n_ydim=1,
        positional=True,
        dense_layer=False,
        pos_offset=48,
        growth_rate=32,
        edge_mlp=False,
    ):
        super(SchEmb, self).__init__()

        def n_width(n):
            return math.floor(pow(grow_size, n) + 1e-2)

        # self.vert_emb = nn.Linear(13, hidden_channels, bias=False)
        self.edge_emb = nn.Linear(4, hidden_channels, bias=False)

        self.edge_mlp = edge_mlp
        if self.edge_mlp:

            RETSTRING = "x, edge_index, edge_attr, x_pos -> x, edge_attr"
        else:
            RETSTRING = "x, edge_index, edge_attr, x_pos -> x"

        self.main = gnn.Sequential(
            "x, edge_index, edge_attr, x_pos",
            [
                (
                    ResGraphModule(
                        n_width(i) * hidden_channels,
                        n_width(i + 1) * hidden_channels,
                        edge_channels=hidden_channels,
                        residual=residual,
                        dense_layer=dense_layer,
                        layer_idx=i,
                        growth_rate=growth_rate,
                        edge_mlp=edge_mlp,
                    ),
                    RETSTRING,
                )
                for i in range(n_layers)
            ],
        )
        EMB_OFFSET = 0
        if positional:
            self.pos_emb = nn.Linear(3, pos_offset, bias=False)
            EMB_OFFSET = pos_offset
        self.positional = positional

        self.vert_emb = nn.Embedding(
            MAX_ITEM + 1, hidden_channels - EMB_OFFSET, padding_idx=MAX_ITEM
        )

        if dense_layer:
            feature_dim = n_layers * growth_rate + hidden_channels
        else:
            feature_dim = n_width(n_layers) * hidden_channels

        # print(feature_dim)

        self.head = nn.Sequential(
            nn.Linear(feature_dim, n_ff),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(n_ff, n_ydim),
        )

    def forward(self, x, edge_index, edge_attr, batch, pos=None):

        # print(x.shape)
        x = self.vert_emb(x)
        if self.positional:
            pos_ = self.pos_emb(pos)
            x = torch.cat([x, pos_], dim=-1)
        # print(x.shape)
        edge_attr = self.edge_emb(edge_attr)

        if self.edge_mlp:
            x, _ = self.main(x, edge_index, edge_attr, pos)
        else:
            x = self.main(x, edge_index, edge_attr, pos)
        # print(x.shape)
        x = global_mean_pool(x, batch)
        # print(x.shape)
        x = self.head(x)
        # print(x.shape)

        return x
