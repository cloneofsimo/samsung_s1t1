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
        skip_connection="residual",
        layer_idx=0,
        growth_rate=32,
        edge_mlp=False,
    ):
        super(ResGraphModule, self).__init__()

        self.conv = SchNetInteraction(
            in_channels=in_channels,
            out_channels=out_channels,
            num_filters=256,
        )
        self.relu = nn.ReLU()
        self.skip_connection = skip_connection

        self.edge_mlp = edge_mlp

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_channels + in_channels, in_channels),
            nn.Tanh(),
        )

    def forward(self, x, edge_index, edge_attr, x_pos, x_0=None):
        x_ = x

        # print(x.shape, edge_index.shape, edge_attr.shape)
        x = self.conv(x, edge_index, edge_attr, x_pos)
        x = self.relu(x)

        if self.skip_connection == "dense":
            x = torch.cat([x, x_], dim=-1)
        elif self.skip_connection == "residual":
            x = x + x_
        elif self.skip_connection == "initial":
            x = x + x_0
        elif self.skip_connection == "res_init":
            x = x + x_0 + x_
        else:
            assert False, "Unknown skip connection type"

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
        dropout=0.5,
        name="SchEmb",
        n_ydim=1,
        positional=True,
        pos_offset=48,
        growth_rate=32,
        edge_mlp=False,
        skip_connection="residual",
    ):
        super(SchEmb, self).__init__()

        def n_width(n, _hidden_channels):
            if skip_connection == "dense":
                _hidden_channels = _hidden_channels + growth_rate * n
                _out_channels = growth_rate
            else:
                _hidden_channels = _hidden_channels
                _out_channels = _hidden_channels

            return _hidden_channels, _out_channels

        # self.vert_emb = nn.Linear(13, hidden_channels, bias=False)
        self.edge_emb = nn.Linear(4, hidden_channels, bias=False)

        self.edge_mlp = edge_mlp
        if self.edge_mlp:

            RETSTRING = "x, edge_index, edge_attr, x_pos, x_0-> x, edge_attr"
        else:
            RETSTRING = "x, edge_index, edge_attr, x_pos, x_0 -> x"

        self.main = gnn.Sequential(
            "x, edge_index, edge_attr, x_pos, x_0",
            [
                (
                    ResGraphModule(
                        n_width(i, hidden_channels)[0],
                        n_width(i, hidden_channels)[1],
                        edge_channels=hidden_channels,
                        skip_connection=skip_connection,
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

        if skip_connection == "dense":
            feature_dim = n_layers * growth_rate + hidden_channels
        else:
            feature_dim = n_width(n_layers, hidden_channels)[0]

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
        x_0 = x
        # print(x.shape)
        edge_attr = self.edge_emb(edge_attr)

        if self.edge_mlp:
            x, _ = self.main(x, edge_index, edge_attr, pos, x_0)
        else:
            x = self.main(x, edge_index, edge_attr, pos, x_0)
        # print(x.shape)
        x = global_mean_pool(x, batch)
        # print(x.shape)
        x = self.head(x)
        # print(x.shape)

        return x
