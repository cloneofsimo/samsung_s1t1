import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.nn import GraphConv, ChebConv
from torch_geometric.nn import GCNConv, GatedGraphConv, GINConv, GINEConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
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
        skip_connection,
        edge_mlp,
        relu_edge,
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

        if relu_edge:
            self.edge_mlp = nn.Sequential(
                nn.ReLU(),
                nn.Linear(edge_channels + in_channels, in_channels),
                nn.Tanh(),
            )
        else:
            self.edge_mlp = nn.Sequential(
                # nn.ReLU(),
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
        elif self.skip_connection == "res":
            x = x + x_
        elif self.skip_connection == "initial":
            x = x + x_0
        elif self.skip_connection == "res_init":
            x = (x + x_0 + x_) / 2.0
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
        n_layers=5,
        n_ff=512,
        dropout=0.5,
        name="SchEmb",
        n_ydim=1,
        positional=True,
        pos_offset=48,
        edge_mlp=False,
        skip_connection="residual",
        aromatic=True,
        ar_offset=32,
        pooling="mean_pool",
        relu_edge=False,
    ):
        super(SchEmb, self).__init__()

        def n_width(n, _hidden_channels):
            if skip_connection == "dense":
                _hidden_channels = _hidden_channels + _hidden_channels // 4 * n
                _out_channels = _hidden_channels // 4
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
                        edge_mlp=edge_mlp,
                        relu_edge=relu_edge,
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

        if aromatic:
            self.a_emb = nn.Embedding(2, ar_offset)
            EMB_OFFSET = EMB_OFFSET + ar_offset
        self.aromatic = aromatic

        self.vert_emb = nn.Embedding(
            MAX_ITEM + 1, hidden_channels - EMB_OFFSET, padding_idx=MAX_ITEM
        )

        self.pooling = pooling

        if self.pooling == "mean_pool":
            feature_dim = n_width(n_layers, hidden_channels)[0]
        elif self.pooling == "max_pool":
            feature_dim = n_width(n_layers, hidden_channels)[0]
        elif self.pooling == "max_mean_pool":
            feature_dim = n_width(n_layers, hidden_channels)[0] * 2

        # print(feature_dim)

        self.head = nn.Sequential(
            nn.Linear(feature_dim, n_ff),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(n_ff, n_ydim),
        )

    def feat(self, x, edge_index, edge_attr, batch, pos, ar):

        # print(x.shape)
        x = self.vert_emb(x)

        if self.positional:
            pos_ = self.pos_emb(pos)
            x = torch.cat([x, pos_], dim=-1)
        if self.aromatic:
            a_ = self.a_emb(ar)
            x = torch.cat([x, a_], dim=-1)

        x_0 = x
        # print(x.shape)
        edge_attr = self.edge_emb(edge_attr)

        if self.edge_mlp:
            x, _ = self.main(x, edge_index, edge_attr, pos, x_0)
        else:
            x = self.main(x, edge_index, edge_attr, pos, x_0)
        # print(x.shape)
        # x = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=-1)

        if self.pooling == "mean_pool":
            x = global_mean_pool(x, batch)
        elif self.pooling == "max_pool":
            x = global_max_pool(x, batch)
        elif self.pooling == "max_mean_pool":
            x = torch.cat(
                [global_max_pool(x, batch), global_mean_pool(x, batch)], dim=-1
            )

        return x

    def forward(self, x, edge_index, edge_attr, batch, pos, ar):
        x = self.feat(x, edge_index, edge_attr, batch, pos, ar)
        x = self.head(x)
        return x
