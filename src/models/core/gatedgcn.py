import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.nn import GraphConv, ChebConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor

import torch
from torch import Tensor
from torch.nn import Parameter as Param
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import global_mean_pool, global_add_pool
import math


class GatedGraphConv(MessagePassing):
    r"""The gated graph convolution operator from the `"Gated Graph Sequence
    Neural Networks" <https://arxiv.org/abs/1511.05493>`_ paper

    .. math::
        \mathbf{h}_i^{(0)} &= \mathbf{x}_i \, \Vert \, \mathbf{0}

        \mathbf{m}_i^{(l+1)} &= \sum_{j \in \mathcal{N}(i)} e_{j,i} \cdot
        \mathbf{\Theta} \cdot \mathbf{h}_j^{(l)}

        \mathbf{h}_i^{(l+1)} &= \textrm{GRU} (\mathbf{m}_i^{(l+1)},
        \mathbf{h}_i^{(l)})

    up to representation :math:`\mathbf{h}_i^{(L)}`.
    The number of input channels of :math:`\mathbf{x}_i` needs to be less or
    equal than :obj:`out_channels`.
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1`)

    Args:
        out_channels (int): Size of each input sample.
        num_layers (int): The sequence length :math:`L`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(
        self,
        out_channels: int,
        num_layers: int,
        aggr: str = "add",
        bias: bool = True,
        **kwargs
    ):
        super(GatedGraphConv, self).__init__(aggr=aggr, **kwargs)

        self.out_channels = out_channels
        self.num_layers = num_layers

        self.weight = Param(Tensor(num_layers, out_channels, out_channels))
        self.rnn = torch.nn.GRUCell(out_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        gnn.inits.uniform(self.out_channels, self.weight)
        self.rnn.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor = None) -> Tensor:
        """"""
        if x.size(-1) > self.out_channels:
            raise ValueError(
                "The number of input channels is not allowed to "
                "be larger than the number of output channels"
            )

        if x.size(-1) < self.out_channels:
            zero = x.new_zeros(x.size(0), self.out_channels - x.size(-1))
            x = torch.cat([x, zero], dim=1)

        for i in range(self.num_layers):
            m = torch.matmul(x, self.weight[i])
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            m = self.propagate(edge_index, x=m, edge_attr=edge_attr, size=None)
            x = self.rnn(m, x)

        return x

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return F.relu(x_j + edge_attr)

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return "{}({}, num_layers={})".format(
            self.__class__.__name__, self.out_channels, self.num_layers
        )


class PseudoIdentity(nn.Module):
    def __init__(self, in_channels, out_channels, bias):
        super().__init__()
        assert out_channels > in_channels
        self.partial_linear = nn.Linear(
            in_channels, out_channels - in_channels, bias=bias
        )

    def forward(self, x):
        return torch.cat([self.partial_linear(x), x], dim=-1)


class ResGraphModule(nn.Module):
    def __init__(self, in_channels, out_channels, edge_channels, residual=False):
        super(ResGraphModule, self).__init__()

        self.conv = GatedGraphConv(out_channels=out_channels, num_layers=2, aggr="mean")

        self.relu = nn.ReLU()
        self.residual = residual
        self.bn = nn.BatchNorm1d(out_channels)

        self.edge_lin = nn.Linear(edge_channels, in_channels, bias=False)
        self.alpha = 0.5

        if residual:
            if in_channels == out_channels:
                self.res_lin = nn.Identity()
            else:
                self.res_lin = PseudoIdentity(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index, edge_attr):
        x = self.bn(x)
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
