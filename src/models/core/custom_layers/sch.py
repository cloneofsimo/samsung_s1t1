from torch_geometric.typing import Adj, OptTensor

import torch
from torch import Tensor
from torch.nn import Parameter as Param
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn


class CfConv(MessagePassing):
    def __init__(
        self,
        n_channels: int,
        mid_channels: int = 64,
        num_filters: int = 16,
        p: float = 2.0,
        gamma: float = 3.0,
    ):
        super().__init__(aggr="add")

        self.num_filter = num_filters
        self.p = p
        centers = [1 / num_filters * i for i in range(num_filters)]
        self.register_buffer("centers", torch.tensor(centers).reshape(1, -1))
        self.gamma = gamma
        self.nn = nn.Sequential(
            nn.Linear(num_filters, mid_channels),
            nn.GELU(),
            nn.Linear(mid_channels, mid_channels),
            nn.GELU(),
            nn.Linear(mid_channels, n_channels),
        )

        self.edge_feat = nn.Linear(n_channels, num_filters)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: Tensor,
        x_pos: Tensor,
    ) -> Tensor:

        x_ = torch.cat([x, x_pos], dim=-1)

        out = self.propagate(edge_index, x=x_, edge_attr=edge_attr)

        return x * out

    def message(self, x_i, x_j, edge_attr):
        # create rotation-invariant filter
        r = torch.norm(x_i[:, -3:] - x_j[:, -3:], p=self.p, dim=-1, keepdim=True)
        feat = r.repeat((1, self.num_filter)) - self.centers
        feat = feat * feat
        feat = (-self.gamma * feat).exp() + self.edge_feat(edge_attr)
        return self.nn(feat) * x_j[:, :-3]


class SchDropoutEdgeConv(MessagePassing):
    def __init__(
        self,
        n_channels: int,
        mid_channels: int = 64,
        num_filters: int = 16,
        p: float = 2.0,
        gamma: float = 3.0,
    ):
        super().__init__(aggr="add")

        self.num_filter = num_filters
        self.p = p
        omeg = [
            10 * n_channels ** (1 - 2 * i / num_filters) for i in range(num_filters)
        ]
        self.register_buffer("omeg", torch.tensor(omeg).reshape(1, -1))
        self.gamma = gamma

        self.nn = nn.Sequential(
            nn.Linear(num_filters * 2 + n_channels * 2, mid_channels),
            nn.GELU(),
            nn.Linear(mid_channels, mid_channels),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mid_channels, n_channels),
        )

        # self.edge_feat = nn.Linear(n_channels, num_filters)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: Tensor,
        x_pos: Tensor,
    ) -> Tensor:

        x_ = torch.cat([x, x_pos], dim=-1)

        out = self.propagate(edge_index, x=x_, edge_attr=edge_attr)

        return x + out, edge_attr

    def message(self, x_i, x_j, edge_attr):
        # create rotation-invariant filter
        # print(x_i.shape, x_j.shape, edge_attr.shape)
        r = torch.norm(x_i[:, -3:] - x_j[:, -3:], p=self.p, dim=-1, keepdim=True)
        # print(r.mean(), r.std())
        feat = r.repeat((1, self.num_filter)) * self.omeg
        feat = torch.cat(
            [torch.sin(feat), torch.cos(feat), edge_attr, x_j[:, :-3]], dim=-1
        )

        return self.nn(feat)


class SchNetInteraction(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = 64,
        num_filters: int = 16,
        p: float = 2.0,
        gamma: float = 3.0,
    ):
        super(SchNetInteraction, self).__init__()

        self.cfconv = SchDropoutEdgeConv(
            out_channels,
            mid_channels,
            num_filters,
            p,
            gamma,
        )

        self.atomwise1 = nn.Linear(in_channels, out_channels)
        self.atomwise2 = nn.Linear(out_channels, out_channels)
        self.atomwise3 = nn.Linear(out_channels, out_channels)
        self.act = nn.ReLU()

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: Tensor,
        x_pos: Tensor,
    ) -> Tensor:

        x = self.atomwise1(x)
        x = self.cfconv(x, edge_index, edge_attr, x_pos)
        x = self.atomwise2(x)
        x = self.act(x)
        x = self.atomwise3(x)

        return x
