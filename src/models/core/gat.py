import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

from torch_geometric.nn import TransformerConv
from torch_geometric.nn import global_mean_pool, global_add_pool
import math

class ResGraphModule(nn.Module):
    def __init__(self, in_channels, out_channels, edge_channels, heads = 8, dropout = 0.2, residual = True):
        super(ResGraphModule, self).__init__()

        self.conv = TransformerConv(in_channels, out_channels, heads = heads, dropout = dropout, edge_dim = edge_channels, concat = False)
        self.relu = nn.ReLU()
        self.residual = residual

        if residual:
            self.res_lin = nn.Linear(in_channels, out_channels, bias = False)

    def forward(self, x, edge_index, edge_attr):
        x_ = x
        x = self.conv(x, edge_index, edge_attr)
        x = self.relu(x) 
        
        if self.residual:
            return x + self.res_lin(x_)
        return x


class GAT(nn.Module):
    def __init__(self, hidden_channels, n_ydim = 1, grow_size = 1.5, n_layers = 5, n_ff = 512, residual = True, dropout = 0.5, heads = 8, name = "GAT"):
        super(GAT, self).__init__()

        def n_width(n):
            return math.floor(pow(grow_size, n) + 1e-2)
        
        self.vert_emb = nn.Linear(13, hidden_channels, bias=False)
        self.edge_emb = nn.Linear(4, hidden_channels, bias=False)


        self.main = gnn.Sequential(
            "x, edge_index, edge_attr",
            
                
                [
                    (
                        ResGraphModule(n_width(i) * hidden_channels, n_width(i + 1)  * hidden_channels, heads = heads, dropout = dropout, residual= residual, edge_channels = hidden_channels),
                        "x, edge_index, edge_attr -> x",
                    )
                    for i in range(n_layers)
                ],
            
        )

        self.head = nn.Sequential(
            nn.Linear(n_width(n_layers) * hidden_channels, n_ff),
            nn.GELU(),
            nn.Dropout(p = 0.05),
            nn.Linear(n_ff, n_ydim),
        )

    def forward(self, x, edge_index, edge_attr, batch):

        x = self.vert_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        x = self.main(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)
        x = self.head(x)

        return x