from mol_dataset import BaselineDataset
import torch

train_dataset = BaselineDataset(
    root="data_train",
    dataset_path="./data/split_0/data_train.txt",
)

val_dataset = BaselineDataset(
    root="data_test",
    dataset_path="./data/split_0/data_val.txt",
)

test_dataset = BaselineDataset(
    root="data_inf",
    dataset_path="./data/split_0/data_test.txt",
)


from torch_geometric.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


from torch_geometric.nn import GraphConv, ChebConv
import torch_geometric.nn as gnn
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GatedGraphConv, GINConv, GINEConv
from torch_geometric.nn import global_mean_pool, global_add_pool


class ResGraphModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResGraphModule, self).__init__()

        self.conv = GINEConv(nn.Linear(in_channels, out_channels), eps=1e-5)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        x_ = x
        x = self.conv(x, edge_index, edge_attr)
        x = self.relu(x) + x_
        return x


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GNN, self).__init__()
        torch.manual_seed(12345)

        self.vert_emb = nn.Linear(13, 2 * hidden_channels, bias=False)
        self.edge_emb = nn.Linear(4, 2 * hidden_channels, bias=False)

        self.main = gnn.Sequential(
            "x, edge_index, edge_attr",
            [
                (
                    ResGraphModule(2 * hidden_channels, 2 * hidden_channels),
                    "x, edge_index, edge_attr -> x",
                ),
                *[
                    (
                        ResGraphModule(2 * hidden_channels, 2 * hidden_channels),
                        "x, edge_index, edge_attr -> x",
                    )
                    for _ in range(8)
                ],
            ],
        )

        self.head = nn.Sequential(
            nn.Linear(2 * hidden_channels, 512),
            nn.GELU(),
            nn.Linear(512, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch):

        x = self.vert_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        x = self.main(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)
        # x = F.dropout(x, p=0.5, training=self.training)

        x = self.head(x)

        return x


model = GNN(hidden_channels=32)
import math

optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=5, factor=0.5, verbose=True
)

criterion = torch.nn.SmoothL1Loss(beta=0.002)
device = "cuda:0"


def train():
    model.train()
    model.to(device)

    tot_loss = 0

    for data in train_loader:  # Iterate in batches over the training dataset.
        data = data.to(device)
        out = model(
            data.x, data.edge_index, data.edge_attr, data.batch
        )  # Perform a single forward pass.

        # print(out.shape, data.y.shape)
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

        tot_loss += loss.item() * data.num_graphs

    # update scheduler
    scheduler.step(tot_loss)


def test(loader):
    model.eval()

    correct = 0
    cnt = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        correct += abs(out - data.y).mean()
        cnt += 1
    return correct / cnt


import time

for epoch in range(1, 171):
    train()
    now = time.time()
    train_mae = test(train_loader)
    test_mae = test(val_loader)
    print(
        f"Epoch: {epoch:03d}, Train MAE: {train_mae:.7f}, Test MAE: {test_mae:.7f} : Total Time {time.time() - now : 2f}"
    )
