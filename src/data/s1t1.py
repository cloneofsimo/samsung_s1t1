import os
import os.path as osp
import torch
from torch_geometric.data import InMemoryDataset, Data
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT

import torch.nn.functional as F


class BaselineDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        dataset_path,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.dataset_path = dataset_path
        self.name = "BaselineDataset"
        super(BaselineDataset, self).__init__(
            root, transform, pre_transform, pre_filter
        )

        print(f"Loading Dataset : {dataset_path}")

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, "processed")

    @property
    def processed_file_names(self):
        return "data.pt"

    def process(self):

        types = {
            "H": 0,
            "C": 1,
            "N": 2,
            "O": 3,
            "F": 4,
            "Br": 5,
            "Cl": 6,
            "S": 7,
            "B": 8,
            "P": 9,
            "I": 10,
            "Si": 11,
            "Ge": 12,
        }
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
        data_list = []

        with open(self.dataset_path, "r") as f:
            data = f.read().strip().split("\n")

        for i, line in enumerate(data):
            line = line.split(" ")
            smiles = line[0]
            target = torch.tensor([float(line[1])])

            mol = Chem.MolFromSmiles(smiles)

            N = mol.GetNumAtoms()
            type_idx = []

            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])

            row, col, edge_type = [], [], []

            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()]]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            row, col = edge_index

            x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()

            # x = torch.cat([x1.to(torch.float), x2], dim=-1)

            y = target.unsqueeze(0)

            data = Data(edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i, x=x1)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

