import os
import os.path as osp
import torch
from torch_geometric.data import Dataset, Data
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
import torch
import pickle
import random
import math

MAX_ITEM = 300
ATOM_DICT = {
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

BOND_DICT = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
ATOM_IDX = "ATOM_IDX"
MAX_ATOM_IDX = 20
PAD_IDX = 0
BOND_OFFSET = 22
CLS_IDX = 40
BOND_DIV = 10.0


def Rotate_Phi(pos):
    # Batch Rotate around z-axis. pos : N, 3
    phi = random.random() * 3.141592 * 2
    rot_mat = torch.tensor([[math.cos(phi), -math.sin(phi), 0],
                        [math.sin(phi), math.cos(phi), 0],
                        [0, 0, 1]])
    pos = pos @ rot_mat
    return pos 

def Rotate_Theta(pos):
    # Batch Rotate around y-axis. pos : N, 3
    theta = random.random() * 3.141592 * 2
    rot_mat = torch.tensor([[math.cos(theta), 0, math.sin(theta)],
                            [0, 1, 0],
                            [-math.sin(theta), 0, math.cos(theta)]])
    pos = pos @ rot_mat
    return pos


class SDFDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        sdf_path,
        train=True,
    ):

        super().__init__()
        self.dataset_path = dataset_path
        self.name = "BaselineDataset"

        print(f"Loading Dataset : {dataset_path}")

        data_list = []

        with open(self.dataset_path, "r") as f:
            data = f.read().strip().split("\n")
            for x in data:
                data_list.append(x.split(" "))

        self.data_list = data_list
        self.sdf_paths = [sdf_path + x[0] + ".sdf" for x in data_list]
        self.train = train

    def len(self):
        return len(self.data_list)

    def get(self, idx):

        line = self.data_list[idx]
        path = line[0]
        target_aux = torch.tensor([float(line[2]), float(line[3])])
        target = torch.tensor([float(line[4])])

        # Create Edge data (with edge index)
        atom_list, atom_pos, mol = self._get_sdf_data(idx)
        atom_pos = atom_pos / BOND_DIV
        atom_pos = atom_pos - atom_pos.mean(dim=0)

        if self.train:
            atom_pos = Rotate_Phi(Rotate_Theta(atom_pos))
            atom_pos = atom_pos + torch.randn((1, 3)) * 0.05
            #atom_pos = atom_pos * torch.exp(torch.randn((1, 3)) * 0.05)

        N = len(atom_list)

        x1 = torch.tensor(atom_list).long()
        # pos = torch.tensor(atom_pos).float()

        row, col, edge_type = [], [], []

        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_type += 2 * [BOND_DICT[bond.GetBondType()]]

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        edge_attr = F.one_hot(edge_type, num_classes=len(BOND_DICT)).to(torch.float)

        perm = (edge_index[0] * N + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_type = edge_type[perm]
        edge_attr = edge_attr[perm]

        y_aux = target_aux.unsqueeze(0)
        y = target.unsqueeze(0)

        data = Data(
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            y_aux=y_aux,
            idx=idx,
            x=x1,
            pos=atom_pos,
        )
        return data

    def _get_sdf_data(self, idx):
        suppl = Chem.SDMolSupplier(self.sdf_paths[idx])
        mols = [mol for mol in suppl if mol is not None]
        mol = mols[-1]
        pos = []
        symb = []
        for i in range(0, mol.GetNumAtoms()):
            poss = mol.GetConformer().GetAtomPosition(i)
            syms = mol.GetAtomWithIdx(i).GetSymbol()
            symb.append(syms)
            pos.append(torch.tensor([poss.x, poss.y, poss.z]))

        atoms = [ATOM_DICT.get(a, MAX_ATOM_IDX) + 1 for a in symb]
        pos = torch.stack(pos)

        

        return atoms, pos, mol
