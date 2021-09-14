import os
import os.path as osp
import torch
from torch_geometric.data import InMemoryDataset, Data
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT

import torch.nn.functional as F

from collections import defaultdict

import numpy as np

from rdkit import Chem

import torch


def create_atoms(mol, atom_dict):
    """Transform the atom types in a molecule (e.g., H, C, and O)
    into the indices (e.g., H=0, C=1, and O=2).
    Note that each atom index considers the aromaticity.
    """
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], "aromatic")
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol, bond_dict):
    """Create a dictionary, in which each key is a node ID
    and each value is the tuples of its neighboring node
    and chemical bond (e.g., single and double) IDs.
    """
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def extract_fingerprints(radius, atoms, i_jbond_dict, fingerprint_dict, edge_dict):
    """Extract the fingerprints from a molecular graph
    based on Weisfeiler-Lehman algorithm.
    """

    if (len(atoms) == 1) or (radius == 0):
        nodes = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges.
            The updated node IDs are the fingerprint IDs.
            """
            nodes_ = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                nodes_.append(fingerprint_dict[fingerprint])

            """Also update each edge ID considering
            its two nodes on both sides.
            """
            i_jedge_dict_ = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    i_jedge_dict_[i].append((j, edge))

            nodes = nodes_
            i_jedge_dict = i_jedge_dict_

    return np.array(nodes)


def split_dataset(dataset, ratio):
    """Shuffle and split a dataset."""
    np.random.seed(1234)  # fix the seed for shuffle.
    np.random.shuffle(dataset)
    n = int(ratio * len(dataset))
    return dataset[:n], dataset[n:]


def create_datasets(task, dataset, radius, device):

    dir_dataset = "../dataset/" + task + "/" + dataset + "/"

    """Initialize x_dict, in which each key is a symbol type
    (e.g., atom and chemical bond) and each value is its index.
    """
    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))

    def create_dataset(filename):

        print(filename)

        """Load a dataset."""
        with open(dir_dataset + filename, "r") as f:
            # smiles_property = f.readline().strip().split()
            data_original = f.read().strip().split("\n")

        """Exclude the data contains '.' in its smiles."""
        data_original = [data for data in data_original if "." not in data.split()[0]]

        dataset = []

        print(filename, "has", len(data_original), "datas.")

        for data in data_original:

            smiles, property = data.strip().split()

            """Create each data with the above defined functions."""
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            atoms = create_atoms(mol, atom_dict)
            molecular_size = len(atoms)
            i_jbond_dict = create_ijbonddict(mol, bond_dict)
            fingerprints = extract_fingerprints(
                radius, atoms, i_jbond_dict, fingerprint_dict, edge_dict
            )
            adjacency = Chem.GetAdjacencyMatrix(mol)

            """Transform the above each data of numpy
            to pytorch tensor on a device (i.e., CPU or GPU).
            """
            fingerprints = torch.LongTensor(fingerprints).to(device)
            adjacency = torch.FloatTensor(adjacency).to(device)
            if task == "classification":
                property = torch.LongTensor([int(property)]).to(device)
            if task == "regression":
                property = torch.FloatTensor([[float(property)]]).to(device)

            dataset.append((fingerprints, adjacency, molecular_size, property))

        return dataset

    dataset_train = create_dataset("data_train.txt")
    dataset_train, dataset_dev = split_dataset(dataset_train, 0.9)
    dataset_test = create_dataset("data_test.txt")

    dataset_inf = create_dataset("data_inf.txt")

    N_fingerprints = len(fingerprint_dict)

    return dataset_train, dataset_dev, dataset_test, N_fingerprints, dataset_inf


class FingerprintDataset(InMemoryDataset):
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
        super(FingerprintDataset, self).__init__(
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


def deg_hist():
    # print degree histogram of the dataset

    dataset = BaselineDataset(
        root="/home/simo/dl/comp2021/samsung_s1t1/data/data_train_0",
        dataset_path="/home/simo/dl/comp2021/samsung_s1t1/data/split_0/data_train.txt",
    )
    from torch_geometric.utils import degree

    deg_sets = []
    for data in dataset[:100000]:
        deg_sets.append((data.edge_index.flatten().bincount() / 2).round().long())

    deg_sets = torch.cat(deg_sets, dim=0).flatten().bincount()
    print(deg_sets)


if __name__ == "__main__":
    deg_hist()
