import os
import os.path as osp
import torch
from torch_geometric.data import InMemoryDataset, Data
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
import torch
import pickle


MAX_ITEM = 300


def create_atoms(mol, atom_dict):
    """Transform the atom types in a molecule (e.g., H, C, and O)
    into the indices (e.g., H=0, C=1, and O=2).
    Note that each atom index considers the aromaticity.
    """
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], "aromatic")
    atoms = [atom_dict.get(a, MAX_ITEM) for a in atoms]
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


def extract_fingerprints(atoms, i_jbond_dict, fingerprint_dict):
    """Extract the fingerprints from a molecular graph
    based on Weisfeiler-Lehman algorithm.
    """

    if len(atoms) == 1:
        nodes = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        """Update each node ID considering its neighboring nodes and edges.
        The updated node IDs are the fingerprint IDs.
        """
        nodes_ = []
        for i, j_edge in i_jedge_dict.items():

            neighbors = [(nodes[j], edge) for j, edge in j_edge]
            fingerprint = (nodes[i], tuple(sorted(neighbors)))
            _node = fingerprint_dict.get(fingerprint, None)
            if _node is None:  # use the atom itself instead
                _node = nodes[i]
            else:
                # it is fingerprint, so shift it by 20
                _node += 20
            nodes_.append(_node)

        nodes = nodes_

    return np.array(nodes)


def count_fingerprints(radius, atoms, i_jbond_dict, fingerprint_dict, edge_dict):
    """Extract & count Fingerprints, keep info on fingerprint_dict.
    Used for pre-processing the dataset. For fairness, we do not look at the test dataset for fingerprint distribution statistics.

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
                fingerprint_dict[fingerprint] = fingerprint_dict.get(fingerprint, 0) + 1
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

        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
        bonds_str = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
        data_list = []

        with open(
            "/home/simo/dl/comp2021/samsung_s1t1/src/dataset/atom_dict.pickle", "rb"
        ) as handle:
            atom_dict = pickle.load(handle)

        with open(
            "/home/simo/dl/comp2021/samsung_s1t1/src/dataset/fingerprint_dict_topk.pickle",
            "rb",
        ) as handle:
            fingerprint_dict = pickle.load(handle)

        with open(self.dataset_path, "r") as f:
            data = f.read().strip().split("\n")

        for i, line in enumerate(data):
            line = line.split(" ")
            smiles = line[0]
            target = torch.tensor([float(line[1]), float(line[2])])

            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            N = mol.GetNumAtoms()

            # Create node data
            atoms = create_atoms(mol, atom_dict)

            i_jbond_dict = create_ijbonddict(mol, bonds_str)
            fingerprints = extract_fingerprints(atoms, i_jbond_dict, fingerprint_dict)

            x1 = torch.tensor(fingerprints).long()

            # Create Edge data (with edge index)

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

            y = target.unsqueeze(0)

            data = Data(edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i, x=x1)
            data_list.append(data)

        print(len(atom_dict))
        # print(len(bond_dict))
        print(len(fingerprint_dict))

        torch.save(self.collate(data_list), self.processed_paths[0])


def deg_hist():
    # print degree histogram of the dataset

    dataset = FingerprintDataset(
        root="/home/simo/dl/comp2021/samsung_s1t1/data/data_train_fp0",
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
