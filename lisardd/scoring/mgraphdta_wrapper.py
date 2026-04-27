import os.path as osp
from pathlib import Path

import networkx as nx
import numpy as np
import rdkit.Chem as Chem
import torch
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from torch_geometric.data import Batch, Data

from score_model import MGraphDTA


VOCAB_PROTEIN = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7, "I": 8,
    "H": 9, "K": 10, "M": 11, "L": 12, "O": 13, "N": 14, "Q": 15,
    "P": 16, "S": 17, "R": 18, "U": 19, "T": 20, "W": 21, "V": 22,
    "Y": 23, "X": 24, "Z": 25,
}

VOCAB_ATOMS = ["H", "C", "N", "O", "F", "Cl", "S", "Br", "I"]

_FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(
    osp.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
)


def smile_to_graph(smile: str):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smile}")

    feats = _FEATURE_FACTORY.GetFeaturesForMol(mol)
    g = nx.DiGraph()

    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        g.add_node(
            i,
            a_type=atom.GetSymbol(),
            a_num=atom.GetAtomicNum(),
            acceptor=0,
            donor=0,
            aromatic=atom.GetIsAromatic(),
            hybridization=atom.GetHybridization(),
            num_h=atom.GetTotalNumHs(),
            ExplicitValence=atom.GetExplicitValence(),
            FormalCharge=atom.GetFormalCharge(),
            ImplicitValence=atom.GetImplicitValence(),
            NumExplicitHs=atom.GetNumExplicitHs(),
            NumRadicalElectrons=atom.GetNumRadicalElectrons(),
        )

    for feat in feats:
        if feat.GetFamily() in ("Donor", "Acceptor"):
            for idx in feat.GetAtomIds():
                g.nodes[idx][feat.GetFamily().lower()] = 1

    for i in range(mol.GetNumAtoms()):
        for j in range(mol.GetNumAtoms()):
            bond = mol.GetBondBetweenAtoms(i, j)
            if bond is not None:
                g.add_edge(i, j, b_type=bond.GetBondType(), IsConjugated=int(bond.GetIsConjugated()))

    feat = []
    for n, d in g.nodes(data=True):
        h = []
        h += [int(d["a_type"] == x) for x in VOCAB_ATOMS]
        h.append(d["a_num"])
        h.append(d["acceptor"])
        h.append(d["donor"])
        h.append(int(d["aromatic"]))
        h += [
            int(d["hybridization"] == x)
            for x in (Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3)
        ]
        h.append(d["num_h"])
        h += [d["ExplicitValence"], d["FormalCharge"], d["ImplicitValence"], d["NumExplicitHs"], d["NumRadicalElectrons"]]
        feat.append((n, h))
    feat.sort(key=lambda item: item[0])
    x = torch.FloatTensor([item[1] for item in feat])

    e = {}
    for n1, n2, d in g.edges(data=True):
        et = [
            int(d["b_type"] == x)
            for x in (Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC)
        ]
        et.append(int(d["IsConjugated"] is False))
        et.append(int(d["IsConjugated"] is True))
        e[(n1, n2)] = et

    if len(e) == 0:
        edge_index = torch.LongTensor([[0], [0]])
        edge_attr = torch.FloatTensor([[0, 0, 0, 0, 0, 0]])
    else:
        edge_index = torch.LongTensor(list(e.keys())).T
        edge_attr = torch.FloatTensor(list(e.values()))

    if x.numel() > 0:
        x = (x - x.min()) / (x.max() - x.min())

    return x, edge_index, edge_attr


def seq_to_tensor(sequence: str, max_len: int = 1200, device: str = "cuda") -> torch.LongTensor:
    seq = [VOCAB_PROTEIN.get(s, 0) for s in sequence]
    if len(seq) < max_len:
        seq = np.pad(seq, (0, max_len - len(seq)))
    else:
        seq = seq[:max_len]
    return torch.LongTensor(seq).to(device)


class MGraphDTAScorer:
    def __init__(self, target_protein: str, ckpt_path: str | Path, device: str = "cuda"):
        self.target_protein = target_protein
        self.device = device

        model = MGraphDTA(3, len(VOCAB_PROTEIN) + 1, embedding_size=128, filter_num=32, out_dim=1).to(device)
        state = torch.load(str(ckpt_path), map_location=device)

        try:
            model.load_state_dict(state)
        except Exception:
            renamed = {}
            for k, v in state.items():
                nk = k.replace("lin_l", "lin_rel").replace("lin_r", "lin_root")
                renamed[nk] = v
            model.load_state_dict(renamed)

        model.eval()
        self.model = model
        self._target_tensor = seq_to_tensor(target_protein, device=device).unsqueeze(0)

    def score(self, smiles_batch: list[str]) -> torch.Tensor:
        data_list = []
        for smile in smiles_batch:
            try:
                x, edge_index, edge_attr = smile_to_graph(smile)
                data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, target=self._target_tensor))
            except Exception:
                continue

        if len(data_list) == 0:
            return torch.zeros(0, device=self.device)

        batch = Batch.from_data_list(data_list).to(self.device)
        with torch.no_grad():
            preds = self.model(batch).squeeze(-1)
        return preds
