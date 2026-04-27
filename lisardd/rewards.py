"""
Reward function factories.

Each factory returns a callable `reward_fn(valid_smiles: list[str]) -> Tensor`
that produces a (n,) reward tensor for the given list of valid SMILES.

Mirrors the reward set published in the workshop paper:
    R_QED       = QED
    R_SA        = 1 - 0.1 * SA
    R_pKd       = predicted pKd from MGraphDTA
    R_pKd_bin   = 1 if 7 <= pKd <= 14 else 0
    R_pKd_diff  = sigmoid(s*(pKd-7)) * sigmoid(-s*(pKd-14))
    R_MO        = w1*R_SA + w2*R_QED + (1-w1-w2)*R_pKd_diff
"""

import os
import sys

import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, QED


_RDKIT_CONTRIB = os.path.join(os.environ.get("RDBASE", ""), "Contrib", "SA_Score")
if _RDKIT_CONTRIB and _RDKIT_CONTRIB not in sys.path:
    sys.path.append(_RDKIT_CONTRIB)
import sascorer  # noqa: E402


def _device(scorer=None):
    if scorer is not None and hasattr(scorer, "device"):
        return scorer.device
    return "cuda" if torch.cuda.is_available() else "cpu"


def reward_qed():
    def fn(valid_smiles):
        out = torch.zeros(len(valid_smiles), device=_device())
        for i, s in enumerate(valid_smiles):
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                out[i] = QED.qed(mol)
        return out
    return fn


def reward_sa():
    def fn(valid_smiles):
        out = torch.zeros(len(valid_smiles), device=_device())
        for i, s in enumerate(valid_smiles):
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                out[i] = 1.0 - 0.1 * sascorer.calculateScore(mol)
        return out
    return fn


def reward_qed_and_sa():
    qed = reward_qed()
    sa = reward_sa()

    def fn(valid_smiles):
        return sa(valid_smiles), qed(valid_smiles)
    return fn


def reward_binding_affinity(scorer):
    def fn(valid_smiles):
        return scorer.score(valid_smiles)
    return fn


def reward_prop_high_binders(scorer, low: float = 7.0, high: float = 14.0):
    def fn(valid_smiles):
        pkd = scorer.score(valid_smiles)
        return ((pkd >= low) & (pkd <= high)).float()
    return fn


def reward_prop_high_binders_diff(scorer, low: float = 7.0, high: float = 14.0, slope: float = 3.0):
    def fn(valid_smiles):
        pkd = scorer.score(valid_smiles)
        lower = torch.sigmoid(slope * (pkd - low))
        upper = torch.sigmoid(-slope * (pkd - high))
        return lower * upper
    return fn


def reward_multi_obj(scorer, w1: float = 0.1, w2: float = 0.1, slope: float = 3.0,
                    low: float = 7.0, high: float = 14.0):
    sa_qed = reward_qed_and_sa()
    pkd_diff = reward_prop_high_binders_diff(scorer, low=low, high=high, slope=slope)

    def fn(valid_smiles):
        sa, qed = sa_qed(valid_smiles)
        binders = pkd_diff(valid_smiles)
        return w1 * sa + w2 * qed + (1.0 - w1 - w2) * binders
    return fn


def reward_mol_weight(low: float = 50.0, up: float = 150.0):
    def fn(valid_smiles):
        out = torch.zeros(len(valid_smiles), device=_device())
        for i, s in enumerate(valid_smiles):
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                w = Descriptors.ExactMolWt(mol)
                out[i] = float(low <= w <= up)
        return out
    return fn
