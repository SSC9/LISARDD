"""
AutoDock Vina validation pipeline.

Site-targeted docking: the box center is derived from the centroid of
the bound ligand atoms in a holo PDB structure. The bound ligand is
stripped before the receptor PDBQT is generated. A positive control
re-docks the bound ligand into its own binding site to calibrate
expected scores.

Stage 6 work — receptor prep, holo-PDB centroid extraction, batch
docking, cohort metrics — populated in a follow-up session.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class VinaTargetSpec:
    name: str
    holo_pdb_id: str
    chain: str
    ligand_resname: str
    box_size: tuple[float, float, float] = (20.0, 20.0, 20.0)
    notes: Optional[str] = None


JNK3_3FI2 = VinaTargetSpec(
    name="jnk3",
    holo_pdb_id="3FI2",
    chain="A",
    ligand_resname="SR2",
    notes="Aminopyrazole inhibitor SR-3451 (PDB resname SR2) at the ATP site.",
)


def fetch_holo_pdb(pdb_id: str, out_dir: str | Path) -> Path:
    raise NotImplementedError("Stage 6")


def extract_ligand_centroid(holo_pdb_path: str | Path, ligand_resname: str, chain: str) -> tuple[list[float], Path]:
    raise NotImplementedError("Stage 6")


def prepare_receptor_pdbqt(stripped_pdb_path: str | Path, out_path: str | Path) -> Path:
    raise NotImplementedError("Stage 6")


def batch_dock_smiles(
    smiles_list: list[str],
    receptor_pdbqt_path: str | Path,
    center: list[float],
    box_size: tuple[float, float, float],
    *,
    exhaustiveness: int = 32,
    n_poses: int = 20,
    cache_path: Optional[str | Path] = None,
) -> list[Optional[float]]:
    raise NotImplementedError("Stage 6")
