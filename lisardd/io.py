"""
Run artifact persistence.

Each run writes to runs/<run_name>/:
    actor.pt, critic.pt    state_dicts (critic.pt only for PPO)
    config.yaml            frozen ExperimentConfig
    history.json           per-epoch reward / loss arrays
    top100.parquet         top-N hits (smiles, reward, epoch)
    meta.json              start/end timestamps, device, hostname

Camera-ready legacy pickles are loadable via load_legacy_pickle() for
the regression notebook's actor-equivalence checks.
"""

import io as _io
import json
import pickle
import platform
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import yaml


@dataclass
class RunArtifacts:
    run_dir: Path
    config: dict
    history: dict
    top100: pd.DataFrame
    meta: dict
    actor_state: Optional[dict] = None
    critic_state: Optional[dict] = None


def save_run(
    run_dir: str | Path,
    config: dict,
    history: dict,
    top_smiles: list[str],
    top_rewards: list[float],
    top_latents: list[list[float]],
    actor_state: dict,
    critic_state: Optional[dict] = None,
    extra_meta: Optional[dict] = None,
) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(actor_state, run_dir / "actor.pt")
    if critic_state is not None:
        torch.save(critic_state, run_dir / "critic.pt")

    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f)

    df = pd.DataFrame({
        "smiles": top_smiles,
        "reward": top_rewards,
        "latent": [list(z) for z in top_latents],
    })
    df.to_parquet(run_dir / "top100.parquet", index=False)

    meta = {
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "host": platform.node(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if extra_meta:
        meta.update(extra_meta)
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return run_dir


def load_run(run_dir: str | Path, load_state: bool = True, device: str = "cpu") -> RunArtifacts:
    run_dir = Path(run_dir)

    with open(run_dir / "config.yaml") as f:
        config = yaml.safe_load(f)
    with open(run_dir / "history.json") as f:
        history = json.load(f)
    with open(run_dir / "meta.json") as f:
        meta = json.load(f)
    top100 = pd.read_parquet(run_dir / "top100.parquet")

    actor_state = torch.load(run_dir / "actor.pt", map_location=device) if load_state else None
    critic_path = run_dir / "critic.pt"
    critic_state = torch.load(critic_path, map_location=device) if (load_state and critic_path.exists()) else None

    return RunArtifacts(
        run_dir=run_dir,
        config=config,
        history=history,
        top100=top100,
        meta=meta,
        actor_state=actor_state,
        critic_state=critic_state,
    )


class _CpuMapUnpickler(pickle.Unpickler):
    """Remap CUDA tensors embedded in legacy pickles to CPU so they load on any machine."""
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            import torch
            return lambda b: torch.load(_io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def load_legacy_pickle(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return _CpuMapUnpickler(f).load()
