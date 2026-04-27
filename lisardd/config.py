from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Optional


Algo = Literal["ppo", "reinforce"]
Reward = Literal["binding_affinity", "prop_high_binders", "prop_high_binders_diff", "multi_obj", "qed", "sa"]
Target = Literal["jnk3", "egyrase", "none"]
Mode = Literal["full", "smoke"]


@dataclass
class ExperimentConfig:
    run_name: str = "ppo_multi_jnk3"
    algo: Algo = "ppo"
    reward: Reward = "multi_obj"
    target: Target = "jnk3"

    seed: int = 42
    mode: Mode = "full"

    latent_dim: int = 32
    hidden_dim: int = 256
    batch_size: int = 64
    n_epochs: int = 100
    n_top: int = 100

    t_steps: int = 1
    ppo_epochs: int = 6
    clip_eps: float = 0.2
    gamma: float = 0.95
    c1: float = 0.5
    c2: float = 0.01

    lr_ppo: float = 5e-4
    lr_reinforce: float = 1e-3

    normalize: bool = True
    penalty_value: float = -1.0
    greedy: bool = True

    w1: float = 0.1
    w2: float = 0.1
    high_binder_low: float = 7.0
    high_binder_high: float = 14.0
    high_binder_slope: float = 3.0

    use_lr_scheduler: bool = False
    lr_scheduler_patience: int = 10
    lr_scheduler_factor: float = 0.5

    repo_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    runs_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "runs")
    vocab_path: Optional[Path] = None
    vae_ckpt_path: Optional[Path] = None
    score_ckpt_path: Optional[Path] = None
    device: str = "cuda"

    def __post_init__(self):
        if self.vocab_path is None:
            self.vocab_path = self.repo_root / "data" / "chembl" / "recovered_vocab_2000.txt"
        if self.vae_ckpt_path is None:
            self.vae_ckpt_path = self.repo_root / "vae_model" / "vae_model.ckpt"
        if self.score_ckpt_path is None:
            self.score_ckpt_path = self.repo_root / "score_model_weights" / "best_scoring_model.pt"

        if self.mode == "smoke":
            self.batch_size = min(self.batch_size, 8)
            self.n_epochs = min(self.n_epochs, 5)
            self.n_top = min(self.n_top, 10)

    def as_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d
