"""
Per-epoch metric collector.

The per-epoch slowdown observed during PPO training is hypothesized to be
driven by climbing decode-failure rate as the policy drifts off-manifold;
this collector captures the signal needed to confirm or refute that
hypothesis (wall time, decode failure count, batch size) without altering
training behavior.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EpochRecord:
    episode: int
    wall_time: float
    avg_reward: float
    loss_actor: Optional[float]
    loss_critic: Optional[float]
    n_invalid: int
    batch_size: int


@dataclass
class Instrumentation:
    out_path: Optional[Path] = None
    records: list = field(default_factory=list)

    def __post_init__(self):
        self._t0 = time.time()

    def log_epoch(self, **kwargs):
        rec = EpochRecord(**kwargs)
        self.records.append(rec.__dict__)
        if self.out_path is not None:
            with open(self.out_path, "w") as f:
                json.dump(self.records, f)

    def to_dict(self):
        return {
            "elapsed_s": time.time() - self._t0,
            "records": self.records,
        }
