"""
Plotting and statistical comparison of trained policies.

plot_ppo_vs_reinforce: per-(reward, target) comparison panel matching
                       camera-ready Figures 2-3.
compare_paired:        paired t-test on shared latent batches transformed
                       by both actors, matching the camera-ready
                       statistical analysis.
"""

import numpy as np
import torch
from scipy.stats import ttest_rel

from lisardd.agents.networks import Actor, ActorReinforce
from lisardd.decoding.safe_decode import safe_decode_batch


def plot_ppo_vs_reinforce(ppo_history, reinforce_history, *, ax=None, title="", y_label="Average reward", show_trendlines=True):
    import matplotlib.pyplot as plt

    ppo_scores = ppo_history["average_obj_scores"]
    reinforce_scores = reinforce_history["average_obj_scores"]
    epochs = list(range(len(ppo_scores)))

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, reinforce_scores, label="REINFORCE", color="blue")
    ax.plot(epochs, ppo_scores, label="PPO", color="orange")

    if show_trendlines:
        rfit = np.polyfit(epochs, reinforce_scores, deg=1)
        pfit = np.polyfit(epochs, ppo_scores, deg=1)
        ax.plot(epochs, np.poly1d(rfit)(epochs), linestyle="--", color="blue", alpha=0.6,
                label=f"REINFORCE trend (slope={rfit[0]:.3f})")
        ax.plot(epochs, np.poly1d(pfit)(epochs), linestyle="--", color="orange", alpha=0.6,
                label=f"PPO trend (slope={pfit[0]:.3f})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True)
    return ax


def compare_paired(
    ppo_actor_state: dict,
    reinforce_actor_state: dict,
    decoder,
    reward_fn,
    *,
    latent_dim: int = 32,
    hidden_dim: int = 256,
    n_samples: int = 100,
    n_trials: int = 10,
    normalize: bool = True,
    greedy: bool = True,
    device: str = "cuda",
    seed: int = 42,
):
    torch.manual_seed(seed)

    actor_ppo = Actor(latent_dim, hidden_dim).to(device)
    actor_ppo.load_state_dict(ppo_actor_state)
    actor_ppo.eval()

    actor_reinforce = ActorReinforce(latent_dim, hidden_dim).to(device)
    actor_reinforce.load_state_dict(reinforce_actor_state)
    actor_reinforce.eval()

    mean_ppo, mean_reinforce = [], []

    for trial in range(n_trials):
        with torch.no_grad():
            z_base = torch.randn(n_samples, 3 * latent_dim, device=device)

            mu_p, std_p = actor_ppo(z_base)
            z_p = torch.distributions.MultivariateNormal(mu_p, torch.diag_embed(std_p)).sample()

            mu_r, std_r = actor_reinforce(z_base)
            z_r = torch.distributions.MultivariateNormal(mu_r, torch.diag_embed(std_r)).sample()

            def _decode_score(z):
                z_chunks = torch.chunk(z, 3, dim=1)
                smiles, valid = safe_decode_batch(decoder, z_chunks, greedy=greedy)
                valid_mask = torch.tensor(valid, device=device, dtype=torch.bool)
                valid_smiles = [s for s, v in zip(smiles, valid) if v]
                rewards = torch.zeros(n_samples, device=device)
                rewards[~valid_mask] = -1
                if len(valid_smiles) > 0:
                    rewards[valid_mask] = reward_fn(valid_smiles)
                rewards = rewards.detach()
                if normalize:
                    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
                return rewards

            mean_ppo.append(_decode_score(z_p).mean().item())
            mean_reinforce.append(_decode_score(z_r).mean().item())

    t_stat, p_val = ttest_rel(mean_ppo, mean_reinforce)
    return {
        "mean_ppo": mean_ppo,
        "mean_reinforce": mean_reinforce,
        "t_stat": float(t_stat),
        "p_val": float(p_val),
    }
