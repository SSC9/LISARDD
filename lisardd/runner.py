"""
End-to-end experiment runner.

run_experiment(config) wires together the generator, scorer, reward,
agent, and IO. One call -> one row in runs/.
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from lisardd.agents.networks import Actor, ActorReinforce, Critic
from lisardd.agents.ppo import train_ppo
from lisardd.agents.reinforce import train_reinforce
from lisardd.config import ExperimentConfig
from lisardd.decoding.safe_decode import safe_decode_batch
from lisardd.generators.hiervae_wrapper import HierVAEGenerator
from lisardd.instrumentation import Instrumentation
from lisardd.io import save_run
from lisardd.rewards import (
    reward_binding_affinity,
    reward_multi_obj,
    reward_prop_high_binders,
    reward_prop_high_binders_diff,
    reward_qed,
    reward_sa,
)
from lisardd.scoring.mgraphdta_wrapper import MGraphDTAScorer
from lisardd.targets import get_sequence


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_reward(cfg: ExperimentConfig, scorer):
    if cfg.reward == "binding_affinity":
        return reward_binding_affinity(scorer)
    if cfg.reward == "prop_high_binders":
        return reward_prop_high_binders(scorer, low=cfg.high_binder_low, high=cfg.high_binder_high)
    if cfg.reward == "prop_high_binders_diff":
        return reward_prop_high_binders_diff(scorer, low=cfg.high_binder_low, high=cfg.high_binder_high, slope=cfg.high_binder_slope)
    if cfg.reward == "multi_obj":
        return reward_multi_obj(scorer, w1=cfg.w1, w2=cfg.w2, slope=cfg.high_binder_slope, low=cfg.high_binder_low, high=cfg.high_binder_high)
    if cfg.reward == "qed":
        return reward_qed()
    if cfg.reward == "sa":
        return reward_sa()
    raise ValueError(f"Unknown reward: {cfg.reward}")


def _decode_top_latents(generator, latents: list, greedy: bool = True):
    if len(latents) == 0:
        return [], []
    z = torch.tensor(latents, dtype=torch.float32, device=generator.device)
    z_chunks = torch.chunk(z, 3, dim=1)
    smiles, valid = safe_decode_batch(generator.decoder, z_chunks, greedy=greedy)
    return smiles, valid


def run_experiment(cfg: ExperimentConfig) -> Path:
    _set_seed(cfg.seed)
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = cfg.runs_dir / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    instrumentation = Instrumentation(out_path=run_dir / "instrumentation.json")

    generator = HierVAEGenerator(
        vocab_path=cfg.vocab_path,
        ckpt_path=cfg.vae_ckpt_path,
        device=cfg.device,
        latent_size=cfg.latent_dim,
    )

    needs_scorer = cfg.reward in ("binding_affinity", "prop_high_binders", "prop_high_binders_diff", "multi_obj")
    scorer = None
    if needs_scorer:
        if cfg.target == "none":
            raise ValueError(f"Reward {cfg.reward} requires a protein target.")
        scorer = MGraphDTAScorer(
            target_protein=get_sequence(cfg.target),
            ckpt_path=cfg.score_ckpt_path,
            device=cfg.device,
        )

    reward_fn = _build_reward(cfg, scorer)

    if cfg.algo == "ppo":
        actor = Actor(cfg.latent_dim, cfg.hidden_dim).to(cfg.device)
        critic = Critic(cfg.latent_dim, cfg.hidden_dim).to(cfg.device)
        opt = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=cfg.lr_ppo)
        result = train_ppo(
            actor, critic, generator.decoder, reward_fn, opt,
            episodes=cfg.n_epochs, batch_size=cfg.batch_size, t_steps=cfg.t_steps,
            ppo_epochs=cfg.ppo_epochs, c1=cfg.c1, c2=cfg.c2,
            normalize=cfg.normalize, penalty_value=cfg.penalty_value,
            clip_eps=cfg.clip_eps, gamma=cfg.gamma, latent_dim=cfg.latent_dim,
            greedy=cfg.greedy, use_lr_scheduler=cfg.use_lr_scheduler,
            lr_scheduler_patience=cfg.lr_scheduler_patience, lr_scheduler_factor=cfg.lr_scheduler_factor,
            n_top=cfg.n_top, device=cfg.device, instrumentation=instrumentation,
        )
        critic_state = critic.state_dict()
    elif cfg.algo == "reinforce":
        actor = ActorReinforce(cfg.latent_dim, cfg.hidden_dim).to(cfg.device)
        opt = optim.Adam(actor.parameters(), lr=cfg.lr_reinforce)
        result = train_reinforce(
            actor, generator.decoder, reward_fn, opt,
            episodes=cfg.n_epochs, batch_size=cfg.batch_size,
            normalize=cfg.normalize, penalty_value=cfg.penalty_value,
            n_top=cfg.n_top, device=cfg.device, instrumentation=instrumentation,
        )
        critic_state = None
    else:
        raise ValueError(f"Unknown algo: {cfg.algo}")

    top_smiles, top_valid = _decode_top_latents(generator, result["best_latent_vectors"], greedy=cfg.greedy)
    top_smiles = [s if v else None for s, v in zip(top_smiles, top_valid)]

    history = {
        "average_obj_scores": result["average_obj_scores"],
        "loss_actor_list": result["loss_actor_list"],
    }
    if "loss_critic_list" in result:
        history["loss_critic_list"] = result["loss_critic_list"]

    save_run(
        run_dir=run_dir,
        config=cfg.as_dict(),
        history=history,
        top_smiles=top_smiles,
        top_rewards=result["best_rewards"],
        top_latents=result["best_latent_vectors"],
        actor_state=actor.state_dict(),
        critic_state=critic_state,
        extra_meta={"instrumentation": instrumentation.to_dict()},
    )

    return run_dir
