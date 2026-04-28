import time

import numpy as np
import torch
from torch.distributions import MultivariateNormal
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from lisardd.decoding.safe_decode import safe_decode_batch


def train_ppo(
    actor,
    critic,
    decoder,
    reward_fn,
    optimizer,
    *,
    episodes: int = 100,
    batch_size: int = 64,
    t_steps: int = 1,
    ppo_epochs: int = 6,
    c1: float = 0.5,
    c2: float = 0.01,
    normalize: bool = True,
    penalty_value: float = -1.0,
    clip_eps: float = 0.2,
    gamma: float = 0.95,
    latent_dim: int = 32,
    greedy: bool = True,
    use_lr_scheduler: bool = False,
    lr_scheduler_patience: int = 10,
    lr_scheduler_factor: float = 0.5,
    n_top: int = 100,
    device: str = "cuda",
    instrumentation=None,
):
    best_latent_vectors: list = []
    best_rewards: list = []
    average_obj_scores: list = []
    loss_actor_list: list = []
    loss_critic_list: list = []

    decoder.eval()
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=lr_scheduler_patience, factor=lr_scheduler_factor) if use_lr_scheduler else None

    for episode in tqdm(range(episodes)):
        ep_t0 = time.time()
        actor.eval()
        critic.eval()

        s_prev = torch.randn(batch_size, 3 * latent_dim, device=device)

        rewards_t = torch.zeros((batch_size, t_steps), device=device)
        adv_t = torch.zeros((batch_size, t_steps), device=device)
        rewards_to_go_t = torch.zeros((batch_size, t_steps), device=device)
        v_phi = torch.zeros((batch_size, t_steps + 1), device=device)
        old_log_probs_t = torch.zeros((batch_size, t_steps + 1), device=device)
        actions_t = torch.zeros((batch_size, 3 * latent_dim, t_steps + 1), device=device)
        states_t = torch.zeros((batch_size, 3 * latent_dim, t_steps + 1), device=device)

        n_invalid_ep = 0

        for t in range(t_steps + 1):
            states_t[:, :, t] = s_prev
            with torch.no_grad():
                mu, std = actor(s_prev)

            dist = MultivariateNormal(mu, torch.diag_embed(std))
            a_t = dist.sample()
            actions_t[:, :, t] = a_t
            old_log_probs_t[:, t] = dist.log_prob(a_t)
            s_t_next = s_prev + a_t
            v_phi[:, t] = critic(s_prev).squeeze()  # V(s_t)

            # Decode the post-action state s_{t+1} so the reward is computed for the
            # latent the action produced. The camera-ready code decoded s_prev (= s_t);
            # this update aligns r_t with r(s_{t+1}), matching the standard TD form
            # A_t = r_{t+1} + gamma V(s_{t+1}) - V(s_t). Whether this materially
            # changes the PPO vs REINFORCE comparison is being evaluated empirically.
            s_next_chunks = torch.chunk(s_t_next, 3, dim=1)
            smiles_batch, valid = safe_decode_batch(decoder, s_next_chunks, greedy=greedy)
            valid_smiles = [s for s, v in zip(smiles_batch, valid) if v]
            valid_mask = torch.tensor(valid, device=device, dtype=torch.bool)
            n_invalid_ep += int((~valid_mask).sum().item())

            un_norm_rewards = torch.zeros(batch_size, device=device)
            un_norm_rewards[~valid_mask] = penalty_value
            if len(valid_smiles) > 0:
                un_norm_rewards[valid_mask] = reward_fn(valid_smiles)
            un_norm_rewards = un_norm_rewards.detach()

            rewards = (un_norm_rewards - un_norm_rewards.mean()) / (un_norm_rewards.std() + 1e-8) if normalize else un_norm_rewards
            if t < t_steps:
                rewards_t[:, t] = rewards
            s_prev = s_t_next

        for t in range(t_steps):
            adv_t[:, t] = (rewards_t[:, t] + gamma * v_phi[:, t + 1] - v_phi[:, t]).detach()

        rewards_t = rewards_t.detach()
        rewards_to_go_t[:, t_steps - 1] = rewards_t[:, t_steps - 1]
        for t in range(t_steps - 2, -1, -1):
            rewards_to_go_t[:, t] = rewards_t[:, t] + gamma * rewards_to_go_t[:, t + 1]

        actor.train()
        critic.train()

        for _ in range(ppo_epochs):
            loss_actor = 0
            loss_critic = 0
            loss_entropy = 0
            for t in range(t_steps):
                new_mu, new_std = actor(states_t[:, :, t])
                new_dist = MultivariateNormal(new_mu, torch.diag_embed(new_std))
                new_log_probs_t = new_dist.log_prob(actions_t[:, :, t])
                ratio = torch.exp(new_log_probs_t - old_log_probs_t[:, t])

                surr1 = ratio * adv_t[:, t]
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t[:, t]
                loss_actor += -torch.min(surr1, surr2).mean()

                loss_entropy += new_dist.entropy().mean()

                v_phi_new = critic(states_t[:, :, t]).squeeze()
                loss_critic += (rewards_to_go_t[:, t] - v_phi_new).pow(2).mean()

            loss_actor /= t_steps
            loss_critic /= t_steps
            loss_entropy /= t_steps

            loss_tot = loss_actor + c1 * loss_critic - c2 * loss_entropy

            optimizer.zero_grad()
            loss_tot.backward()
            optimizer.step()

        avg_obj_score = un_norm_rewards.mean().item()
        if scheduler is not None:
            scheduler.step(avg_obj_score)

        loss_actor_list.append(loss_actor.item())
        loss_critic_list.append(loss_critic.item())
        average_obj_scores.append(avg_obj_score)

        # Pair the last decoded latent with un_norm_rewards from the same iteration.
        # With the post-action decode update above, un_norm_rewards now corresponds
        # to s_prev (post-loop = final s_t_next), not states_t[:, :, t_steps].
        best_latent_vectors.extend(s_prev.detach().tolist())
        best_rewards.extend(un_norm_rewards.detach().tolist())
        sorted_idx = np.argsort(best_rewards)
        best_latent_vectors = [best_latent_vectors[i] for i in sorted_idx][-n_top:]
        best_rewards = [best_rewards[i] for i in sorted_idx][-n_top:]

        if instrumentation is not None:
            instrumentation.log_epoch(
                episode=episode,
                wall_time=time.time() - ep_t0,
                avg_reward=avg_obj_score,
                loss_actor=loss_actor.item(),
                loss_critic=loss_critic.item(),
                n_invalid=n_invalid_ep,
                batch_size=batch_size,
            )

        if episode % 5 == 0:
            print(f"Episode {episode}: loss_actor={loss_actor.item():.4f}, loss_critic={loss_critic.item():.4f}, avg_reward={avg_obj_score:.4f}")

    return {
        "average_obj_scores": average_obj_scores,
        "loss_actor_list": loss_actor_list,
        "loss_critic_list": loss_critic_list,
        "best_latent_vectors": best_latent_vectors,
        "best_rewards": best_rewards,
    }
