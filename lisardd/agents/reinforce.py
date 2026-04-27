import time

import numpy as np
import torch
from tqdm import tqdm

from lisardd.decoding.safe_decode import safe_decode_batch


def train_reinforce(
    actor,
    decoder,
    reward_fn,
    optimizer,
    *,
    episodes: int = 100,
    batch_size: int = 64,
    normalize: bool = True,
    penalty_value: float = -1.0,
    n_top: int = 100,
    device: str = "cuda",
    instrumentation=None,
):
    average_obj_scores: list = []
    loss_actor_list: list = []
    best_latent_vectors: list = []
    best_rewards: list = []

    decoder.eval()
    actor.train()

    for episode in tqdm(range(episodes)):
        ep_t0 = time.time()

        z, log_probs = actor.sample(batch_size)
        z_chunks = torch.chunk(z, 3, dim=1)
        smiles_batch, valid = safe_decode_batch(decoder, z_chunks)

        valid_mask = torch.tensor(valid, device=device, dtype=torch.bool)
        valid_smiles = [s for s, v in zip(smiles_batch, valid) if v]
        n_invalid_ep = int((~valid_mask).sum().item())

        un_norm_rewards = torch.zeros(batch_size, device=device)
        un_norm_rewards[~valid_mask] = penalty_value
        if len(valid_smiles) > 0:
            un_norm_rewards[valid_mask] = reward_fn(valid_smiles)

        rewards = (un_norm_rewards - un_norm_rewards.mean()) / (un_norm_rewards.std() + 1e-8) if normalize else un_norm_rewards
        rewards = rewards.detach()

        loss = -torch.mean(log_probs * rewards)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        best_latent_vectors.extend(z.detach().tolist())
        best_rewards.extend(un_norm_rewards.detach().tolist())
        sorted_idx = np.argsort(best_rewards)
        best_latent_vectors = [best_latent_vectors[i] for i in sorted_idx][-n_top:]
        best_rewards = [best_rewards[i] for i in sorted_idx][-n_top:]

        avg = un_norm_rewards.mean().item()
        average_obj_scores.append(avg)
        loss_actor_list.append(loss.item())

        if instrumentation is not None:
            instrumentation.log_epoch(
                episode=episode,
                wall_time=time.time() - ep_t0,
                avg_reward=avg,
                loss_actor=loss.item(),
                loss_critic=None,
                n_invalid=n_invalid_ep,
                batch_size=batch_size,
            )

        if episode % 5 == 0:
            print(f"Episode {episode}: avg_reward={avg:.4f}, loss={loss.item():.4f}")

    return {
        "average_obj_scores": average_obj_scores,
        "loss_actor_list": loss_actor_list,
        "best_latent_vectors": best_latent_vectors,
        "best_rewards": best_rewards,
    }
