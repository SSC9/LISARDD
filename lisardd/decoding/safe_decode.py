"""
Safe decoding wrapper for HierVAE.

HierVAE's decoder can raise KeyError mid-batch when the latent vectors
drift off the manifold of (motif, anchor) pairs covered by the vocab.
The wrapper attempts a batch decode first; on failure, it falls back
to per-sample decoding so that one bad sample does not lose the rest
of the batch. As policy training drifts the latent distribution, the
exception rate climbs and the per-sample fallback dominates wall-time.
A multiprocessing decode pool addresses this in a follow-up.
"""

from typing import Sequence

import torch


def decode_smiles_from_latent(decoder, z_chunks, greedy: bool = True, max_decode_step: int = 150):
    return decoder.decode(z_chunks, greedy=greedy, max_decode_step=max_decode_step)


def safe_decode_batch(decoder, z_chunks: Sequence[torch.Tensor], greedy: bool = True, max_decode_step: int = 150):
    batch_size = z_chunks[0].shape[0]
    try:
        smiles = decode_smiles_from_latent(decoder, z_chunks, greedy=greedy, max_decode_step=max_decode_step)
        valid = [True] * len(smiles)
        return smiles, valid
    except Exception:
        smiles, valid = [], []
        for i in range(batch_size):
            single = tuple(c[i].unsqueeze(0) for c in z_chunks)
            try:
                s = decode_smiles_from_latent(decoder, single, greedy=greedy, max_decode_step=max_decode_step)[0]
                smiles.append(s)
                valid.append(True)
            except Exception:
                smiles.append(None)
                valid.append(False)
        return smiles, valid
