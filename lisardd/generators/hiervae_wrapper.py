from pathlib import Path
from types import SimpleNamespace

import torch

from hgraph import HierVAE, PairVocab, common_atom_vocab

from lisardd.decoding.safe_decode import safe_decode_batch


class HierVAEGenerator:
    def __init__(
        self,
        vocab_path: str | Path,
        ckpt_path: str | Path,
        device: str = "cuda",
        rnn_type: str = "LSTM",
        hidden_size: int = 250,
        embed_size: int = 250,
        latent_size: int = 32,
        depthT: int = 15,
        depthG: int = 15,
        diterT: int = 1,
        diterG: int = 3,
        dropout: float = 0.0,
    ):
        self.device = device
        self.latent_dim = latent_size

        with open(vocab_path) as f:
            pairs = [line.strip("\r\n ").split() for line in f]
        vocab = PairVocab(pairs, cuda=(device == "cuda"))

        args = SimpleNamespace(
            vocab=vocab,
            atom_vocab=common_atom_vocab,
            rnn_type=rnn_type,
            hidden_size=hidden_size,
            embed_size=embed_size,
            latent_size=latent_size,
            depthT=depthT,
            depthG=depthG,
            diterT=diterT,
            diterG=diterG,
            dropout=dropout,
        )

        model = HierVAE(args).to(device)
        state = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(state[0] if isinstance(state, (list, tuple)) else state)
        model.eval()

        self.vocab = vocab
        self.model = model
        self.decoder = model.decoder

    def sample_prior(self, n: int) -> torch.Tensor:
        return torch.randn(n, 3 * self.latent_dim, device=self.device)

    def decode(self, z: torch.Tensor, greedy: bool = True, max_decode_step: int = 150):
        z_chunks = torch.chunk(z, 3, dim=1)
        return safe_decode_batch(self.decoder, z_chunks, greedy=greedy, max_decode_step=max_decode_step)
