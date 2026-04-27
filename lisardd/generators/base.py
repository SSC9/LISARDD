"""
Generator contract.

Any class implementing the following surface area is a valid generator
for the LISARDD training loop. No abstract base class enforcement;
the contract is documented and duck-typed.

Required attributes:
    latent_dim: int
        Per-vector latent dimensionality. The full latent state passed
        to the actor has shape (batch, 3 * latent_dim).
    decoder
        The underlying decoder object, passed through to the training
        loop's `safe_decode_batch` calls.

Required methods:
    sample_prior(n: int) -> torch.Tensor
        Returns (n, 3 * latent_dim) tensor from the generator's prior
        (N(0, I) for HierVAE).
    decode(z: torch.Tensor, greedy: bool = True,
           max_decode_step: int = 150) -> tuple[list[str | None], list[bool]]
        Decodes (n, 3 * latent_dim) into SMILES, returning (smiles, valid).

To plug in a different generative model, write a wrapper class that
exposes these and pass it to run_experiment().
"""
