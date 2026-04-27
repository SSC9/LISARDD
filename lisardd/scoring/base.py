"""
Scoring contract.

Any class implementing the following surface area is a valid scorer
for the LISARDD reward functions.

Required attributes:
    target_protein: str
        Single-letter amino acid sequence of the protein target. Bound
        at scorer construction time so reward functions can be built
        against a fixed target.

Required methods:
    score(smiles_batch: list[str]) -> torch.Tensor
        Returns a (n,) tensor of predicted pKd values (higher = better
        binder). Reward functions wrap this output in normalization,
        binarization, or multi-objective combination.

To plug in a different scoring model, write a wrapper class that
exposes these and pass it to the reward factories in lisardd.rewards.
"""
