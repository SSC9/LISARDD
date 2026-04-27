# ICML 2025 Workshop Submission Artifacts

Provenance directory for the artifacts produced during the camera-ready submission to *Generative AI for Biology* at ICML 2025 (LISARDD: Ligand Iterative Sampling for Affinity Refinement and Drug Discovery).

These files reflect the state of the project at the time of submission and are preserved for reproducibility audits. They are **not** consumed by the cleaned `lisardd/` package — that package produces equivalent outputs in `runs/`.

## Layout

| Subdirectory | Contents |
|---|---|
| `pickles/` | Per-run training artifacts. Filename convention: `{algo}_{reward}_{target}_normalize.pkl`. Twelve files cover `{ppo, reinforce}` × `{binding_jnk3, binding_egyrase, multi_jnk3, multi_egyrase, qed, sa}`. Note: `ppo_binding_jkn3_normalize.pkl` retains a typo in the original filename. |
| `csv_outputs/` | Top-100 SMILES tables and 25-molecule sampled-from-actor CSVs. |
| `figures/` | Camera-ready figures (Google Slides exports). |
| `vina/` | Receptor PDB used for AutoDock Vina validation (`3fi2_unbound.pdb`) and the validation notebook. |
| `historical_notebooks/` | Pre-camera-ready development notebooks, kept for traceability. |
| `historical_validation/` | Earlier-iteration validation pipeline targeting Streptavidin and CDK2, prior to the JNK3 / gyrA target selection. |

## Reproducing camera-ready behavior

Each pickle stores actor (and critic, for PPO) state dicts plus reward histories and top-100 hits. They can be loaded with `lisardd.io.load_legacy_pickle()`, which returns the raw run dict; `notebooks/00_regression.ipynb` replays the actor through the cleaned pipeline using the camera-ready sequence and confirms architecture equivalence.

## Sequence and reward updates in the cleaned pipeline

As part of the post-submission Vina validation refresh, the cleaned pipeline aligns target sequences with the deposited holo PDB structures used for docking, and switches the binding-only reward to the bounded sigmoid formulation that the workshop paper's discussion advocates. New runs in `runs/` will produce trajectories that are not directly comparable to the camera-ready trajectories preserved here.

| Component | Camera-ready (this directory) | Cleaned pipeline (`runs/`) |
|---|---|---|
| JNK3 sequence | 423 aa custom range | 353 aa, PDB 3FI2 chain A (kinase domain, with 6 'X' for disordered loop) |
| gyrA sequence | 875 aa, full UniProt P0AES4 | 493 aa, PDB 1AB4 chain A (59 kDa N-terminal fragment containing the quinolone pocket) |
| Binding-only reward | `reward_binding_affinity` (raw pKd, unbounded) | `reward_prop_high_binders_diff` (sigmoid-binarized to [0, 1] in the [7, 14] pKd window) |
| Multi-objective reward | `reward_multi_obj` (uses `prop_high_binders_diff` for the pKd term) | unchanged |

Both sequence sets remain accessible from `lisardd.targets`: `get_sequence(name)` returns the cleaned-pipeline sequence; `get_legacy_sequence(name)` returns the camera-ready sequence. The regression notebook uses the latter to verify actor-architecture compatibility against the camera-ready pickles archived here.
