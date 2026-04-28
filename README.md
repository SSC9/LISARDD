# LISARDD

**Ligand Iterative Sampling for Affinity Refinement and Drug Discovery**

Reinforcement-learning framework for optimizing sampling in the latent space of a pretrained, target-agnostic molecular generative model. Generates candidate molecules that simultaneously optimize multiple drug properties — target-specific binding affinity, drug-likeness (QED), and synthetic accessibility (SA) — through multi-objective reward composition.

**Authors:** Valentin Badea, Shyam Chandra, John Lin (Department of Biomedical Informatics, Harvard Medical School)

**Paper:** *Ligand Iterative Sampling for Affinity Refinement and Drug Discovery (LISARDD)*, Workshop on Generative AI for Biology, ICML 2025. PMLR 267.

## Why this exists

Targeted de novo drug generation typically requires either expensive conditional retraining or hand-crafted scoring heuristics. LISARDD frames generation as a reinforcement-learning problem: an agent learns to perturb latent vectors of a frozen, target-agnostic generator (HierVAE) to maximize a reward function. Target specificity adapts at training time — no retraining the generator. The framework is fully modular: any generator, scoring model, or RL algorithm can plug in as long as it implements the documented contracts in `lisardd/generators/base.py` and `lisardd/scoring/base.py`.

We evaluate two RL algorithms (PPO and REINFORCE) on two protein targets (human JNK3 and *E. coli* gyrA) using a Davis-trained MGraphDTA scoring model, with external validation through AutoDock Vina docking.

## Quick start

```bash
git clone https://github.com/SSC9/LISARDD
cd LISARDD
pixi install
pixi run smoke         # ~3-5 min end-to-end wiring check
pixi run train         # one full training run; edit notebooks/01_train.ipynb config first
```

Or use the notebooks directly in JupyterLab after `pixi shell`:

| Notebook | Purpose |
|---|---|
| `notebooks/00_smoke.ipynb` | 3–5 min end-to-end wiring test (smoke mode: 5 epochs, batch=8). |
| `notebooks/00_regression.ipynb` | Loads a camera-ready pickle and replays its actor through the cleaned pipeline; verifies architecture/protocol equivalence. |
| `notebooks/01_train.ipynb` | One full training run, configurable at the top. Saves to `runs/<run_name>/`. |

## Architecture

| Component | Implementation |
|---|---|
| Generator | HierVAE (Jin et al. 2020) — a hierarchical motif-based VAE — pretrained on ChEMBL. Imported from the upstream [`hgraph2graph`](https://github.com/wengong-jin/hgraph2graph) codebase. Vocab is `recovered_vocab_2000.txt` from [bsaldivaremc2's fork](https://github.com/bsaldivaremc2/hgraph2graph), addressing [issue #47](https://github.com/wengong-jin/hgraph2graph/issues/47). Wrapped in `lisardd.generators.HierVAEGenerator`. |
| Scorer | MGraphDTA (Yang et al. 2022) — a deep multiscale graph neural network for drug-target binding affinity. Trained on Davis. Wrapped in `lisardd.scoring.MGraphDTAScorer`. Upstream codebase: [`MGraphDTA`](https://github.com/guaguabujianle/MGraphDTA). |
| Agents | Actor-critic PPO with clipped surrogate loss (γ=0.95); REINFORCE with learnable Gaussian policy. Networks are 3-layer MLPs (BatchNorm + ReLU). See `lisardd.agents`. |
| Rewards | QED, SA, raw pKd, binarized-and-differentiable pKd, multi-objective composition (default w₁=w₂=0.1). See `lisardd.rewards`. |
| Targets | JNK3 (PDB 3FI2 chain A, kinase domain) and *E. coli* gyrA (PDB 1AB4 chain A, 59 kDa N-terminal fragment containing the quinolone pocket). Sequences in `lisardd.targets`. |
| Validation | AutoDock Vina with site-targeted box derived from holo-PDB bound-ligand centroid. See `lisardd.validation.vina` (Stage 6 — being built out). |

## Repo layout

```
LISARDD/
├── lisardd/                                  # importable package
│   ├── agents/         # Actor, Critic, train_ppo, train_reinforce
│   ├── generators/     # HierVAEGenerator + base contract
│   ├── scoring/        # MGraphDTAScorer + base contract
│   ├── decoding/       # safe_decode_batch (try/batch -> per-sample fallback)
│   ├── rewards.py      # paper-aligned reward factories
│   ├── targets.py      # JNK3, gyrA sequences
│   ├── config.py       # ExperimentConfig dataclass
│   ├── runner.py       # run_experiment(config) entrypoint
│   ├── io.py           # save_run / load_run / load_legacy_pickle
│   ├── analyze.py      # plot_ppo_vs_reinforce, paired t-test
│   ├── instrumentation.py
│   └── validation/     # Vina site-targeted docking pipeline
├── notebooks/          # 00_smoke, 00_regression, 01_train (+ 02, 03 in follow-up)
├── runs/               # gitignored; one subdirectory per training run
├── scripts/            # batch runners (added in follow-up)
├── data/               # ChEMBL training data + recovered_vocab_2000.txt
├── hgraph/             # HierVAE source (upstream)
├── vae_model/          # HierVAE checkpoint
├── score_model.py      # MGraphDTA architecture (upstream)
├── score_model_weights/# MGraphDTA checkpoint
├── ICML_2025_Workshop_Submission_Artifacts/  # camera-ready provenance
├── pixi.toml           # cross-platform reproducible environment
└── README.md
```

## Modularity contract

Plug in a different generative model by writing a class with these attributes/methods (no abstract base class enforcement — duck typing):

```python
class MyGenerator:
    latent_dim: int
    decoder                      # passed to safe_decode_batch
    def sample_prior(self, n) -> Tensor   # (n, 3*latent_dim)
    def decode(self, z, greedy, max_decode_step) -> tuple[list[str|None], list[bool]]
```

Same for scorers:

```python
class MyScorer:
    target_protein: str
    def score(self, smiles: list[str]) -> Tensor   # (n,) predicted pKd
```

That's the entire interface. See `lisardd/generators/hiervae_wrapper.py` and `lisardd/scoring/mgraphdta_wrapper.py` for reference implementations.

## Known limitations

- **Davis dataset coverage.** MGraphDTA was trained on Davis (442 kinase proteins × 68 ligands). Predictions for kinase targets such as JNK3 are in-distribution. Predictions for non-kinase targets such as *E. coli* gyrA (a topoisomerase) are out-of-distribution; pKd values for gyrA should be interpreted accordingly. AutoDock Vina external validation partially mitigates. Future work: retrain MGraphDTA on a more diverse dataset (BindingDB, KIBA) for better cross-family generalization. The modular `MGraphDTAScorer` accepts any compatible checkpoint via `ckpt_path`, so a swap is one line of config.

- **Per-epoch slowdown during PPO training.** As the policy drifts the latent distribution off N(0, I), HierVAE's decoder raises KeyError on unseen (motif, anchor) vocab pairs at increasing rate. The current `safe_decode_batch` catches the exception and falls back to per-sample decoding, which is correct but quadratically slow as failure rate climbs. A multiprocessing decode pool addressing this is in the planned follow-up. Not a vocab insufficiency issue — the recovered_vocab_2000.txt fix is already applied.

- **Vina box derivation.** The post-submission Vina pipeline standardizes on a holo-PDB-derived ligand-centroid box (vs. the camera-ready 20³Å box at coordinates [8.56, 35.87, 12.02] applied to the JNK3 apo PDB). Stage 6 work in progress.

- **PPO reward target update vs camera-ready.** The cleaned PPO implementation (`lisardd/agents/ppo.py`) decodes the post-action state `s_{t+1}` for reward computation; the camera-ready code decoded the pre-action state `s_t`. This aligns the reward with the standard TD credit-assignment form. The empirical effect on the PPO vs REINFORCE comparison is being evaluated; until the new 8-run sweep completes, treat trajectories from the cleaned pipeline as not directly comparable to the camera-ready Figure 3 PPO curves. See `ICML_2025_Workshop_Submission_Artifacts/README.md` for the full delta.

## Citation

```bibtex
@inproceedings{badea2025lisardd,
  title={Ligand Iterative Sampling for Affinity Refinement and Drug Discovery (LISARDD)},
  author={Badea, Valentin and Chandra, Shyam and Lin, John},
  booktitle={Workshop on Generative AI for Biology at the 42nd International Conference on Machine Learning},
  series={Proceedings of Machine Learning Research},
  volume={267},
  year={2025},
}
```

## References

The LISARDD framework builds on:

```bibtex
@inproceedings{jin2020hiervae,
  title={Hierarchical Generation of Molecular Graphs using Structural Motifs},
  author={Jin, Wengong and Barzilay, Regina and Jaakkola, Tommi},
  booktitle={Proceedings of the 37th International Conference on Machine Learning},
  year={2020},
}

@article{yang2022mgraphdta,
  title={MGraphDTA: deep multiscale graph neural network for explainable drug-target binding affinity prediction},
  author={Yang, Ziduo and Zhong, Weihe and Zhao, Lu and Chen, Calvin Yu-Chian},
  journal={Chemical Science},
  volume={13},
  number={3},
  pages={816--833},
  year={2022},
}

@article{haddad2025molrl,
  title={Targeted molecular generation with latent reinforcement learning},
  author={Haddad, R. and Litsa, E. E. and Liu, Z. and Yu, X. and Burkhardt, D. and Bhisetti, G.},
  journal={Scientific Reports},
  volume={15},
  number={1},
  pages={15202},
  year={2025},
}

@article{trott2010vina,
  title={AutoDock Vina: improving the speed and accuracy of docking with a new scoring function, efficient optimization, and multithreading},
  author={Trott, Oleg and Olson, Arthur J.},
  journal={Journal of Computational Chemistry},
  volume={31},
  number={2},
  pages={455--461},
  year={2010},
}
```

## License

MIT — see [LICENSE](./LICENSE). Copyright (c) 2025 Valentin Badea, Shyam Chandra, John Lin.
