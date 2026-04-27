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

Each pickle stores actor (and critic, for PPO) state dicts plus reward histories and top-100 hits. They can be loaded with `lisardd.io.load_legacy_pickle()`, which returns a `RunArtifacts` object compatible with the analysis pipeline in `notebooks/02_analyze.ipynb`.

The trained scoring model and target sequences referenced here used a JNK3 sequence whose docking-target alignment is being updated as part of the post-submission Vina validation refresh; the new pipeline standardizes on PDB-derived domain sequences. Predicted pKd values from the new pipeline will differ slightly from the values stored here.
