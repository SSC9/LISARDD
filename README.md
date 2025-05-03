# LISARDD
**BMI702 Final Project - MMSc BMI**

**Authors:** Valentin Badea, Shyam Chandra, John Lin

## Description 

This work studies the applicability of the MOLRL framework introduced in Haddad et al. (2025, [1]) to the generation of target-specific high-affinity binders. Following the original architecture in [1], we implement a Latent Reinforcement Learning PPO agent that learns the contours of high-scoring manifolds in the latent space of a target-agnostic molecule generator. We show that this strategy can be applied to the targeted generation of ligands with high affinity towards any given protein, while preserving other chemical properties, such as SA or QED through multi-objective reward optimization. 

> [1]: R. Haddad, E. E. Litsa, Z. Liu, X. Yu, D. Burkhardt, and G. Bhisetti. Targeted molecular
generation with latent reinforcement learning. Scientific Reports, 15(1):15202, Apr. 2025. ISSN
2045-2322. doi: 10.1038/s41598-025-99785-0. Publisher: Nature Publishing Group

## Architecture
> * **Molecule generator**: HierVAE (see: https://github.com/wengong-jin/hgraph2graph) a hierarchical molecular generative model using structural motifs. 
> * **Binding Affinity model**: MGraphDTA (see: https://github.com/guaguabujianle/MGraphDTA) a GNN for explainable Drug-Target affinity prediction. 
> * **PPO agent**: see [1] for implementation details. We reconstructed from the paper description, the actor-critic agent architecture.

Overall, this is a tentative architecture, based on what we perceived to be the best models at the time we completed this work. We encourage you to make use of our modular framework and adapt it to better-suited models. 

## Results

Rewards tested include SA, QED, MGraphDTA binding affinity and a multi-objective reward based on all the above. 

Over a 100-200 epochs, our RL framework shows significant improvements of all the above metrics, taken individually or all together in a multi-objective reward objective, suggesting that our PPO agent learns chemically relevant high-scoring manifolds in HierVAE molecular latent spaces. In particular, we found that switching from the simple maximization of the predicted binding affinity to the proportion of high-affinity binders within a batch helped the agent discover better ligand candidates with sensible chemical structures. 

## Repo structure: 

> * **data**: This folder contains the ChEMBL data used for HierVAE training. More importantly, it contains a vocabulary of structural motifs which are essential to decode molecules. 
> * **hgraph**: This folder contains the Python scripts necessary to define and run HierVAE. 
> * **vae_model**: This folder contains the weights of the trained molecular VAE. 
> * **model.py**: This Python scripts is necessary to define and run MGraphDTA. 
> * **score_model_weights**: This folder contains the weights of the scoring model. 
> * **LISARDD.ipynb** contains our main notebook with our PPO agent training and minimal visualizations of model performance. 
> * **Binding Affinity Scores Pipeline (Validation)** contains our validation pipeline with Vienna on two binding tasks (Streptavidin and CDK2). 