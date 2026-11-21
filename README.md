# RayQuery: Depth-Uncertainty & Demand-Guided Query Exchange for Camera-Only Cooperative 3D Detection

A communication-efficient, anti-dilution collaborative perception architecture for Connected Autonomous Vehicles (V2V), evaluated on the **OPV2V** benchmark.

---

## 1. Problem Statement & Research Hypothesis

In multi-agent collaborative perception (V2X), sharing intermediate Bird's-Eye-View (BEV) feature grids (e.g., Where2comm, JigsawComm, CoSDH) consumes **200–600 KB/frame**, creating severe wireless communication bottlenecks.

While 3D object query exchange (e.g., DETR-style tokens) can theoretically reduce bandwidth to under 20 KB/frame, **monocular/surround camera queries suffer from severe ray-depth ambiguity**. Single-view depth projections have high variance along epipolar lines:
1. **Depth-Hallucination Trap:** Collaborators transmit queries that appear confident in 2D but are spatially misplaced in 3D by 5–15 meters, poisoning the ego vehicle's cross-attention.
2. **Dilution / More-is-Worse Trap:** Standard query cross-attention dilutes high-precision ego queries with lower-quality collaborator queries.

---

## 2. Key Contributions

1. **Ray-Depth Uncertainty-Aware Query Selection:**
   Every candidate 3D query is grounded in its underlying multi-camera categorical depth distribution entropy $\mathcal{H}_q$. The transmission utility penalizes ray-depth ambiguity:
   $$\mathcal{U}(q) = \text{Score}(q) \cdot \exp\left(-\alpha \cdot \mathcal{H}_q\right)$$
2. **Asymmetric Demand-Guided Requesting:**
   The ego vehicle issues a compact spatial demand field representing occlusions and blind cones. Collaborators only transmit queries that intersect the ego's demand field.
3. **Epistemic Anti-Dilution Cross-Attention:**
   Cross-vehicle attention weights are dynamically constrained by the collaborator's relative geometric certainty. If the collaborator's query is more certain than the ego's, it is absorbed; otherwise, it is suppressed—preventing feature dilution.
4. **Honest Ultra-Low Bandwidth Accounting:**
   With 8-bit quantized query features and coordinate packing, transmission consumes only **8–15 KB/frame** (over 95% reduction compared to BEV feature sharing).

---

## 3. Repository Structure

```
c:\PhD\coperception\
├── configs/
│   └── config.py                 # Central hyperparameter configuration
├── src/
│   ├── data/
│   │   └── opv2v_dataset.py      # OPV2V loader (links to test_culver_city)
│   ├── models/
│   │   ├── camera_backbone.py    # Multi-camera feature & depth entropy head
│   │   ├── query_generator.py    # Ray-uncertainty 3D query generator
│   │   ├── demand_supply_selector.py # Demand matching & anti-dilution cross-attention
│   │   └── rayquery_model.py     # End-to-end cooperative detector
│   └── utils/
│       └── metrics.py            # AP@0.3, AP@0.5, AP@0.7 evaluation metrics
├── scripts/
│   └── verify_pipeline.py        # Complete pipeline verification script
└── slurm/
    └── train_cluster.sh          # Cluster launch script for Kelvin2 HPC
```

---

## 4. Quickstart

To run the pipeline verification on the local OPV2V Culver City dataset:

```bash
python scripts/verify_pipeline.py
```
