# Research Problem Statement & Theoretical Motivation

## 1. Background: The Promise and Bottleneck of V2V Cooperative Perception

Connected and Automated Vehicles (CAVs) have the potential to transcend single-agent perceptual boundaries by communicating via Vehicle-to-Vehicle (V2V) wireless links (5G-NR V2X, DSRC). Single-vehicle perception systems inevitably suffer from:
* **Severe Physical Occlusions:** Blind spots created by large obstacles such as trucks, buses, and urban buildings.
* **Limited Sensor Range:** Degraded signal-to-noise ratio and point sparsity beyond 40–50 meters.
* **Adverse Weather & Optical Degradations:** Rain droplets, optical glare, lens flare, and thermal noise.

Cooperative perception allows CAVs to share perceptual context, effectively "seeing through obstacles" and eliminating occlusions. However, the fundamental question that governs real-world cooperative perception is:

$$\mathbf{\text{What primitive should vehicles transmit to maximize detection accuracy while minimizing bandwidth?}}$$

---

## 2. The Failure of Existing Paradigms

Existing cooperative perception frameworks predominantly fall into two extremes, both of which face severe real-world failure modes:

```
                      COOPERATIVE PERCEPTION PARADIGMS
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         ▼                                                       ▼
   Paradigm 1: Intermediate BEV Maps                      Paradigm 2: Generic 3D Queries
   (Where2comm, JigsawComm, CoSDH)                        (CoopDETR, QUEST, SparseCoop)
         │                                                       │
         ├─► Transmits 200–600 KB/frame                          ├─► Compact tokens (~15 KB/frame)
         ├─► Wireless network congestion                         ├─► FAILS on cameras due to
         └─► High transmission latency (>200 ms)                 │   longitudinal depth ambiguity
                                                                 └─► Epipolar noise poisons cross-attention
```

### Paradigm 1: Intermediate Feature Sharing (The Bandwidth Trap)
State-of-the-art methods such as **Where2comm** (NeurIPS 2022), **CoSDH** (CVPR 2025), and **JigsawComm** (ECCV 2024) transmit Bird's-Eye-View (BEV) feature grids ($C \times H \times W$).
* **The Bandwidth Crisis:** Even when pruned with spatial confidence masks, transmitting intermediate feature maps consumes **200 to 600 KB per frame per vehicle**.
* At a standard 10 Hz perception rate across a modest platoon of 10 to 20 vehicles, intermediate fusion demands over **40 to 80 MB/second of continuous radio spectrum**.
* Under realistic 5G-NR V2X (PC5 Sidelink) or DSRC channels with limited 10–20 MHz channel allocations, intermediate feature transmission leads to severe packet drops, queue delays, and latency exceeding 200–300 ms—rendering cooperation useless for high-speed collision avoidance.

### Paradigm 2: Generic 3D Query Exchange (The Camera-Depth Trap)
To bypass the bandwidth bottleneck, sparse query-based methods (e.g. DETR-style 3D object tokens) represent objects as compact coordinate-feature pairs ($[x, y, z, w, l, h, \theta] + \mathbf{f} \in \mathbb{R}^{128}$). While this theoretically slashes payload volume to < 15 KB/frame, **it breaks down completely when applied to multi-camera setups**:

1. **The Longitudinal Ray-Depth Ambiguity Trap:**  
   In camera-only autonomous driving, projecting 2D camera pixels into 3D metric coordinates is an ill-posed inverse problem. A monocular or surround-camera feature may possess very high 2D classification confidence, yet its 3D depth estimate along the camera projection ray (the epipolar line) has high spatial variance ($\pm 5\text{ to }15\text{ meters}$).
   * When a collaborator CAV broadcasts queries based solely on 2D confidence, it transmits **depth-hallucinated queries**.
   * When the ego vehicle fuses these depth-corrupted queries via cross-attention, they act as **adversarial spatial noise**, corrupting the ego vehicle's local object predictions with phantom objects and misplaced bounding boxes.

2. **The "More-is-Worse" Feature Dilution Trap (The JigsawComm Dilemma):**  
   Recent theoretical work in collaborative perception (e.g., Wang et al., *JigsawComm*, ECCV 2024) proven that with standard attention/pooling fusion operators, adding more collaborator features often **dilutes** clean ego representations rather than reinforcing them.
   * If a collaborator transmits low-quality or spatially uncertain queries, standard multi-head attention averages them into the ego query set, actively degrading detection precision below that of a single vehicle.

---

## 3. The Research Hypothesis & Objectives

This research introduces **RayQuery**, centered around the following formal hypothesis:

> **Core Research Hypothesis:**  
> *In camera-only cooperative 3D perception, the communication utility of an object query is fundamentally determined by its **ray-depth certainty**, not merely its visual classification score. By formulating query utility as a joint function of classification confidence, categorical depth entropy, and receiver spatial demand, and by enforcing epistemic uncertainty gating during cross-vehicle attention, a multi-agent system can achieve superior 3D detection while transmitting **under 4 KB/frame** and mathematically eliminating the "more-is-worse" feature dilution trap.*

---

## 4. Key Scientific Questions Addressed

1. **How can a vehicle evaluate the 3D spatial reliability of its camera queries without ground truth?**  
   *Answer:* By measuring the Shannon entropy $\mathcal{H}_q$ of the categorical depth probability distribution along the query's camera projection ray.
2. **How can vehicles coordinate query exchange without incurring heavy two-way negotiation delay?**  
   *Answer:* By piggybacking an ultra-compact spatial demand bitmask (< 0.2 KB) onto standard 10 Hz Basic Safety Messages (SAE J2735 / ETSI CAM).
3. **How can cross-agent attention be safeguarded against depth-hallucinated collaborator noise?**  
   *Answer:* By introducing an Epistemic Anti-Dilution Gate that dynamically restricts cross-attention updates based on the relative geometric certainty between the ego and collaborator queries.
