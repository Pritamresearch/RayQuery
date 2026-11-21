# RayQuery: Methodology & Architecture Walkthrough

This document provides a comprehensive technical walkthrough of **RayQuery**, detailing the mathematical formulations, network components, and algorithms that enable ultra-low-bandwidth, anti-dilution collaborative 3D detection.

---

## 1. System Pipeline Overview

```
[4 Surround RGB Cameras] ────────► [Pretrained ResNet-18 Backbone]
                                            │
                                            ▼
                       [Categorical Depth Prob Head (50 Bins)]
                                            │
                                            ├─► [Feature Map F ∈ R^(128 x H/8 x W/8)]
                                            └─► [Depth Entropy Map H ∈ R^(1 x H/8 x W/8)]
                                            │
                                            ▼
                           [3D Query Generator (100 Anchors)]
                                            │
                                            ├─► [Query Embeddings Q ∈ R^(100 x 128)]
                                            ├─► [3D Proposal Bounding Boxes B ∈ R^(100 x 7)]
                                            ├─► [Classification Scores S ∈ [0, 1]]
                                            └─► [Ray-Depth Uncertainty H_q ∈ [0, 1]]
                                            │
                                            ▼
                           [Demand-Supply Utility Scorer]
                           U_q = Score · exp(-α·H_q) · D_q
                                            │
                                            ▼
                           [Top-K Sparse Transmission]
                           (~3.12 KB / frame @ Top-20)
                                            │
                                            ▼  (5G-NR V2X Wireless Sidelink)
                           [Ego Coordinate Transformation]
                           p_ego = R_rel · p_collab + T_rel
                                            │
                                            ▼
                           [Epistemic Anti-Dilution Cross-Attention]
                           G = Sigmoid(MLP([Cert_ego, Cert_collab]))
                                            │
                                            ▼
                           [Final 3D Detection Head]
                           (Focal Loss + Smooth L1 3D Regression)
```

---

## 2. Multi-Camera Feature & Depth Distribution Extraction

Each Connected Automated Vehicle (CAV) $i$ is equipped with $N_{\text{cam}} = 4$ surround cameras providing a $360^\circ$ horizontal field of view.

### Pretrained ResNet-18 Feature Extraction
Images $\mathbf{I}_i \in \mathbb{R}^{N_{\text{cam}} \times 3 \times H \times W}$ are passed through an ImageNet-pretrained ResNet-18 feature extractor up to `layer2` (stride 8):
$$\mathbf{X}_i = \text{ResNet18}_{\text{layer2}}(\mathbf{I}_i) \in \mathbb{R}^{N_{\text{cam}} \times 128 \times \frac{H}{8} \times \frac{W}{8}}$$
An adaptation neck projects these into task-specific visual features $\mathbf{F}_i \in \mathbb{R}^{N_{\text{cam}} \times C \times \frac{H}{8} \times \frac{W}{8}}$ where $C = 128$.

### Categorical Depth Probability & Ray Entropy
A parallel $1 \times 1$ convolutional depth head discretizes the camera optical ray into $D = 50$ depth bins ($d_{\min} = 2.0\text{ m}$ to $d_{\max} = 52.0\text{ m}$, resolution $\Delta d = 1.0\text{ m}$):
$$\mathbf{P}_i(d \mid u, v) = \text{Softmax}\left(\text{Conv}_{1\times 1}(\mathbf{X}_i)\right) \in \mathbb{R}^{N_{\text{cam}} \times D \times \frac{H}{8} \times \frac{W}{8}}$$
To quantify the longitudinal ambiguity along each projection ray $(u, v)$, we compute the normalized Shannon depth entropy:
$$\mathcal{H}_i(u, v) = \frac{-\sum_{d=1}^{D} \mathbf{P}_i(d \mid u, v) \ln \left(\mathbf{P}_i(d \mid u, v) + \epsilon\right)}{\ln(D)} \in [0, 1]$$
* **$\mathcal{H}_i(u, v) \approx 0.0$:** Distinct geometric surface; depth is highly reliable and deterministic.
* **$\mathcal{H}_i(u, v) \approx 1.0$:** Optical glare, wet surface, or textureless sky; depth is completely smeared and ambiguous.

---

## 3. Ray-Uncertainty 3D Query Generation

Each agent maintains a set of $M = 100$ learnable 3D reference query anchors $\mathbf{p}_k = [x_k, y_k, z_k]$ uniformly distributed across the region of interest ($[-50\text{ m}, 50\text{ m}] \times [-50\text{ m}, 50\text{ m}]$).

### Camera Projection & Feature Sampling
Using the camera intrinsic matrix $\mathbf{K}_c$ and extrinsic matrix $\mathbf{T}_{\text{cam}\leftarrow\text{ego}}^c$, each 3D point $\mathbf{p}_k$ is projected onto the camera image plane:
$$\mathbf{p}_{\text{cam}}^c = \mathbf{T}_{\text{cam}\leftarrow\text{ego}}^c \begin{bmatrix} \mathbf{p}_k \\ 1 \end{bmatrix},\quad \begin{bmatrix} u_k^c \\ v_k^c \\ 1 \end{bmatrix} = \frac{1}{Z_k^c} \mathbf{K}_c \mathbf{p}_{\text{cam}, :3}^c$$
Visual features and depth entropy are bilinearly sampled across valid camera frustums:
$$\mathbf{f}_k = \sum_{c=1}^{N_{\text{cam}}} \mathbb{I}_{\text{FoV}}(u_k^c, v_k^c) \cdot \text{Sample}(\mathbf{F}_i^c, u_k^c, v_k^c)$$
$$\mathcal{H}_q(k) = \min_{c \in \text{FoV}} \text{Sample}(\mathcal{H}_i^c, u_k^c, v_k^c)$$
Taking the minimum entropy across overlapping camera views assigns the query the certainty of its best, most un-occluded viewpoint.

---

## 4. Demand–Supply Query Selection

To prevent the transmission of redundant or depth-ambiguous queries, each collaborator scores its queries before transmission:

$$\mathcal{U}(q_k) = \text{Score}(q_k) \cdot \exp\left(-\alpha \cdot \mathcal{H}_q(k)\right) \cdot \mathcal{D}(q_k)$$

* $\text{Score}(q_k) \in [0, 1]$: Local classification confidence predicted by the proposal head.
* $\exp(-\alpha \cdot \mathcal{H}_q(k))$: Ray-depth certainty gate ($\alpha = 3.0$). Queries with high depth entropy have their utility crushed toward $0$.
* $\mathcal{D}(q_k)$: Spatial demand factor indicating whether the query falls inside the ego vehicle's requested occlusion cone.

Only the **Top-$K$ queries** ($K=20$) maximizing $\mathcal{U}(q_k)$ are transmitted over the V2X link.

### Honest Byte Accounting per Transmitted Query:
| Field | Data Type | Payload Size |
| :--- | :--- | :---: |
| 3D Box Coordinates $(x, y, z, l, w, h, \theta)$ | $7 \times \text{float32}$ | 28 bytes |
| Quantized Query Feature Embedding | $128 \times \text{int8}$ | 128 bytes |
| Ray-Depth Uncertainty Scalar $\mathcal{H}_q$ | $1 \times \text{float32}$ | 4 bytes |
| **Total per Query** | — | **160 bytes** |

For $K=20$ transmitted queries, total transmission payload is:
$$\text{Bandwidth} = 20 \times 160\text{ bytes} = \mathbf{3.12\text{ KB/frame}}$$

---

## 5. Epistemic Anti-Dilution Cross-Attention

When collaborator queries arrive at the ego vehicle, they are transformed into the ego coordinate frame via relative pose:
$$\mathbf{p}_{\text{ego}} = \mathbf{R}_{\text{rel}} \mathbf{p}_{\text{collab}} + \mathbf{T}_{\text{rel}},\quad \theta_{\text{ego}} = \theta_{\text{collab}} + \Delta\theta_{\text{rel}}$$

To prevent low-quality collaborator queries from diluting clean ego queries, cross-attention is gated by **Epistemic Certainty**:

$$\mathbf{A}_{\text{out}} = \text{MultiHeadAttention}(\mathbf{Q}_{\text{ego}}, \mathbf{K}_{\text{collab}}, \mathbf{V}_{\text{collab}})$$
$$\mathbf{G} = \sigma\left(\text{MLP}\left(\left[(1 - \mathcal{H}_{\text{ego}}), (1 - \bar{\mathcal{H}}_{\text{collab}})\right]\right)\right) \in [0, 1]$$
$$\mathbf{Q}_{\text{fused}} = \text{LayerNorm}\left(\mathbf{Q}_{\text{ego}} + \mathbf{G} \odot \mathbf{A}_{\text{out}}\right)$$

* If the collaborator is significantly more geometrically certain ($1 - \bar{\mathcal{H}}_{\text{collab}} \gg 1 - \mathcal{H}_{\text{ego}}$), the gate $\mathbf{G} \to 1.0$, absorbing the collaborator's observation.
* If the ego is already confident or the collaborator is noisy, $\mathbf{G} \to 0.0$, suppressing the update and mathematically preventing the dilution trap.

---

## 6. End-to-End Multi-Task Loss Formulation

The full model is supervised end-to-end via a multi-component loss:

$$\mathcal{L}_{\text{total}} = w_{\text{cls}} \mathcal{L}_{\text{cls}} + w_{\text{reg}} \mathcal{L}_{\text{reg}} + \lambda_{\text{comm}} \mathcal{L}_{\text{comm}}$$

1. **Bipartite / Spatial 3D Nearest-Neighbor Matching:**  
   Each ground-truth 3D box $g_j$ is matched to the closest predicted query anchor $q_i$ within distance threshold $d_{\text{thresh}} = 4.0\text{ m}$.
2. **Focal Classification Loss ($\mathcal{L}_{\text{cls}}$):**
   $$\mathcal{L}_{\text{cls}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
3. **Smooth L1 3D Regression Loss ($\mathcal{L}_{\text{reg}}$):**
   $$\mathcal{L}_{\text{reg}} = \text{SmoothL1}(\mathbf{c}_{\text{pred}} - \mathbf{c}_{\text{gt}}) + \text{SmoothL1}(\log \mathbf{s}_{\text{pred}} - \log \mathbf{s}_{\text{gt}}) + |\sin(\theta_{\text{pred}} - \theta_{\text{gt}})|$$
4. **Communication Regularization ($\mathcal{L}_{\text{comm}}$):**
   $$\mathcal{L}_{\text{comm}} = 0.001 \times \text{KB}_{\text{transferred}}$$
