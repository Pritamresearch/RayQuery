# Real-World Applications, Deployment Scenarios & V2X Integration

This document outlines the real-world utility, edge deployment scenarios, and telecommunication standards integration for **RayQuery**.

---

## 1. Safety-Critical Autonomous Driving Scenarios

```
Scenario 1: Blind Left Turn           Scenario 2: Highway Platooning       Scenario 3: Adverse Weather / Glare
      (Urban Occlusion)                     (Bandwidth Crisis)               (Phantom Braking Rejection)

        Oncoming Car                        Car 1 ──► Car 2 ──► Car 3          Low Sun / Wet Windshield
            ▲                                      ...                             \   \   \
            │                               Car 20 ──► Emergency Debris             ▼   ▼   ▼
       ┌────┴────┐                                                             ┌─────────────────┐
       │ Big Bus │ (Blocks View)            20 cars x 3.12 KB = 0.62 MB/s       │ Collaborator    │
       └─────────┘                          (vs. 60 MB/s dense BEV)            │ Glare Streak    │
            ▲                                                                  └────────┬────────┘
            │                                                                           │ H_q ≈ 1.0 (Entropy)
       ┌────┴────┐                                                                      ▼
       │ Collab  │ ──► Sends 1 query (160 B)                                    [Query Filtered Out]
       └─────────┘     in < 5 ms                                                        │
            ▲                                                                           ▼
            │                                                                  [Ego Stays Safe!]
       ┌────┴────┐                                                              (No Phantom Brake)
       │   Ego   │ ──► Stops Safely!
       └─────────┘
```

### Scenario 1: The Blind Left Turn (Unprotected Urban Intersection)
* **The Setting:** At a busy urban intersection, an autonomous vehicle (Ego) seeks to turn left across traffic. A large stationary city bus in the opposing turn lane blocks the ego's cameras from seeing oncoming traffic traveling at 50 km/h.
* **The Intervention:** A connected vehicle behind the bus has an un-occluded line of sight. 
* **The RayQuery Execution:** The ego vehicle broadcasts an ultra-compact demand bitmask. The collaborator identifies the oncoming vehicle with high depth certainty (low ray entropy $\mathcal{H}_q$) and transmits **just 1 query token (160 bytes) in under 5 ms**. The ego vehicle renders the 3D bounding box through the bus and holds the brakes until traffic clears.

### Scenario 2: High-Speed Highway Platooning (Bandwidth Congestion)
* **The Setting:** A platoon of 20 connected automated vehicles travels on a multi-lane highway at 100 km/h (28 meters per second). Suddenly, 4 cars ahead, debris falls from a cargo truck.
* **The Bandwidth Crisis:** If all 20 vehicles transmit dense BEV feature grids (300 KB at 10 Hz), the local cellular tower or DSRC spectrum requires over **60 MB/second**, triggering severe packet loss and queue delays.
* **The RayQuery Execution:** Each vehicle transmits only its **Top-20 queries (3.12 KB/frame)**. Total traffic across all 20 vehicles is **0.62 MB/second (a 99% reduction)**, easily fitting within standard 10 MHz 5G-NR V2X channels with sub-10 ms latency.

### Scenario 3: Adverse Weather & Optical Glare (Rejecting Phantom Objects)
* **The Setting:** Driving into a low setting sun with wet asphalt and windshield water droplets. Visual flare produces high-confidence 2D activations that monocular projection misidentifies as an obstacle.
* **The RayQuery Execution:** The categorical depth head flags high entropy along the glare ray ($\mathcal{H}_q \approx 1.0$). The query utility $\mathcal{U}(q)$ collapses toward zero, automatically filtering out the fake obstacle and **preventing hazardous phantom braking**.

---

## 2. Telecommunication Standards & V2X Integration

RayQuery is engineered for direct compatibility with international Intelligent Transportation Systems (ITS) standards:

### 1. Integration with SAE J2735 & ETSI Standards
* **SAE J2735 BSM (Basic Safety Message) / ETSI CAM (Cooperative Awareness Message):**  
  Autonomous vehicles already broadcast 10 Hz heartbeats containing vehicle position, velocity, and heading.
* **Zero Protocol Overhead:**  
  RayQuery's 64-byte spatial demand bitmask is appended directly to the optional `dataFrames` container of standard BSM/CAM packets, requiring **zero extra protocol round-trips**.

### 2. 5G-NR V2X Sidelink (PC5 Interface) Compatibility
* Operates over direct vehicle-to-vehicle PC5 Sidelink without routing through base stations (cellular infrastructure-independent).
* The 3.12 KB payload fits comfortably within single-packet transport block sizes (TBS), eliminating transport-layer packet fragmentation and reassembly latency.

---

## 3. Real-World Packet Accounting

| Component | Dimensions / Fields | Raw Float Size | Quantized Payload |
| :--- | :--- | :---: | :---: |
| **3D Bounding Box Center** | $(x, y, z)$ in meters | $3 \times 4$ bytes = 12 B | 12 bytes |
| **3D Bounding Box Extent** | $(l, w, h)$ in meters | $3 \times 4$ bytes = 12 B | 12 bytes |
| **Heading Angle** | $\theta$ in radians | $1 \times 4$ bytes = 4 B | 4 bytes |
| **Latent Feature Embedding** | $\mathbf{f}_q \in \mathbb{R}^{128}$ | $128 \times 4$ bytes = 512 B | **128 bytes** (8-bit quantized) |
| **Ray-Depth Uncertainty** | $\mathcal{H}_q \in [0, 1]$ | $1 \times 4$ bytes = 4 B | 4 bytes |
| **Total per Query** | — | **544 bytes** | **160 bytes** |

For $K=20$ transmitted queries:
$$\text{Per-Frame Payload} = 20 \times 160\text{ bytes} = 3,200\text{ bytes} = \mathbf{3.12\text{ KB/frame}}$$
At 10 Hz transmission rate:
$$\text{Bandwidth per Vehicle} = 3.12\text{ KB} \times 10 = \mathbf{31.2\text{ KB/second}}$$
This achieves full real-world deployability on existing automotive wireless hardware.
