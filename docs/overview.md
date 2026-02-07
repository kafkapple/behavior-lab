# behavior-lab: Research Overview

> [← MoC](README.md) | [Architecture →](architecture.md) | [Model Taxonomy →](model_taxonomy.md)

## Purpose

Skeleton-based behavior recognition across species (human + animal) through a unified modular framework. Three fragmented repositories consolidated into one platform supporting supervised, self-supervised, and unsupervised approaches.

## Core Research Questions

| Level | Question | Status |
|-------|----------|--------|
| **RQ1** | Can skeleton topology encode sufficient information for behavior classification? | Validated (LSTM 96%, InfoGCN 43% unsupervised) |
| **RQ2** | Can SSL methods learn meaningful behavior representations without labels? | Validated (DINO > JEPA > MAE for skeleton data) |
| **RQ3** | Do universal behavior representations exist across species? | In progress |
| **RQ4** | Can a single framework unify all three paradigms (supervised/SSL/unsupervised)? | This project |

## Hypotheses

### H1: Information Bottleneck for Skeleton GCN
InfoGCN's attention-based graph convolution with variational bottleneck captures more discriminative joint relationships than fixed-topology GCN.

- **Evidence**: InfoGCN-Interaction outperforms GCN/InfoGCN on CalMS21 social behaviors (NMI 0.12 vs 0.05)
- **Mechanism**: `min I(X; Z) - beta * I(Z; Y)` compresses irrelevant pose variation while preserving action-relevant features

### H2: Latent-Space SSL > Reconstruction for Sparse Data
Skeleton sequences are sparse (~1.8K values vs image ~150K), making pixel-level reconstruction (MAE) less effective than latent prediction (JEPA) or distribution matching (DINO).

```
Dense (image): ~150K values, 40% mask -> ~90K remain (sufficient)
Sparse (skeleton): ~1.8K values, 40% mask -> ~1.1K remain (risky)
```

- **Prediction**: DINO >= JEPA >> MAE for skeleton SSL
- **Observed**: DINO consistently best; MAE struggles with sparse reconstruction targets

### H3: Cross-Species Canonical Representation
Morphologically different skeletons (7-joint mouse, 25-joint human) can be mapped to a shared 5-part canonical skeleton, enabling cross-species behavior transfer.

```
| Part    | Mouse (7)          | Human (25)              |
|---------|--------------------|-----------------------  |
| Head    | nose, l/r_ear      | head, neck              |
| Spine   | neck               | spine_base/mid/shoulder |
| L-Limb  | l_hip              | l_shoulder...l_foot     |
| R-Limb  | r_hip              | r_shoulder...r_foot     |
| Tail    | tail_base, tail_tip| l_foot, r_foot          |
```

- **Strategy**: Species-specific SSL pre-training -> Cross-species contrastive alignment -> Joint fine-tuning

### H4: Multi-Subject Interaction Modeling
Social behaviors (attack, mount, investigation) require explicit inter-subject modeling beyond simple feature concatenation.

- **Approach**: Cross-attention between subject features + interaction pooling `[f1, f2, f1*f2, |f1-f2|]`
- **Evidence**: InfoGCN-Interaction > InfoGCN on CalMS21 (multi-mouse dataset)

## Methodology Overview

### Three Paradigms, One Pipeline

```
                    Raw Video / Pre-extracted Poses
                              |
                    +---------+---------+
                    |                   |
              [Pose Estimation]    [Pre-processed NPZ]
              DLC SuperAnimal       NTU / CalMS21
              (T, K, 3)            (N, C, T, V, M)
                    |                   |
                    +----> (T, K, D) <--+    Canonical Format
                              |
            +-----------------+-----------------+
            |                 |                 |
      [Supervised]      [Self-Supervised]  [Unsupervised]
      LSTM/MLP/         MAE/JEPA/DINO     PCA -> UMAP
      Transformer       + GCN Encoders    -> KMeans
            |                 |                 |
      Action Labels    Learned Features   Cluster Labels
            |                 |                 |
            +---------> Evaluation <-----------+
                    Accuracy, NMI, ARI,
                    Silhouette, F1
```

### Learning Paradigms

| Paradigm | Input | Output | Models (Dir) | Use Case |
|----------|-------|--------|--------------|----------|
| **Supervised** | (T,K,D) + labels | Action class | graph/, sequence/ | Labeled datasets |
| **SSL** | (N,C,T,V,M) | Feature embeddings | ssl/ | Label-scarce settings |
| **Discovery** | (T,K,D) | Cluster/syllables | discovery/ | Behavior discovery |

### Model Zoo (30+)

> 상세 카탈로그 + 분류 체계 분석: [Model Taxonomy](model_taxonomy.md)

| Category | Model | Input Format | Key Innovation |
|----------|-------|-------------|----------------|
| **Graph** | InfoGCN | (N,C,T,V,M) | SA-GC + Info Bottleneck + MMD |
| **Graph** | ST-GCN | (N,C,T,V,M) | Fixed spatial partition + TCN |
| **Graph** | 2s-AGCN | (N,C,T,V,M) | Adaptive adjacency learning |
| **Sequence** | LSTM | (T,K,D) | Bidirectional, best for locomotion |
| **Sequence** | MLP | (T,K,D) | Frame-wise, fast inference |
| **Sequence** | Transformer | (T,K,D) | Self-attention, long-range deps |
| **SSL** | MAE | (N,C,T,V,M) | Masked joint reconstruction |
| **SSL** | JEPA | (N,C,T,V,M) | Latent prediction (no decoder) |
| **SSL** | DINO | (N,C,T,V,M) | Self-distillation + centering |
| **Discovery** | B-SOiD | (T,K,2) | UMAP + HDBSCAN + RF two-space |
| **Discovery** | MoSeq | (T,K,D) | AR-HMM / SLDS syllables |
| **Discovery** | SUBTLE | (T,K,D) | Wavelet spectrogram + UMAP |
| **Discovery** | BehaveMAE | (T,K,D) | Hierarchical masked autoencoder |
| **Graph (PySKL)** | ST-GCN++ | (T,K,D)→auto | Good practices + multi-stream |
| **Graph (PySKL)** | CTR-GCN | (T,K,D)→auto | Channel-wise topology refinement |
| **Graph (PySKL)** | MS-G3D | (T,K,D)→auto | Multi-scale disentangled graph |

## Datasets

| Dataset | Species | Joints | Subjects | Classes | Sequences | Dims |
|---------|---------|--------|----------|---------|-----------|------|
| **CalMS21** | Mouse | 7 | 2 | 4 | 19,144 | 2D |
| **NTU RGB+D** | Human | 25 | 1-2 | 60 | 56,880 | 3D |
| **UCLA** | Human | 20 | 1-10 | 10 | 1,475 | 3D |
| **DLC TopViewMouse** | Mouse | 27 | 1 | - | varies | 2D |
| **DLC Quadruped** | Animal | 39 | 1 | - | varies | 2D |

## Key References

| Paper | Venue | Contribution |
|-------|-------|-------------|
| InfoGCN | CVPR 2022 | Information Bottleneck GCN for skeleton |
| ST-GCN | AAAI 2018 | Spatial-temporal graph convolution |
| 2s-AGCN | CVPR 2019 | Adaptive graph convolution |
| MAE | CVPR 2022 | Masked autoencoder |
| I-JEPA | CVPR 2023 | Joint-embedding predictive architecture |
| DINO | ICCV 2021 | Self-distillation with no labels |
| SuperAnimal | Nat Comm 2024 | Foundation model for animal pose |
| CalMS21 | NeurIPS 2021 | Mouse social behavior benchmark |
| CEBRA | Nature 2023 | Neural-behavior joint embedding |
| B-SOiD | Nat Comm 2021 | Unsupervised behavioral segmentation |
| keypoint-MoSeq | Nat Methods 2024 | Behavioral syllables via SLDS |
| SUBTLE | IJCV 2024 | Wavelet-UMAP behavioral repertoires |
| BehaveMAE | ECCV 2024 | Hierarchical MAE for behavior |
| PySKL | ACM MM 2022 | ST-GCN good practices toolbox |

---

> [← MoC](README.md) | [Architecture](architecture.md) | [Model Taxonomy](model_taxonomy.md) | [Theory →](theory/)
