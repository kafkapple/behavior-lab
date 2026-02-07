# behavior-lab Documentation

## Map of Content

### Core Documents

| Document | Description |
|----------|-------------|
| **[Overview](overview.md)** | Research purpose, hypotheses, methodology, datasets, model zoo |
| **[Architecture](architecture.md)** | Module map, data formats, config system, skeleton registry |
| **[Integration Plan](INTEGRATION_PLAN.md)** | Migration phases, source mapping, directory structure |

### Theory

| Document | Topics |
|----------|--------|
| **[Graph Models](theory/graph_models.md)** | GCN fundamentals, ST-GCN, 2s-AGCN, InfoGCN, SA-GC, MS-TCN |
| **[SSL Methods](theory/ssl_methods.md)** | MAE vs JEPA vs DINO, sparse data challenges, encoder compatibility |
| **[Cross-Species](theory/cross_species.md)** | Canonical 5-part skeleton, behavior alignment, training strategy |
| **[Evaluation](theory/evaluation.md)** | Metrics (NMI/ARI/Silhouette), Hungarian matching critique, protocol |

### Quick Reference

**Data format**: `(T, K, D)` canonical -> `(N, C, T, V, M)` for graph models only

**Model selection guide**:
- Have labels? -> Supervised (LSTM for sequences, InfoGCN for graphs)
- No labels, want features? -> SSL (DINO + InfoGCN)
- No labels, want clusters? -> Unsupervised (PCA + UMAP + KMeans)
- Multi-subject interaction? -> InfoGCN-Interaction

**Skeleton selection**:
- Mouse top-view: `mars` (7 joints) or `dlc_topviewmouse` (27 joints)
- Human: `ntu` (25 joints, 3D) or `coco` (17 joints, 2D)
- Quadruped: `dlc_quadruped` (39 joints)

### Source Repositories

| Repository | Role | Key Assets |
|------------|------|------------|
| [infogcn-project](https://github.com/kafkapple/infogcn-project) | SSL framework + GCN models | InfoGCN, SSL methods, skeleton registry |
| superanimal-behavior-poc | Pose -> Action pipeline | DLC wrapper, sequence classifiers, evaluation |
| animal-behavior-analysis | Web app + clustering | FastAPI backend, React frontend, UMAP pipeline |
