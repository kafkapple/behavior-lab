# Cross-Species Behavior Representation Learning

> *Backlinks: [Overview](../overview.md) | [SSL Methods](ssl_methods.md) | [Graph Models](graph_models.md)*

## Motivation

Behavioral neuroscience studies diverse species (mice, rats, primates, humans) using skeleton-based tracking. A cross-species representation enables:
1. **Transfer learning** from data-rich species (human: 56K sequences) to data-scarce ones
2. **Comparative ethology**: Discover conserved behavior patterns across species
3. **Unified models**: One framework for all species instead of per-species pipelines

## Skeleton Topology Comparison

### CalMS21 Mouse (7 joints, 2D top-view)

```
    0:nose
      |
  1:l_ear--2:r_ear
      |
    3:neck       <- center joint
      |
  4:l_hip--5:r_hip
      |
   6:tail_base
      |
   7:tail_tip
```

### NTU Human (25 joints, 3D front-view)

```
         3:head
           |
         2:neck
      /    |    \
 4:l_shldr 20:spine 8:r_shldr
     |      |       |
 5:l_elbow  1:mid  9:r_elbow
     |      |       |
 6:l_wrist  0:base 10:r_wrist
             |
       12:l_hip--16:r_hip
          |         |
       13:l_knee 17:r_knee
          |         |
       14:l_ankle 18:r_ankle
          |         |
       15:l_foot 19:r_foot
```

## Canonical 5-Part Mapping

```
| Part     | ID | Mouse (CalMS21)    | Human (NTU)            |
|----------|----|--------------------|------------------------|
| Head     | 0  | nose, l_ear, r_ear | head, neck             |
| Spine    | 1  | neck               | spine_base, mid, shldr |
| Left     | 2  | l_hip              | l_arm chain, l_leg     |
| Right    | 3  | r_hip              | r_arm chain, r_leg     |
| Tail/End | 4  | tail_base, tail_tip| l_foot, r_foot         |
```

**Mapping function**: `f: V_species -> {0,1,2,3,4}` (many-to-one)

Each skeleton's `body_parts` field in `SkeletonDefinition` defines this mapping.

## Behavior Category Alignment

| Abstract Behavior | Mouse (CalMS21) | Human (NTU) |
|-------------------|-----------------|-------------|
| Aggression | attack | punch, slap, kick |
| Affiliation | mount | hug, handshake |
| Investigation | investigation | pointing, touching |
| Neutral | other | standing, sitting |

## Cross-Species Training Strategy

### Phase 1: Species-Specific Pre-training
- Train separate SSL models (DINO/JEPA) on each species
- Use species-specific skeleton topology and augmentation

### Phase 2: Cross-Species Alignment
- Map features to canonical 5-part representation
- Contrastive loss aligns similar behaviors across species:
```
L_cross = -log( exp(sim(z_mouse, z_human_pos) / tau) /
                sum( exp(sim(z_mouse, z_human_neg) / tau) ) )
```

### Phase 3: Joint Fine-Tuning
```
L_total = L_ssl_mouse + L_ssl_human + lambda * L_cross_species
```

## Challenges

| Challenge | Description | Mitigation |
|-----------|-------------|------------|
| **Joint count mismatch** | 7 vs 25 joints | Canonical 5-part pooling |
| **Coordinate system** | 2D top-view vs 3D front-view | Body-size normalization |
| **Temporal dynamics** | Different frame rates, action speeds | Temporal normalization |
| **Semantic gap** | "Attack" means different things | Hierarchical behavior ontology |
| **Label mismatch** | 4 classes vs 60 classes | Abstract category mapping |

## Body-Size Normalization

For cross-species velocity comparison:

```
v_normalized = v_pixel / body_size * fps

body_size = distance(nose, tail_base)  [mouse]
body_size = distance(head, mid_hip)    [human]

Behavior thresholds (body-lengths/sec):
  stationary:  v < 0.5 BL/s
  locomotion:  0.5 <= v < 3.0 BL/s
  fast motion: v >= 3.0 BL/s
```

## Keypoint Importance Analysis

Methods to identify which joints matter most per behavior:

1. **Gradient-based**: `importance(j) = ||dL/dx_j||` (backprop saliency)
2. **Attention-based**: InfoGCN attention weights per joint
3. **Occlusion**: Performance drop when joint j is zeroed out

---

*See also: [Overview](../overview.md) | [SSL Methods](ssl_methods.md) | [Evaluation](evaluation.md)*
