# SUBTLE transitions, DIB, and TPI: focused source survey (2026-10-29)

This note explains what the three calculations answer. It is a focused reading of the [SUBTLE paper](https://link.springer.com/article/10.1007/s11263-024-02072-0), the [DIB method paper](https://pubmed.ncbi.nlm.nih.gov/28410050/), and the installed SUBTLE implementation (`mapper.py`, `module.py`, `tpi.py` under `~/dev/external/subtle_ref/SUBTLE_storage/engine/subtle/`). It is not a systematic comparison of behavior-discovery methods. See also [theory](subtle_theory.md), [viewer manual](subtle_viewer_manual.md), and [walkthrough](subtle_workthrough.md).

## One sequence, three questions

Imagine a subcluster sequence `A A B B C`. An ordinary one-frame transition table counts `A→A`, `A→B`, `B→B`, and `B→C` once each. **Retention** is the share that stays in the same cluster on the next frame. **Exit probability** removes those self-transitions, then asks: once the animal leaves `A`, which cluster follows? A row with no exits is undefined, not zero probability. The viewer's Transitions screen computes these descriptive tables from the selected project's labels. It never links the end of one session to the start of another.

**DIB (Deterministic Information Bottleneck)** asks a different question: can we replace many fine subcluster labels at time `t` with a smaller supercluster label while retaining information about the subcluster at `t + τ`? In the example, if `A` and `B` are followed by similar future subclusters, DIB may assign both to one supercluster. This does not mean they have identical poses or that every immediate transition between them is strong. The DIB method uses a deterministic (hard) assignment and trades label compression, measured by `H(supercluster)`, against predictive information, `I(supercluster; future subcluster)`. The β parameter controls this trade-off. The [DIB paper](https://pubmed.ncbi.nlm.nih.gov/28410050/) establishes the general objective; the [SUBTLE paper](https://link.springer.com/article/10.1007/s11263-024-02072-0) applies it to fine behavior states.

In this SUBTLE implementation, `mapper.py` first obtains subcluster labels `data.y`. It derives a session-specific lag from an eigenvalue of its exit-transition matrix, averages the lags, and passes `a = y[:-int(avg_tau)]`, `b = y[int(avg_tau):]` to `run_DIB(a, b)`. Thus DIB's direct input is the paired **past and future subcluster labels**, not a TPI score. The lag heuristic and its behavior for sparse or reducible transition matrices deserve sensitivity checks; it should not be interpreted as a biological time constant without validation.

**TPI (Temporal Proximity Index)** asks whether transitions in an embedding tend to connect spatially nearby clusters. SUBTLE's `tpi.py` clusters the embedding at several `k` values, calculates between-cluster transition probabilities and inverse-distance weights between centroids, and sums their products. This is an embedding evaluation score, not the feature vector or objective handed to DIB. High TPI therefore cannot by itself establish that a supercluster predicts future behavior or that a treatment effect exists. The [SUBTLE paper](https://link.springer.com/article/10.1007/s11263-024-02072-0) introduces TPI for temporal connectivity assessment.

## What the viewer computes

| Display | Definition | Interpretation |
|---|---|---|
| Self-retention | `count(i→i) / count(all transitions from i)` | How often label `i` persists one more frame. |
| Exit matrix | `count(i→j) / count(i→any j≠i)`, for `j≠i` | Destination conditional on leaving `i`; diagonal is excluded. |
| Group matrix | Mean of each animal/session's exit row, where that animal has exits from `i` | A descriptive group comparison. The screen lists the number of contributing animals per row. |

The viewer currently shows **one-frame** exit probabilities. DIB uses an estimated lag `τ`, which may differ by fit. The viewer heatmaps are therefore useful for inspecting fitted labels, but they are not a direct visualization of the contingency table optimized by DIB. Cluster IDs are arbitrary within a fit; align groups by the same `fit_id` and level before comparing. Small samples, rare source clusters, and frame autocorrelation make raw differences unstable; use animals as the unit for inference, report support, and avoid treating heatmap color as a significance test.

## Evidence and scope

| Source | Directly supported point | Limit |
|---|---|---|
| Kwon et al., IJCV 2024, [SUBTLE](https://link.springer.com/article/10.1007/s11263-024-02072-0) | Phenograph fine states, DIB predictive coarse states, TPI embedding assessment | Published method claims; does not validate this viewer's group comparisons. |
| Strouse and Schwab, Neural Computation 2017, [DIB](https://pubmed.ncbi.nlm.nih.gov/28410050/) | Entropy-based compression and deterministic labels | General method, not an animal-specific validation. |
| Local `mapper.py:316–343, 495–510`; `tpi.py:8–43` | Actual lagged labels, exit normalization, and separate TPI call in the installed code | Code can diverge from another SUBTLE version. |

Coverage: two primary papers and the local implementation were selected by the named method questions; no full literature-pool screening was performed. The mathematical relationships above are source-backed; the cautions about interpretation are inferences from the definitions and sample structure.
