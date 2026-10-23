# SUBTLE theory notes (261023)

What each SUBTLE stage does, where randomness enters, and how to read cluster levels.
- Code facts: the read-only copy under `~/dev/external/subtle_ref/SUBTLE_storage/engine/subtle/`.
- Stage input and output: `subtle_pipeline_io.md`. Viewer use: `subtle_viewer_manual.md`.

## 1. Four steps of one SUBTLE fit
- Input: 3D skeleton per frame (11 joints x 3), centered by the global mean (`avatar_preprocess`).
- Step 1, spectrogram: a Morlet wavelet transform of the posture time series gives a feature vector per frame (`morlet_cwt`).
- Step 2, embedding: UMAP maps each frame to 2D (`n_neighbors=50`). This 2D plane is the cluster map in the viewer.
- Step 3, subclusters: Phenograph builds a k-nearest-neighbor graph (k=30) on the 2D points and cuts it into communities with Leiden. Result: fine clusters (49-55 in our 8-animal fits). The number is decided by the data.
- Step 4, superclusters: DIB merges subclusters into coarser groups that still predict what the animal does next (section 2).

## 2. DIB
- DIB = Deterministic Information Bottleneck (Strouse and Schwab, UAI 2016; [paper](https://arxiv.org/abs/1604.00268)).
- Problem: compress X into a label T with few distinct values while keeping information about Y.
- The original information bottleneck minimizes I(X;T) - beta I(T;Y). DIB replaces the compression term by the entropy of T: H(T) - beta I(T;Y).
- Effect: each X gets exactly one T (a hard assignment, not a probability), which is what a cluster label needs.
- In SUBTLE, X is the subcluster at time t and Y is the subcluster tau frames later (`tau` is a per-recording time lag averaged over recordings, `mapper.py:333-335`; `Mapper.run` computes it as tau = -2 / ln|lambda2| from the second-largest eigenvalue of the subcluster transition matrix, floored at half a frame interval; read in the installed `subtle` source 261023).
- So a supercluster is a group of subclusters that lead to the same futures: it keeps the predictive structure of behavior.
- beta sets the trade-off. Large beta keeps more information and gives more superclusters; small beta merges more.
- The code cites a MATLAB implementation from `bermanlabemory/behavioral-evolution` (`module.py` comment); we did not check that repository.

## 3. Superclusters are levels, not one number
- `run_DIB(X, Y, N=1000, minClusters=2, maxClusters=30)` runs N iterations. Each draws a random beta (log-uniform, 10^-1 to 10^4) and a random cluster count k between 2 and 30 (`module.py:269-270`), and solves DIB for that pair.
- It keeps the solutions it finds, one per distinct cluster count, and returns them as a list. The fit stores them as `Y`, shape (frames, levels); column j is the supercluster label of every frame at level j.
- In our 8-animal fits the levels had 1, 2, 3, 4 and 5 clusters. Level with 1 cluster is trivial.
- You do not set the number of superclusters before the fit. You pick a level after it. The Justin app does this with the select box `subtleSuperclusterSize`; the viewer does it with the level select (`L0: n clusters`).
- Unequal count of levels between runs is expected: random (beta, k) pairs find a different set of distinct counts (4 levels in one fit, 5 in another).

## 4. Where randomness enters
| Step | Source | Control |
|---|---|---|
| Frame sampling | `np.random` when frames > `n_train_frames` | `np.random.seed` |
| UMAP | no `random_state` in the engine | `UMAP(random_state=seed)` |
| Leiden (Phenograph) | `seed` argument exists, engine does not pass it | `Phenograph(seed=seed)` |
| DIB | random beta and k each iteration | `np.random.seed` |

- The viewer wrapper sets all four with one `seed` (default 0). Two fits of the 8 animals with seed 0 gave identical sub and super labels (261023).
- Without a seed, two fits of the same data gave 4 and 5 levels and 39-55 subclusters. A group result must come from a fixed seed, and a second seed should be run to see if it holds.
- The Justin app sets no seed anywhere, so its own results are not repeatable.

## 5. How to read the output
- Cluster id is a name, not a quantity. Id 3 in one fit has no relation to id 3 in another fit. Compare ids only inside one `fit_id` (or one saved model).
- Colors follow the id, so the same id has the same color in every view of the viewer; a new fit reuses the same colors for different meanings. The legend shows the `fit_id`.
- Frequencies per animal (frames in a cluster / frames) are the numbers to compare between groups; the unit is the animal (`subtle_viewer_prd.md`, P6, P11).
- DEK (kinematic features per cluster) is planned in `subtle_stats_plan.md`.
