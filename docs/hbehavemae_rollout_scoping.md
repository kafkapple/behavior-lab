# hBehaveMAE keypoint-attribution — scoping (2026-07-07)

Deferred item from the CalMS21 ST-GCN study (handoff issue 1). Goal was to **cross-validate
the ST-GCN Grad-CAM keypoint importance** with a second, independent model (hBehaveMAE).
This doc scopes what that actually takes and — critically — whether attention-rollout is the
right tool. **Data is ready on gpu03; nothing here is implemented yet.**

## Assets (verified on gpu03, 2026-07-07)
- Checkpoints: `/node_data/joon/outputs_bmae/calms21/checkpoint-000{20,40,60,99}.pth`
- Raw data: `/node_data/joon/data/calms21_bmae/calms21_train.npy`
- Code: `external/BehaveMAE/models/general_hiera.py` (`GeneralizedHiera`, `MaskUnitAttention`, `Unroll`/`Reroll`)
- Config (from `checkpoint-00099.pth['args']`): `input_size=[90, 2, 14]` (90 frames × 2 mice × 14 = 7 kp × 2 coords), `stages=[2,3]` (2 stages, depth 5), `q_strides=[(3,1,1)]` (one temporal /3 pool), `mask_unit_attn=(True, False)`.

## What attention-rollout is
Abnar & Zuidema (2020): approximate how much each **input token** influences the output by
recursively multiplying per-layer attention matrices, adding identity for the residual path:
`R = (0.5 A_L + 0.5 I) @ ... @ (0.5 A_1 + 0.5 I)`. Needs a **single fixed token set** and a
**full N×N attention** at every layer. Standard ViT satisfies both. Hiera satisfies neither.

## Why standard rollout does NOT apply to this Hiera (3 concrete breaks)
1. **Windowed (mask-unit) attention in stage 0** (`mask_unit_attn=(True,False)`, `MaskUnitAttention` L35-104): attention is computed **within local mask units**, not globally — the stage-0 attention is block-diagonal over windows, not a full N×N. Must be scattered into the full token space before it can be composed.
2. **Token pooling between stages** (`q_stride=(3,1,1)`, `do_pool` L146): the temporal axis is pooled /3 at the stage-0→1 boundary, so the token count (and thus attention-matrix dimension) **changes across stages**. Chain-multiplying A_stage0 (N tokens) with A_stage1 (N/3 tokens) requires an explicit pooling operator P so the product is `A1 @ P @ A0`, not `A1 @ A0`.
3. **Unroll/Reroll token reordering** (`Unroll` L280, `Reroll` L281): Hiera permutes tokens into mask-unit-contiguous order for windowed attention. Any attribution computed in token space must be pushed back through `Reroll` (and then the `PatchEmbed` conv, `patch_stride`) to land on the original `(T=90, M=2, V=7×2coords)` layout — i.e. to become a per-keypoint map.

## What a correct implementation entails
1. Load `GeneralizedHiera` + calms21 checkpoint (SSL encoder; drop/ignore the MAE decoder).
2. Forward a batch with hooks capturing each block's post-softmax attention (`attn` in `MaskUnitAttention.forward`, L102-103).
3. Stage 0: scatter windowed attention into a full N×N (zeros off-window).
4. Insert the pooling operator P at the q_pool boundary; compose `R = (½A1+½I) @ P @ (½A0+½I)` per stage with residual.
5. Push the input-token attribution back through `Reroll` + un-patchify to `(90, 2, 14)`, then reduce coords → **per-keypoint, per-frame importance** comparable to the ST-GCN Grad-CAM output.
6. Validate: sanity-check that attention sums, pooling scatter, and reroll indices are correct (unit test on a toy forward), else the map is silently wrong.

Estimate: a focused implementation + validation session (not a quick script). The reroll/pool
index bookkeeping is the error-prone part.

## --devil: is attention-rollout even the right tool here?
Three reasons it is **low-ROI and possibly the wrong method**, ranked:

1. **SSL-vs-classifier mismatch (fatal to the comparison's meaning).** hBehaveMAE is a *masked
   autoencoder* — its attention reflects "what helps reconstruct masked keypoints," not "what
   discriminates attack/mount/investigate/other." ST-GCN Grad-CAM is *class-discriminative*.
   These answer different questions; agreement/disagreement is hard to interpret as validation.
2. **Attention ≠ explanation.** Jain & Wallace (2019), Serrano & Smith (2019): attention weights
   are not a faithful importance measure — different attention maps yield the same output. So
   even a *correctly* implemented rollout is a contested attribution, not ground truth.
3. **The target is already downgraded.** The ST-GCN keypoint-level claims are hypothesis-level
   (noise-causality unconfirmed; fixed-code Tier C shows importance ≈ uniformly tail_base).
   Cross-validating a hypothesis with a contested method on a different task is poor priority.

### Cheaper, more-valid alternatives (recommended over rollout)
- **Occlusion on hBehaveMAE embeddings** — mask each keypoint in the input, measure the change
  in reconstruction error (or in a downstream linear-probe logit). Model-agnostic, no rollout,
  directly interpretable, reuses the occlusion pattern already in `calms21_keypoint_ablation.py`.
- **Linear-probe + input-gradient** — freeze the hBehaveMAE encoder, train a 4-class linear
  probe on its embeddings, then do integrated-gradients/occlusion on *that* — gives a
  class-discriminative keypoint importance directly comparable to ST-GCN Grad-CAM (same question).

## Recommendation
Do **not** implement attention-rollout first. If the cross-check is still wanted, do the
**linear-probe + gradient** route (answers the same class-discriminative question hBehaveMAE
rollout would not) or **occlusion**. Reserve rollout only if the specific interest is
"where does the SSL encoder's attention go," which is a different research question. Priority:
low — the ST-GCN conclusions (supervised signal real; data-quantity underpowered; noise effect
retracted) stand without this cross-check.
