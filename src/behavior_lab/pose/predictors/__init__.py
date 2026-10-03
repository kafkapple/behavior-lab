"""2D keypoint predictors on a fixed image set: one long table per model, one comparison.

    tables    long prediction table (model, frame, camera, keypoint, x_px, y_px, conf) + SUBTLE/DLC adapters
    joints    keypoint name -> body part per model, so models with different skeletons can be compared
    compare   label-free agreement (L/R-agnostic) and multi-view reprojection residual
    registry  every candidate model with its run status
    run_dlc, run_vitpose   zero-shot runners (GPU host; heavy imports are lazy)

Without hand labels none of this measures accuracy; see docs/avatar_keypoint_models.md.
"""
