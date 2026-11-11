# ~/dev 루트 잔여 — 3차 팩트 (261112)

MoC §1 연구 레포는 **루트 유지**. 아래는 §6 예시 밖 항목의 **판정만** (이동은 별 승인).

| 경로 | last commit | 판정 | 3차 |
|---|---|---|---|
| **active research** | | | |
| cell-segmentation | 2026-11-04 | active | 유지 |
| cerebellar-manifold | 2026-11-05 | active | 유지 |
| mouse-counting-neural | 2026-11-09 | active | 유지 |
| MAMMAL_mouse | 2026-10-23 | active | 유지 |
| mouse_gaussian_avatar | 2026-10-23 | active | 유지 |
| neuro-behavior-embedding | 2026-09-30 | active | 유지 |
| usagi-media | 2026-10-27 | active | 유지 |
| **LLM / memory** | | | |
| ai-engram | 2026-09-19 | maintained | 유지 |
| ai-engram-repro | 2026-08-14 | repro | ai-engram 대조 후 archive 검토 |
| avllm-symbolic | 2026-08-07 | active | 유지 |
| avllm-temporal | 2026-10-28 | active | 유지 |
| **PoC** | | | |
| conf-radar | 2026-10-07 | PoC | → `poc/conf-radar` + symlink ✓ |
| multimodal-embedding-poc | 2026-07-30 | PoC | 루트 git 정본 |
| neural_information_processing | 2026-09-05 | course/hub | 유지 |
| brain-inspired-podcast | 2026-09-20 | media | 유지 |
| cerebellum-fn | 2026-08-13 | research | 유지 |
| poc-archive | 2026-08 | cold | 유지 |
| **non-git / infra** | | | |
| _ext | — | vendor ref 45M | BKinD-3D, dinov3 |
| _wt | 0B | worktree stub | 유지 |
| app | — | vendor 11M | czkawka → tools 검토 |
| iphone | 80M | mobile assets | mobile 근처 검토 |

## scratch (P4 미삭제)

| path | size | consumer |
|---|---|---|
| scratch/261025_instagram_resources | 339M | usagi app json |
| scratch/261024_* | 581M | — |
| _scratch/raw_keep | 1.4G | grep 0 active |
| _scratch/260928_tower | 1.6M | gyeol archive only |

검증: `readlink ~/dev/conf-radar` → `poc/conf-radar`
