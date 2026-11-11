# ~/data · ~/results — mac 레이아웃 (261112)

정본 체인 (bscode·BS):

1. `BehaviorSplatter/docs/data/dataset_naming.md` — alias 3단
2. `BehaviorSplatter/configs/preprocessing/DATASET_INDEX.yaml` — config 경로
3. `BehaviorSplatter/docs/data/README.md` — 온디스크 현황
4. `~/data/_LAYOUT_MOVES.tsv` — legacy→alias 이동·symlink 감사 로그 (append-only)

## ~/data

```text
~/data/<dataset-alias>/{raw,derived,processed}/
~/data/weights/          # 공용 체크포인트
~/data/_archive/         # gpu03_salvage 등
```

금지: 루트 flat `preprocessed/` 등 — 신규는 alias 하위만. 이동 시 `_LAYOUT_MOVES.tsv` 한 줄 추가.

## ~/results

[~/dev/CLAUDE.md](~/dev/CLAUDE.md) §Results — `~/results/{project}/`, 이미지 `ep{NNNN}_{metric}.png`.

역할: **data** = 입력·파생 캐시, **results** = 실험·평가 산출. `~/data/BehaviorSplatter` vs `~/results/BehaviorSplatter` 중복 시 data=입력만.

## mac 현황 (261112 실사)

| 경로 | 역할 | 조치 |
|---|---|---|
| `mouse-dannce-m1/` · `rat-sdannce-*` | dataset-first 정본 | 유지 |
| `derived/` · `preprocessed/` | **호환 stub** — 자식만 symlink → alias 트리 | 삭제 금지 |
| `M5_local`, `WK1_*`, `raw/markerless_*` 등 | legacy 이름 → alias ( `_LAYOUT_MOVES.tsv` ) | 유지 |
| `BehaviorSplatter/` (~15G) | BS **mac 캐시** (sceneflow_viewer, annotations, cache) | `~/results` 와 별도; 코드가 `~/data/BehaviorSplatter` 하드코딩 |
| `results/` (6M) | legacy `MAMMAL_mouse/kp22_dlt` 만 | 신규 산출은 `~/results/MAMMAL_mouse/`; 통합은 차기 승인 |
| `weights/`, `garciagarcia-2026/`, `subtle_*` | 프로젝트별 데이터 | alias 등재는 INDEX·README 동기화 때 |

## ~/dev 레이아웃 (261112)

정본: [README.md §6](README.md). 아카이브 repo = `_archive/dev-repos/` + 루트 symlink.

| 활성 | 위치 |
|---|---|
| avllm-symbolic · avllm-temporal | 루트 (archive 아님) |
| beast | 루트 (BEAST3D baseline) |
| TIL | 루트 |

## R0 점검

```bash
ls ~/data
# INDEX datasets 키와 대조 — 불일치는 BS docs/data/README.md §에 기록
rg '~/data/' ~/dev/BehaviorSplatter/configs/preprocessing/DATASET_INDEX.yaml
```
