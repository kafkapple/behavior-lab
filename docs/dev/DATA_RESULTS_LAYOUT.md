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

## R0 점검

```bash
ls ~/data
# INDEX datasets 키와 대조 — 불일치는 BS docs/data/README.md §에 기록
```
