# SBeA 원본 취득

figshare 직링크 매니페스트 + 다운로드 러너. **260814 gpu03 홈 루트에서 회수** — 서버에만 있어
소멸 직전이었다(`~/download_sbea.sh`·`~/download_sbea_social_parallel.sh`·`~/sbea_*_files.tsv`).

## 쓰는 법

```bash
./download_sbea.sh social   /mnt/d/data/raw/SBeA/social   4
./download_sbea.sh individual /mnt/d/data/raw/SBeA/individual 4
```

- resume-safe — 크기가 매니페스트와 맞는 파일은 건너뛴다. kill 후 재실행해도 안전.
- 세션 1개 = `caliParas.mat` 1 + `camera-{0..3}.mp4` 4 = 5파일.

## 매니페스트

| 파일 | 행 | 세션 | 용량 | 260814 보유 |
|---|---:|---:|---:|---|
| `sbea_individual_files.tsv` | 100 | 20 | ~11.2 GB | 🟢 `/mnt/d/data/raw/SBeA/individual` 100파일 |
| `sbea_social_files.tsv` | 150 | 30 | ~16.8 GB | 🟢 `/mnt/d/data/raw/SBeA/social` 150파일 (260814 재취득, 150/150 SKIP 대조) |

TSV 열 = `파일명 \t 바이트수 \t URL`. 바이트수가 resume 판정 기준이므로 임의 편집 금지.

## 왜 레포에 있나

260724 교훈(`~/dev/CODING_PRINCIPLES.md`) — *전처리·취득 스크립트를 로컬에만 두면 산출물 소실 시
재현 불가*. PS `convert_m5_for_ps.py` 미커밋 소실로 34일 정체한 전례가 있다.
이 스크립트들은 gpu03 홈 루트에 2026-06-14 부터 미추적 상태로 있었고, 8/14 서버 종료로
같은 사고가 될 뻔했다.
