# 260705 — behavior-lab gpu03 실험 셋업 정찰 + 계획

## gpu03 상태 (2026-07-05, SSH 복구됨)
- SSH 접속 OK (260628 이후 불가였다가 복구). 8× GPU (0-7), 각 ~96GB, **전부 유휴(0%)**.
- 디스크: /home/joon(NFS) 6.5T free · /node_data 486G(93%) · /node_data_2 790G(89%).
- conda: `behaviorsplatter` env (torch 2.11.0+cu128, CUDA True). **behavior_lab 미설치**.
- behavior-lab: `~/dev/behavior-lab` 존재하나 **git repo 아님(plain dir)**, vame.py 없는 구버전. behavemae/bsoid/moseq/subtle/clustering 보유.

## 데이터·모델 자산
- ✅ hBehaveMAE 체크포인트: `~/dev/behavior-lab/checkpoints/behavemae/hBehaveMAE_MABe22.pth` (244MB, MABe22 전용).
- ✅ s-DANNCE 데이터: `~/data/sdannce` + preprocessed `WK1_*`/`SOC1r*` (6뷰 3D pose, rat/mouse).
- ❌ SBeA: 미보유(예상 경로 없음).
- ❌ BehaVERT 공개 데이터/코드: 없음(vaporware, /fact 검증).

## 블로커
1. **git 동기화 위험**: mac behavior-lab(main, remote kafkapple/behavior-lab)에 **내 VAME 외 사전 미커밋 변경 13개**(bsoid/moseq/subtle_wrapper/html_report/skeleton/compare_clustering 등) + untracked 신규(catalog.py·run_behavior_workbench_batch.py 포함). 내 편집이 이들과 뒤섞임 → 안전 분리 불가. gpu03은 non-repo. **사용자 판단 필요**.
2. **hBehaveMAE 체크포인트=MABe22 전용(single-view)** → 멀티뷰 s-DANNCE엔 재학습 필요.
3. behavior-lab에 **s-DANNCE loader 부재** → s-DANNCE 3D pose를 (T,K,D)로 넣는 변환 필요.

## 계획 (proposal-review)
- **P0 (즉시 가능, 경량)**: s-DANNCE 3D pose → 경량 비교(VAME·MoSeq·KP-MoSeq·B-SOiD·SUBTLE) via compare_discovery_methods → html_report. 선행: (a) 안전 git 정리 후 코드 동기화, (b) env 설치(+vame-py), (c) sdannce loader.
- **P1 (GPU)**: hBehaveMAE를 s-DANNCE에 재학습 OR BEAST pretrain(video). 
- **평가**: silhouette+ARI/NMI(s-DANNCE HLAC 라벨=외부앵커)+bout.

## 다음 결정 필요
- git: 사용자가 사전 미커밋 변경 처리 방침(별도 커밋/stash) → 그 후 내 VAME만 브랜치 커밋.
- 실험: P0부터 착수 승인(env 설치+sdannce loader 작성).
