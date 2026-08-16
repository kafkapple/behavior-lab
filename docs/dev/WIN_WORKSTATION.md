# win 개인 서버 이관 — gpu03 이후의 작업 거점

> 2026-08-14 작성. gpu03 접속 종료(AMILab 오프보딩, 실무마감 8/14)에 대응해 behavior-lab 의
> 작업 거점을 개인 win 서버로 옮긴 기록이자 진입점.
> **gpu03 기준 문서(`STORAGE_GUIDE.md`·`DATASETS.md` 의 `/node_data` 표기)는 이 시점부터 사료(史料)다.**
> 이 문서가 현행 경로의 정본.

## 0. 열린 이슈

- 🟡 **SHOT7M2 데이터 3.4G 부재 — 출처 확보, 재취득은 보류(YAGNI 판정 260816).**
  배포처 = HuggingFace `amathislab/SHOT7M2` (공개). 필요해지면 1커맨드로 복구된다:
  `git clone https://huggingface.co/datasets/amathislab/SHOT7M2 <dest>` 후
  `BL_SHOT7M2=<dest>` 주입. 가중치(`hBehaveMAE_Shot7M2.pth` 111M)는 이미 확보.
  **지금 받지 않는 이유**: 소비처가 `scripts/shot7m2_probe.py` 하나뿐이고 그것도 파이프라인·
  테스트 어디에서도 참조되지 않는 일회성 linear probe 다. 나머지 `bmae_probe_cv3.py` 는
  데이터를 받아도 못 돈다 — `BL_BMAE_OUT`(학습 산출 체크포인트)이 소실됐고 그건 **재다운로드
  불가·재학습만 가능**하다. 3.4G 를 받아도 복구되는 건 절반뿐이다.
- **코드 내 gpu03 절대경로 29곳** 미치환 (`/node_data/joon` 23 · `/node_data_2/joon` 6). §6 참조.
- ✅ **S1 완료 (260816) — 판정: 저자 기준선 부재.** 논문은 3D 재투영 잔차 절대값을 보고하지
  않는다. 저자 코드는 클론했으나 자체 로직 54개가 `.pyd`(Windows DLL) 컴파일 바이너리다.
  근거 = vault `[[260816_behaviorlab_sbea_author_baseline_absent]]` · 착수순서 = `docs/sbea_social_and_sdannce_status.md` §0·§4.
- 🔴 **저자 코드 실행은 WSL 에서 불가** — `.pyd` 는 네이티브 Windows + Python 3.9 + CUDA 11.7
  전용. 이 문서의 WSL 셋업(§4)과 **별개 env** 가 필요하다. 착수 전 비용 재산정할 것.
- 🟢 **S3(social) 블로킹 해소** — 260814 데이터 확보. S1 선행 조건도 260816 로 해제.
  단 `scripts/sbea_acquisition/README.md` 의 "social 미보유" 표기는 미갱신(드리프트).

> 🟢 260814 회수 — 전부 매니페스트/해시 대조로 판정했다(서술 아님).
>
> | 회수분 | 규모 | 판정 |
> |---|---|---|
> | SBeA `social` 30세션 | 17G · 150파일 | 150/150 SKIP |
> | ICML 재현 런 | 4.23 GiB · 19파일 | gpu03↔win 일치 |
> | gpu03 홈 6디렉토리 | 521 MiB · 16,281파일 | gpu03↔mac 일치 + SHA-256 `ef2aab47…` |
> | FaceLift gaussians 3종 | 30.55 GiB · 7,920파일 | gpu03↔win 일치 |
> | 인계볼륨 잔여 6항목 | 3.6 GiB · 77,715파일 | gpu03↔mac 일치 |
> | git stash 20 + bundle | 4.4 MiB | SHA-256 `7b6b3f2f…` |
>
> ⇒ **gpu03 에서 회수할 것은 더 남아 있지 않다.** 서버 흔적 소거(삭제)는 별건이며
> 오프보딩 트래커의 v8 SHA-256 게이트 소관이다 — 이 문서 범위 밖.

## 1. 하드웨어 대조

| 항목 | gpu03 (종료) | win 개인 서버 |
|---|---|---|
| 호스트 | `gpu03` | `kafkapple-win` = `DESKTOP-U91M55P` |
| 작업 쉘 | Linux 직접 | WSL2 Ubuntu 22.04 (`wsl -e bash -lc`) |
| GPU | Blackwell (cc 12.0) | RTX 3060 12GB |
| 데이터 볼륨 | `/node_data` 7T · `/node_data_2` 7T | `D:` 3.7T (여유 648G) |
| WSL 루트 | — | 1007G (여유 687G) |

- ssh 진입 시 **기본 쉘이 PowerShell** — bash 명령은 `wsl -e bash -lc '...'` 로 감쌀 것.
- 중첩 따옴표가 자주 깨진다. 긴 명령은 base64 로 감싸 전달하면 안정적.
  ⚠️ **단 stdin 을 쓰는 원격 명령에는 base64 우회를 쓰지 말 것** — `echo B64 | base64 -d | bash`
  는 bash 의 stdin 이 그 파이프가 되어 `tar xf -` 로 스트림이 도달하지 않는다(260814 실패).
- `rsync` 는 win 쪽에 없다. mac→win 전송 = `scp <파일> kafkapple-win:D:/경로/`,
  디렉토리는 `tar cf - -C SRC dir | ssh kafkapple-win "wsl -e bash -lc 'cd DST && tar xf -'"`.
- 🔴 **소파일 수천 개는 rsync 대신 tar 스트림.** 260814 실측: rsync 파일별 왕복이 병목이라
  3,600파일에 8시간 추정 → tar 스트림으로 12분. 대용량 단일 파일은 rsync(재개 가능)가 낫다.
- 🔴 **클론 직후 브랜치 upstream 을 확인할 것.** win FaceLift 는 upstream 미설정이라
  `git pull --ff-only` 가 "Already up to date" 를 찍고 **아무것도 안 했다**(4커밋 뒤처짐, 260814 발견).
  single-branch 클론이면 `git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"` 도 필요.

## 2. 자산 위치 매핑 (WSL 경로 기준)

| 자산 | win 위치 | 실측 |
|---|---|---|
| 레포 | `~/dev/behavior-lab` | 260814 clone, `b8ea32a` |
| 레포-로컬 데이터 캐시 | `/mnt/d/data/behavior-lab/` | 3,070 파일 (calms21·nwucla·mammal_mouse·markerless_mouse_1·splits) |
| SBeA 원본 individual | `/mnt/d/data/raw/SBeA/individual` | 12G · 100파일 = 20세션 × (calib 1 + camera 4) |
| SBeA 원본 social | `/mnt/d/data/raw/SBeA/social` | 17G · 150파일 = 30세션. **260814 figshare 재취득** |
| SBeA release assets | `/mnt/d/data/raw/SBeA/sbea_release_assets` | 108M |
| SBeA 3D 키포인트 20세션 | `/mnt/d/data/derived/mac_backups_260813/gpu03_nonM5_260813/sbea_kp3d_full` | 183M · 41파일. **재생성에 GPU 13.2h** |
| SBeA 마스크 | 〃 `/sbea_masks` | 272파일 |
| behavior-lab 실험 산출 | `/mnt/d/data/derived/gpu03_offserver_260812/node_data_joon_260813/results/behavior_lab_essentials_260813` | 52M · 77파일 |
| KP 벤치마크 DLC 프로젝트 | `/mnt/d/data/derived/gpu03_offserver_260812/behavior-lab-kp-benchmark` | 9.4G · 6,879파일 |
| BehaveMAE 가중치 | `/mnt/d/data/behavior-lab/checkpoints/` | MABe22 233M · Shot7M2 111M |
| conda env 스냅샷 | 레포 `env_snapshots/*.yml` | dlc2·dlc3·sdannce·vame·kpms |
| SBeA 취득 매니페스트·러너 | 레포 `scripts/sbea_acquisition/` | figshare 직링크 100행/150행 |
| gpu03 홈 잡스크립트 아카이브 | `…/gpu03_offserver_260813/icml_repro_meta_260814/gpu03_home_scripts_260814.tgz` | 14파일. inspect_*·viz_*·install_* |
| ICML 재현 런 전량 | `…/gpu03_offserver_260813/icml_reproduction_runs_full_260814/` | 4.23 GiB · 19파일. 2런(20260429·20260516) |
| gpu03 홈 6디렉토리 | `…/gpu03_offserver_260813/gpu03_home6_260814/gpu03_home6_260814.tgz` | 521 MiB · 16,281파일. archives·outputs·wandb·backups·models·downloads |
| **FaceLift gaussians 3종** | `…/gpu03_offserver_260813/FaceLift_gaussians_260814/` | **30.55 GiB · 7,920파일**. gpu03↔win 전건 일치 |
| **git 로컬 전용분** | 〃 `gpu03_home6_260814/git_local_only_260814.tgz` | 4.4 MiB. **BS stash 15 · FaceLift stash 5 · PS-official unpushed bundle** |
| BS 논문 figures·metrics | 〃 `gpu03_home6_260814/bs_paper_artifacts_260814.tgz` | 446 KiB |
| 인계볼륨 잔여 6항목 | `…/gpu03_offserver_260813/handover_vol_260814/` | 3.6 GiB · 77,715파일. `pose_splatter_official_260807`(2.4G)·`raw`·`synthetic`·`ps_m5_baseline_260724_full`·`results`·`env_snapshots` |

> 검증 방법 = 크기 매니페스트 대조(`find -type f -printf "%s %p\n"` 양쪽 정렬 후 diff).
> `behavior_lab_essentials_260813` 은 gpu03↔win 77파일 전건 일치를 260814 확인했다.
> 크기 일치만으로는 부족한 자산은 SHA-256 을 쓸 것 — 260813 오프보딩 세션의 판정 기준과 동일.

## 3. 이미 안전한 것 (재확인 불요)

- BehaveMAE 서브모듈 로컬 작업 → `patches/BehaveMAE_260813.patch` 커밋·push 완료(`b8ea32a`).
  복원 = `cd external/BehaveMAE && git apply ../../patches/BehaveMAE_260813.patch`.
- `pose-splatter-official` 브랜치 `repro/260807_m5` → win WSL 에 내용 동일본 존재.
  gpu03↔win `git diff origin/master..HEAD` 해시 `820bcb7a…` 일치 확인(260814). 업스트림 push 권한 없음.
- gpu03 개인폴더 잔여(`exp_ve_pose_2608`·`g1_kp_magnitude`·`logs`·`outputs_*`·`results`·`paper_repro`)
  → 전부 `/mnt/d/data/derived/gpu03_offserver_260812/` 하위에 존재.

## 4. 셋업 — 260814 완료 상태

env `behavior-lab` (python 3.11). **`pytest` 93 passed** = gpu03 기준선과 동일.
torch `2.13.0+cu130` · `cuda.is_available() True` · RTX 3060 인식.

```bash
# WSL 안에서. conda activate 가 비대화형 셸에서 미적용되는 일이 잦아 절대경로를 쓴다
P=~/miniconda3/envs/behavior-lab/bin/python
cd ~/dev/behavior-lab

git submodule update --init --recursive          # ← 신규 클론이면 필수
cd external/BehaveMAE && git apply ../../patches/BehaveMAE_260813.patch && cd ../..
$P -m pip install -e ".[dev]"
$P -m pytest -q                                   # 93 passed
```

- **서브모듈 2단계를 빼면 `test_behavemae.py` 2건이 반드시 깨진다.** BehaveMAE 는 upstream
  (amathislab) 이라 push 권한이 없어, 로컬 수정분이 패치로만 존재한다.
- `PYTHONPATH` 수동 export 는 불필요 — `pyproject.toml` 의 pytest `pythonpath` 가 잡는다(`e63068c`).

- 형제 레포도 같이 맞춰야 규약 드리프트 테스트가 통과한다. 260814 정렬분:
  `sdannce-poc` 최신 pull · `FaceLift` 브랜치 **`refactor/mouse-extensions`** 체크아웃
  (`main` 에는 `configs/keypoints/mouse_22.yaml` 이 없다) · `BehaviorSplatter` 브랜치 **`dev`**.
- FaceLift 브랜치 전환 시 `git fetch origin "+refs/heads/*:refs/remotes/origin/*"` 선행.
  win 클론이 single-branch 라 원격 브랜치가 안 보인다.
- 파이프라인 전용 env 는 `env_snapshots/{dlc2,dlc3,sdannce,vame,kpms}.yml` 로 재생성.
  gpu03 리눅스 빌드 기준이라 win 에서 그대로 solve 되지 않을 수 있다 — 실패 시 핀을 완화할 것.
- `dlc2` 는 gpu03 에서도 tensorflow-cpu 였다(Blackwell 커널 미지원). RTX 3060 이라고 나아지지 않는다.

## 5. 자산별 사용법

§2 는 "어디 있나", 여기는 "어떻게 쓰나". 전부 WSL 기준, `P=~/miniconda3/envs/behavior-lab/bin/python`.

**SBeA 원본 (individual 20 / social 30세션)**
```bash
# 재취득·무결성 확인 겸용 — 크기가 매니페스트와 맞으면 SKIP 만 찍힌다 (다운로드 0)
./scripts/sbea_acquisition/download_sbea.sh individual /mnt/d/data/raw/SBeA/individual 4
./scripts/sbea_acquisition/download_sbea.sh social     /mnt/d/data/raw/SBeA/social     4
```
DL/FAIL 이 한 줄이라도 나오면 그 파일이 손상·부분본이다. **이게 SBeA 원본의 유일한 검증 절차다.**

**SBeA 3D 키포인트 (재생성 GPU 13.2h — 지우지 말 것)**
```bash
KP3D=/mnt/d/data/derived/mac_backups_260813/gpu03_nonM5_260813/sbea_kp3d_full
$P scripts/sweep_sbea_sessions.py "$KP3D"                              # 20세션 잔차 재확인
$P scripts/diag_sbea_residual.py --npz "$KP3D/rec10-M2-20221108-kp3d.npz"   # 단일세션 진단
```

**카메라 intrinsics 추정 (독립 3방법 대조)**
```bash
SBEA_DIR=/mnt/d/data/raw/SBeA/individual SBEA_SESSION=rec1-M7-20221108 \
  $P scripts/estimate_sbea_intrinsics.py
```

**BehaveMAE 가중치**
```bash
# 코드가 REPO/checkpoints/ 를 본다. 실물은 D: 에 두고 링크로 잇는다
ln -sf /mnt/d/data/behavior-lab/checkpoints/hBehaveMAE_MABe22.pth  checkpoints/
ln -sf /mnt/d/data/behavior-lab/checkpoints/hBehaveMAE_Shot7M2.pth checkpoints/
```
⚠️ Shot7M2 는 **가중치만** 있다. 대응 데이터 3.4G 는 부재라 `scripts/shot7m2_probe.py` 는 아직 못 돈다(§0).

**FaceLift gaussians (cinematic 렌더·BS ve_pipeline 입력)**
```bash
export FL_GAUSSIAN_ROOT=/mnt/d/data/derived/mac_backups_260813/gpu03_offserver_260813/FaceLift_gaussians_260814
# 이 값이 없으면 config 의 ${FL_GAUSSIAN_ROOT} 가 리터럴로 남아 FileNotFoundError 로 조기 실패한다
```
| 하위 | frames | 소비처 |
|---|---:|---|
| `M5t2_6view_alpha03_v3_maskcarve16k` | 3,600 | BS `canonical.yaml`(논문 정본) · cinematic FINAL paper preset |
| `filtered/a0.3_t0.2` | 3,600 | BS host config ×5 · `extract_per_frame_npz.py` |
| `ply_a0.3` | 720 (test·val) | BS `canonical_compare.py` |

260814 서버 정리로 `base_uniform_v2_6view_v2`·`match_16k`·`alpha10_maskcarve16k`·`filtered/a0.067`
4종(86.37 GiB)은 삭제했다. 소비처 0 판정이며 stage 1~3 재생성 가능 —
근거·재생성 절차 = FaceLift `docs/theory/GAUSSIAN_FILTERING_THEORY.md §4.1`.

**git 로컬 전용분 (stash·bundle) — push 로는 절대 안 나가는 것**
```bash
tar tzf $B/gpu03_home6_260814/git_local_only_260814.tgz
# BehaviorSplatter/stash_{0..14}.patch  FaceLift/stash_{0..4}.patch
# pose-splatter-official/unpushed.bundle
# 복원 = git apply <stash_N.patch>  /  git fetch <bundle> <branch>
```
🔴 stash 는 `git push` 대상이 아니다. 서버를 놓기 전 반드시 별도 회수해야 하는 유일한 git 자산.

**파이프라인 전용 conda env**
```bash
conda env create -f env_snapshots/dlc2.yml     # SBeA DLC (tensorflow-cpu)
conda env create -f env_snapshots/sdannce.yml  # SAM2 마스크
```
gpu03 리눅스 빌드 기준이라 solve 실패 가능. 그때는 핀을 완화하되 **yml 원본은 수정하지 말 것**(당시 상태 기록물).

**gpu03 홈 회수분 (tgz 2종)**
```bash
B=/mnt/d/data/derived/mac_backups_260813/gpu03_offserver_260813
tar tzf $B/icml_repro_meta_260814/gpu03_home_scripts_260814.tgz   # 잡스크립트 14파일
tar tzf $B/gpu03_home6_260814/gpu03_home6_260814.tgz              # 홈 6디렉토리 16,281파일
```
- 잡스크립트 = `inspect_*`·`viz_*`·`install_*`. 레포에 안 넣은 탐색용이라 필요할 때만 꺼낸다.
- 홈 6디렉토리 안에서 **가장 값나가는 것 = `backups/bs_presync_260729/{tools_never_committed,uncommitted}.tgz`**.
  이름 그대로 어디에도 커밋된 적 없는 산물이다. 중첩 tgz 이므로 꺼낸 뒤 한 번 더 푼다.
- `icml_repro_meta_260814/` 의 json 5건은 `icml_reproduction_runs_full_260814/` 의 부분집합이다(전량 회수로 대체됨).

**ICML 재현 런** (2런: `20260429_1757` · `20260516_1755`)
```bash
ls $B/icml_reproduction_runs_full_260814/icml_reproduction_runs/
```
런당 실체 = `results.json` · `dim_sweep.json` · `logs/*.log` · `per_frame/{ps,gs}_per_frame.npz`
(각 1129.8 / 1037.7 MiB). 20260429 에는 `verification_report.md`, 20260516 에는 `regen_canonical_d50.json` 추가.
수치만 필요하면 json 으로 충분하고, npz 는 per-frame 재현용이다.

> ⚠️ **각 런의 `ps_npz`·`gs_ply_a0.3`·`gs_maskcarve_16k` 는 파일이 아니라 심링크였다** —
> gpu03 `/node_data/dataset/animal_behavior/shared/…` 로 나간다. 사본에서는 **끊겨 있다**.
> 대상 실체(2.6G + 12G + 13G = 27.6 GiB)는 랩 공용 볼륨에 남아 후임자에게 인계되며,
> 우리 쪽 재생성 경로도 있다(`FaceLift/inference_mouse.py` ckpt→PLY · `cinematic_sequence.py::build_count_match_mask`).
> 즉 **입력 가우시안은 이 사본에 없고, 있는 것은 그 입력으로 계산된 per-frame 결과다.**

## 6. 코드 내 gpu03 경로

- 29곳이 `/node_data/joon`(23) · `/node_data_2/joon`(6) 을 하드코딩한다.
  주요 파일 = `scripts/{sbea_dlc_triangulate,diag_sbea_residual,sweep_sbea_sessions,triangulate_full}.py`,
  `scripts/{01..04}_*.sh`, `scripts/run_sbea_all_sessions.sh`, `configs/dataset/{li2023_m1,mammal_m1}.yaml`.
- **일괄 치환은 하지 않았다.** 데이터가 win 에서 단일 루트로 모여 있지 않아(원본 `raw/`, 파생 `derived/…`)
  1:1 매핑이 성립하지 않는다. §2 표를 보고 호출 시점에 인자로 넘기는 편이 안전하다.
- 별개 사안: 레포 스크립트 50줄/18파일이 `REPO_ROOT/"data"` 를 하드코딩한다.
  `${paths.data_dir}`(env `BEHAVIOR_LAB_DATA`) 경유는 11줄뿐이라 env 만 바꿔선 이전되지 않는다 (`README.md` §3).

## 7. 되돌아보기

gpu03 자산은 **8/12~8/13 두 차례 rsync 로 이미 win 에 내려와 있었다.** 이 세션이 한 일은
새 이관이 아니라 역대조(서버 목록 → 로컬 원장 차집합)와 누락 3건(conda yml 5종, Shot7M2 가중치,
레포 클론) 보충이다. 260813 오프보딩 세션이 남긴 판정 규칙 — *"완료 판정은 서술이 아니라 해시 대조의
exit code 로 한다"* — 를 그대로 적용했고, 실제로 서술만 믿었으면 conda env 5종은 놓쳤을 것이다.
