# win 개인 서버 이관 — gpu03 이후의 작업 거점

> 2026-08-14 작성. gpu03 접속 종료(AMILab 오프보딩, 실무마감 8/14)에 대응해 behavior-lab 의
> 작업 거점을 개인 win 서버로 옮긴 기록이자 진입점.
> **gpu03 기준 문서(`STORAGE_GUIDE.md`·`DATASETS.md` 의 `/node_data` 표기)는 이 시점부터 사료(史料)다.**
> 이 문서가 현행 경로의 정본.

## 0. 열린 이슈

- **SBeA `social` 30세션 부재** — gpu03 삭제, win 미보유. S3(social 처리) 착수 불가. 재취득 = 저자 배포처.
- **SHOT7M2 데이터 3.4G 부재** — gpu03 `/node_data/joon/data/shot7m2/` 삭제됨, win 미보유.
  `scripts/shot7m2_probe.py`·`scripts/bmae_probe_cv3.py` 가 이 죽은 경로를 참조 중.
  가중치(`checkpoints/hBehaveMAE_Shot7M2.pth` 111M)는 확보돼 있으므로 데이터만 재취득하면 복구.
- **코드 내 gpu03 절대경로 29곳** 미치환 (`/node_data/joon` 23 · `/node_data_2/joon` 6). §5 참조.
- **torch 계열 미설치** — `tests/test_models/test_behavemae.py` 만 실행 불가. §4 참조.
- **S1(저자 파이프라인 클론) 미착수** — 260729 핸드오프의 최우선 항목이 그대로 남음.

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
- `rsync` 는 win 쪽에 없다. mac→win 전송은 `scp <파일> kafkapple-win:D:/경로/`.

## 2. 자산 위치 매핑 (WSL 경로 기준)

| 자산 | win 위치 | 실측 |
|---|---|---|
| 레포 | `~/dev/behavior-lab` | 260814 clone, `b8ea32a` |
| 레포-로컬 데이터 캐시 | `/mnt/d/data/behavior-lab/` | 3,070 파일 (calms21·nwucla·mammal_mouse·markerless_mouse_1·splits) |
| SBeA 원본 individual | `/mnt/d/data/raw/SBeA/individual` | 12G · 100파일 = 20세션 × (calib 1 + camera 4) |
| SBeA release assets | `/mnt/d/data/raw/SBeA/sbea_release_assets` | 108M |
| SBeA 3D 키포인트 20세션 | `/mnt/d/data/derived/mac_backups_260813/gpu03_nonM5_260813/sbea_kp3d_full` | 183M · 41파일. **재생성에 GPU 13.2h** |
| SBeA 마스크 | 〃 `/sbea_masks` | 272파일 |
| behavior-lab 실험 산출 | `/mnt/d/data/derived/gpu03_offserver_260812/node_data_joon_260813/results/behavior_lab_essentials_260813` | 52M · 77파일 |
| KP 벤치마크 DLC 프로젝트 | `/mnt/d/data/derived/gpu03_offserver_260812/behavior-lab-kp-benchmark` | 9.4G · 6,879파일 |
| BehaveMAE 가중치 | `/mnt/d/data/behavior-lab/checkpoints/` | MABe22 233M · Shot7M2 111M |
| conda env 스냅샷 | 레포 `env_snapshots/*.yml` | dlc2·dlc3·sdannce·vame·kpms |

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

env `behavior-lab` (python 3.11) 생성·설치 완료. **`pytest` 86 passed · 1 skipped**
(`tests/test_models/test_behavemae.py` 는 torch 미설치로 제외).

```bash
# WSL 안에서. conda activate 가 비대화형 셸에서 미적용되는 일이 잦아 절대경로를 쓴다
P=~/miniconda3/envs/behavior-lab/bin/python
cd ~/dev/behavior-lab && $P -m pytest -q --ignore=tests/test_models/test_behavemae.py

# torch 계열은 아직 미설치 (대용량 다운로드라 보류). 필요해지면:
$P -m pip install -e ".[dev]"
```

- 형제 레포도 같이 맞춰야 규약 드리프트 테스트가 통과한다. 260814 정렬분:
  `sdannce-poc` 최신 pull · `FaceLift` 브랜치 **`refactor/mouse-extensions`** 체크아웃
  (`main` 에는 `configs/keypoints/mouse_22.yaml` 이 없다) · `BehaviorSplatter` 브랜치 **`dev`**.
- FaceLift 브랜치 전환 시 `git fetch origin "+refs/heads/*:refs/remotes/origin/*"` 선행.
  win 클론이 single-branch 라 원격 브랜치가 안 보인다.
- 파이프라인 전용 env 는 `env_snapshots/{dlc2,dlc3,sdannce,vame,kpms}.yml` 로 재생성.
  gpu03 리눅스 빌드 기준이라 win 에서 그대로 solve 되지 않을 수 있다 — 실패 시 핀을 완화할 것.
- `dlc2` 는 gpu03 에서도 tensorflow-cpu 였다(Blackwell 커널 미지원). RTX 3060 이라고 나아지지 않는다.

## 5. 코드 내 gpu03 경로

- 29곳이 `/node_data/joon`(23) · `/node_data_2/joon`(6) 을 하드코딩한다.
  주요 파일 = `scripts/{sbea_dlc_triangulate,diag_sbea_residual,sweep_sbea_sessions,triangulate_full}.py`,
  `scripts/{01..04}_*.sh`, `scripts/run_sbea_all_sessions.sh`, `configs/dataset/{li2023_m1,mammal_m1}.yaml`.
- **일괄 치환은 하지 않았다.** 데이터가 win 에서 단일 루트로 모여 있지 않아(원본 `raw/`, 파생 `derived/…`)
  1:1 매핑이 성립하지 않는다. §2 표를 보고 호출 시점에 인자로 넘기는 편이 안전하다.
- 별개 사안: 레포 스크립트 50줄/18파일이 `REPO_ROOT/"data"` 를 하드코딩한다.
  `${paths.data_dir}`(env `BEHAVIOR_LAB_DATA`) 경유는 11줄뿐이라 env 만 바꿔선 이전되지 않는다 (`README.md` §3).

## 6. 되돌아보기

gpu03 자산은 **8/12~8/13 두 차례 rsync 로 이미 win 에 내려와 있었다.** 이 세션이 한 일은
새 이관이 아니라 역대조(서버 목록 → 로컬 원장 차집합)와 누락 3건(conda yml 5종, Shot7M2 가중치,
레포 클론) 보충이다. 260813 오프보딩 세션이 남긴 판정 규칙 — *"완료 판정은 서술이 아니라 해시 대조의
exit code 로 한다"* — 를 그대로 적용했고, 실제로 서술만 믿었으면 conda env 5종은 놓쳤을 것이다.
