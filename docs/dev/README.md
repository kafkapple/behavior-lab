# `~/dev` — 단일 진입점 (MoC)

> 여기서 "어디를 봐야 하는지"만 찾고, 내용은 링크된 정본에서 읽는다. **이 파일은 포인터만 담는다** —
> 사실을 여기 복사하면 그 순간부터 드리프트한다.
> 작성 2026-07-27 · 모든 경로는 작성 시점에 실재 확인함.
> **경로 표기**: 모두 `~/dev/` 기준 상대경로다 — 이 파일이 어디에 있든 동일하게 읽힌다.
>
> **실체는 `behavior-lab/docs/dev/` 안에 있고 `~/dev/` 의 셋(`README.md`·`DATASETS.md`·`STORAGE_GUIDE.md`)은
> 심볼릭 링크다.** `~/dev` 가 git 저장소가 아니라 추적이 안 되던 것을 260727 에 behavior-lab 으로 옮겨
> GitHub 백업에 태웠다. 편집은 어느 쪽을 열어도 같은 파일이다.

---

## 1. 저장소 지도

| 레포 | 역할 | 문서 진입점 | 위치 |
|---|---|---|---|
| **BehaviorSplatter** | GS-LRM 3D 재구성 · 4D deform (논문 본체) | `BehaviorSplatter/docs/references.md` → `BehaviorSplatter/docs/HANDOFF.md` | mac + gpu03 |
| **behavior-lab** | 포즈 로더 · `(T,K,D)` 정규화 · feature · 비지도 행동발견 · KP 벤치마크 | `behavior-lab/docs/README.md` | mac + gpu03 |
| **sdannce-poc** | 멀티뷰 어노테이션 · 6뷰 뷰어 · kp-guided SAM2 마스크 | `sdannce-poc/docs/README.md` | mac + gpu03 |
| **behavior-tools** | 영상 분할 · 프레임 추출 · 이미지 큐레이션 | `behavior-tools/docs/README.md` | mac only |
| **FaceLift** | 업스트림 레퍼런스 (KP22 정의 보유) | — | mac + gpu03 |
| **pose-splatter** | 별개 논문 구현 (arXiv 2505.18342) | `README.md` | gpu03 only |
| ~~mouse-kp-benchmark~~ | **deprecated** → behavior-lab 로 대체 | — | GitHub only (로컬 클론 260728 삭제) |

**누가 무엇을 소유하는가** = 경계 문서 3부작 (서로 참조, 내용 중복 없음):
`sdannce-poc/docs/repo_boundary.md` ·
`behavior-tools/docs/behavior_lab_boundary.md` ·
`behavior-lab/docs/260727_repo_consolidation_audit.md`

---

## 2. 좌표 · 규약

> **단일 진입점 = `behavior-lab/docs/conventions.md`.**
> 포즈 텐서 `(T,K,D)` · 스켈레톤 3종(KP22/rat23/SBeA16) · 카메라·투영 규약 3종 ·
> 마스크 npz 키 · 소프트스팟까지 전부 거기 있다.
>
> 표를 여기에 두지 않는 이유: 두 벌이면 갈라진다. 그 문서의 값은
> `behavior-lab/tests/test_conventions_doc.py` 가 실제 소스와 대조하므로 코드보다 뒤처질 수 없다.
> 좌표 관련 질문은 예외 없이 거기서 출발할 것.

---

## 3. 데이터 · 스토리지

> 🔴 **2026-08-14 이후 = win 개인 서버가 작업 거점.** gpu03 접속 종료(AMILab 오프보딩).
> 현행 경로 정본 = **`WIN_WORKSTATION.md`**. 아래 두 문서의 `/node_data` 표기는 그 시점부터 사료다.

| 주제 | 정본 |
|---|---|
| win 개인 서버 경로·셋업·부재 자산 (**현행**) | `WIN_WORKSTATION.md` |
| 데이터셋 온디스크 실사 (32개, 도메인별 경로·용량·포맷·shape) | `DATASETS.md` (gpu03 기준, 사료) |
| 스토리지 3-tier 정책 (NFS vs NVMe vs HDD, 쓰기 경로 규칙) | `STORAGE_GUIDE.md` (gpu03 기준, 사료) |
| 결과 폴더 SSOT (`~/results/{project}/`) | `CLAUDE.md` §5 |
| 코딩 원칙 | `CODING_PRINCIPLES.md` |

빠른 좌표:
- `~/data` — gpu03에서는 `/node_data/joon/data` 심링크, **mac에서는 실디렉토리**(1.4G). 머신 전역 데이터 허브.
- `behavior-lab/data/` — repo-local 파생 캐시(mac 156M, 전부 `.gitignore:22 /data/`). `~/data` 와 **역할이 다른 별개 저장소**이며 내용 중복 아님 (260802 실사: 상대경로 겹침 0, 동일파일 1개뿐 = `mammal_mouse/.../keypoints_22_3d.npz`, `kp_benchmark_v0.1.md:93` scp 착지점으로 의도된 사본).
- `~/dev/datasets` — 심링크 1개짜리 스텁이지 데이터셋 허브가 아니다.

경로 변경 주의: repo 스크립트 50줄/18파일이 `REPO_ROOT/"data"` 를 하드코딩(`test_e2e.py` 18줄 최다). `${paths.data_dir}` (env `BEHAVIOR_LAB_DATA`) 경유는 11줄뿐이므로 env만 바꿔선 이전 불가.

---

## 4. 리포트 (HTML, 자체 완결형)

| 리포트 | 내용 | 경로 |
|---|---|---|
| SBeA 파이프라인 | 저자 DLC → DLT 삼각측량 → SAM2. 2D 오버레이·잔차 분포·3D 스켈레톤·마스크 | `behavior-lab/outputs/sbea_report.html` |
| SAM2 변종 대조 (Rat7M) | v1 vs v4 파라미터 판정 — IoU 0.368, 연결성분 51→631 | `sdannce-poc/outputs/sam2_variant_compare.html` |
| SAM2 변종 대조 (s-DANNCE) | segment v1 vs v2 — IoU 0.970, 사실상 동일 | `sdannce-poc/outputs/sam2_segment_variant_compare.html` |

재생성: `behavior-lab/scripts/render_sbea_report.py` · `sdannce-poc/scripts/compare_sam2_variants.py`
(`outputs/` 는 gitignore 대상 — 리포트는 재생성 가능한 산출물이지 소스가 아니다.)

---

## 5. 파이프라인 한눈에

```
영상 + 캘리브
   └─ behavior-lab  scripts/sbea_dlc_triangulate.py     DLC 2D → DLT → (T,K,3) npz
        └─ sdannce-poc  segmentation/kp_sam2.py         키포인트 프롬프트 → mask npz
             └─ BehaviorSplatter  scripts/preprocess_run.py   (mask.source=npz_dir)
                  └─ opencv_cameras.json + RGBA 512² → GS-LRM 학습
```

레포 간 인계는 **파일 포맷**이지 코드 import 가 아니다. mask npz 키 규약(`animal_N` 쓰기 / `mask`·`ratN` 도 읽기)은
`sdannce-poc/docs/repo_boundary.md` §mask npz 키 규약 참조.

데이터셋 추가 = `sdannce-poc/configs/segmentation/*.yaml` 1개 (캘리브·키포인트·영상레이아웃 3축이 기존 조합이면 코드 0줄).
