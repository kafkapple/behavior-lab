# Repo Portfolio Audit

> Date: 2026-07-29  
> Scope: `/Users/joon/dev` 아래 Git working tree 전수 조사  
> Mode: architecture and over-engineering audit + 비-Engram 저위험 정리 적용

## TL;DR

- 발견 Git 경계 155개
- archive·backup·external·benchmark 산출물 103개 제외
- first-party/활성 후보 52개 감사
- 추적 파일 13,274개
- 유지 코드 추정 2,784파일·585,328 LOC
- 2026-07 활동 31개
- dirty working tree 26개
- commit 0 working tree 0개 (감사 시점 2개였으나 baseline commit 완료)
- 통합 원칙: repo merge보다 공통 생성기·패키지 계약·artifact 정책 통합 우선
- 비-Engram 1차 실행 완료: 생성물 1,095,186줄 untrack, 폐기 코드 929줄 순감소
- Engram 5종은 변경하지 않았으며 별도 계획으로 분리

## 1. Scope and Method

### 1.1 Included

- 독립 `.git` 경계를 가진 first-party working tree
- tracked code 기준 언어·파일·LOC·테스트·manifest·최근 commit 집계
- README·registry·repo-boundary 문서와 실제 import 경로 대조
- exact content hash로 cross-repo 코드 중복 검사
- 대형 파일·`sys.path` 주입·단일 구현 추상화·tracked vendor 검사

### 1.2 Excluded

- `.archive/`, `archive/`, `.backup_*`
- `external/`
- `experiments/karpathy-ponytail-bench/runs/`의 생성 Git fixture
- `node_modules/`, virtualenv, build output, dataset의 LOC

### 1.3 Limitations

- correctness·security·performance 결함은 `ponytail-audit` 범위 밖
- dirty repo 26개는 사용자 변경 보호를 위해 정리·테스트 미실행
- LOC는 비교용 근사치
- repo 삭제·merge 전에는 remote·consumer·복구 경로 별도 검증 필요

## 2. Portfolio Statistics

| 지표 | 값 | 해석 |
|---|---:|---|
| 전체 `.git` 경계 | 155 | benchmark fixture가 경계 수를 크게 부풀림 |
| 제외 경계 | 103 | archive·backup·external·generated |
| 감사 repo | 52 | 현재 portfolio 후보 |
| tracked files | 13,274 | Jekyll vendor 포함 |
| code files | 2,784 | generated/vendor 제외 |
| code LOC | 585,328 | 연구 upstream/reimplementation 포함 |
| test matches | 1,832 | vendored Jekyll test 포함으로 상한값 |
| 2026-07 활동 repo | 31 | 최근 활동 기준 |
| 이전/정체 repo | 19 | archive 판정은 별도 필요 |
| commit 0 repo | 0 | 초기 2개였던 `curation-hub`, `mobile/apps/habitflow` baseline 확정 |
| dirty repo | 26 | 일괄 이동·삭제 금지 |

## 3. Module Disposition

### 3.1 Keep Separate

| 모듈군 | 판정 | 근거 |
|---|---|---|
| `BehaviorSplatter` | 유지 | 논문 본체·GS-LRM·4D deform 소유 |
| `behavior-lab` | 유지 | `(T,K,D)` 분석·학습·평가 플랫폼 소유 |
| `sdannce-poc` | 유지 후 축소 | 멀티뷰 annotation·SAM2·DANNCE 규약 소유 |
| `behavior-tools` | 유지 | 영상 분할·프레임·collection utility 소유 |
| `FaceLift` | reference 유지 | upstream/reimplementation 역할 |
| `ai-engram` | 유지 | 독립 배포 가능한 연구 reference package |
| `agent-monitor` | 유지 | cross-agent observability 단일 UI |
| `gyeol` | 유지 | identity·episode memory SSOT |
| `media-hub` | monorepo 유지 | 공유 media package와 여러 소비 앱의 결합이 실재 |
| `mobile/apps/{Game,Learn,Life,MediaViewer}` | repo 분리 유지 | 배포 단위·bundle ID·release cadence 독립 |
| paper repos | 유지 | 제출·버전·LaTeX 이력 독립 |

### 3.2 Integrate by Contract

| 모듈군 | 제안 | 하지 않을 것 |
|---|---|---|
| behavior 4종 | 파일 포맷·좌표 규약·package dependency만 통합 | mega-repo merge |
| `avllm-symbolic` + `avllm-temporal` | symbolic wrapper를 설치형 package로 만들고 temporal이 dependency로 사용 | sibling checkout `sys.path` import |
| Engram 5종 | `engram-core`의 parameterized builder 1개 사용 | domain log/schema를 한 repo로 강제 merge |
| mobile 4종 | `mobile/Packages`의 CoreKit·DesignKit만 공유 | app source 통합 |
| media subprojects | `media-hub/shared`의 stable API만 공유 | DB·UI를 하나의 giant app으로 통합 |

### 3.3 Archive or Resolve

| 대상 | 판정 | 다음 게이트 |
|---|---|---|
| `curation-hub` | 유지·baseline 확정 | stdlib PoC와 self-check 12개 통과, initial commit 완료 |
| `mobile/apps/habitflow` | 유지·baseline 확정 | Expo domain test 14개 통과, initial commit 완료 |
| `mobile/experiments/playground` | 보류 | Game 포팅의 reference consumer 여부를 release 후 판단 |
| `learning-lab/research/grid-cells` | 보류 | NIP 비교본은 현재 소멸; 대규모 uncommitted research source/results의 보존 경계 확정 전 변경 금지 |
| `pages_repo/markdown-cv` | 유지 | 공개 사이트가 `/markdown-cv/`를 실제 참조 |
| 오래된 PoC 3종 | dormant/archive 분류 | 최근 consumer·재현 필요성 확인 |

## 4. Ranked Ponytail Findings

1. `native:` tracked Bundler tree 7,316파일·1,379,634줄 제거. `Gemfile`/lockfile + `bundle install` 또는 GitHub Pages build 사용. [`pages_repo/kafkapple.github.io/vendor`](../pages_repo/kafkapple.github.io/vendor)
2. `shrink:` Engram 5개 `build_config.sh`의 skill frontmatter·annotation parser·dual-output 생성 약 634줄을 `engram-core` parameterized builder 하나로 축소. 모듈별 wrapper는 metadata와 post-hook만 유지. [`agent/engram-core/scripts`](../agent/engram-core/scripts)
3. `delete:` 생성된 `CLAUDE.md`·`AGENTS.md`의 tracked 복제 약 4,582줄을 source release artifact로 전환. `AGENT_COMMON.md` + adapter fragment를 SSOT로 유지하고 install/build에서 생성. [`agent/engram-comms`](../agent/engram-comms)
4. `native:` sibling checkout을 `sys.path`로 주입하는 AVLLM import 제거. `avllm-symbolic`을 `pyproject.toml` package로 설치하고 temporal이 명시 dependency 사용. [`avllm-temporal/avllm_temporal/evaluate.py`](../avllm-temporal/avllm_temporal/evaluate.py)
5. `delete:` `sdannce-poc`의 SAM1 폐기 코드 5파일·895줄 제거. 역사 문서 3개는 보존하고 현재 SAM2 경로만 유지. [`sdannce-poc/archive`](../sdannce-poc/archive)
6. ~~`delete:` commit 0인 `curation-hub`·`habitflow` 경계 해소~~ — **완료**. 테스트 후 각각 initial commit으로 보존했다. [`curation-hub`](../curation-hub)
7. `shrink:` Pixiv viewer의 6,653줄 f-string UI와 후반 `page_html` monkey-patch를 Jinja/static asset으로 흡수. 기존 router extraction을 완료하고 override를 원 정의에 병합. [`media-hub/pixiv-manager/src/pixiv_manager/viewer.py`](../media-hub/pixiv-manager/src/pixiv_manager/viewer.py)
8. ~~`native:` benchmark 내부 Git 경계 제거~~ — **철회**. 103개 fixture는 `bench.py`가 `git diff` 기반으로 점수화하기 위해 의도적으로 생성한다. 일반 repo inventory에서만 제외한다. [`experiments/karpathy-ponytail-bench`](../experiments/karpathy-ponytail-bench)
9. ~~`delete:` NIP와 `grid-cells`의 exact-copy 제거~~ — **해소**. 재검증 시 NIP 비교본 경로가 더 이상 존재하지 않아 중복 제거 과제는 소멸했다. 현재 grid repo의 미커밋 연구 작업만 별도 보호한다.
10. ~~`yagni:` `behavior-lab` Protocol/ABC 하나 제거~~ — **철회**. Protocol은 공개 structural API이고 ABC는 4개 구현의 공통 평가 로직을 소유해 역할이 다르다.
11. ~~`shrink:` `behavior-lab/scripts/test_e2e.py`를 pytest로 전환~~ — **철회**. 파일은 unit-test runner가 아니라 다중 dataset 실험·보고서 생성기다. 전환해도 도메인 로직이 줄지 않는다.
12. `native:` repo-local virtualenv/build/vendor를 source tree 밖 cache로 재생성. 즉시 회수 가능 후보 약 5GB이며 source 삭제와 별도 승인 필요. [`mobile`](../mobile)

`applied net: -1,096,095 tracked lines`  
`runtime dependency delta: 0` (AVLLM의 기존 의존성을 package metadata에 명시했을 뿐 신규 기능 의존성은 없음)

> 적용 절감의 대부분은 Jekyll 생성물 1,095,186줄 untrack이다. 직접 source 단순화는
> SAM1 archive 895줄과 Pixiv 중복 renderer 34줄이다. 초기 8k–12k 추정은
> consumer·dirty-state 재검증 없이 실행하기에 과도해 철회했다.

## 4.1 Applied Results (2026-07-29)

| Repo | 적용 | 검증 | Commit |
|---|---|---|---|
| `pages_repo/kafkapple.github.io` | tracked `vendor/`, `_site/`만 index에서 제거; disk와 ignore 유지 | `bundle check`, `jekyll build` 성공 | `aef556d` |
| `sdannce-poc` | superseded SAM1 code 5개 삭제, 역사 문서 보존 | `pytest -q tests/test_io.py`: 15 passed | `65eddac` |
| `curation-hub` | commit-0 경계 해소, 현재 PoC baseline 보존 | self-check 12 passed | `04f4b99` |
| `mobile/apps/habitflow` | commit-0 경계 해소, agent/cache/backup 제외 | domain tests 14 passed | `472426c` |
| `avllm-symbolic` | `avllm_symbolic` 설치 package와 호환 shim 추가 | wheel 내용·metadata 확인 | `3b03dad` |
| `avllm-temporal` | sibling `sys.path` 제거, package import 사용 | `./run check`: placement/trials all pass | 미커밋¹ |
| `media-hub` | Pixiv `page_html` monkey-patch를 원 정의에 병합 | 변경 전후 동일 32 pass/1 known fail | `0ae1bb6` |

¹ `avllm-temporal`은 기존 사용자 변경과 같은 파일이 겹쳐 자동 commit하지 않았다.
수정은 유지되어 있으나 owner diff 분리 후 commit해야 한다.

### 보호한 기존 변경

- Jekyll repo의 `Gemfile.lock`, `assets/img/logo_64.png`는 건드리지 않았다.
- `avllm-temporal`의 README·data·placement·notebook 등 기존 dirty 변경은 보존했다.
- `grid-cells`의 대규모 dirty 연구 작업은 변경하지 않았다. 과거 NIP exact-copy는 현재 소멸했다.
- 어디에도 push하지 않았다.

## 5. Proposed Execution Plan

### Phase 0 — Freeze and Classify — **완료**

- 52 repo에 `active/maintained/experimental/dormant/archive` 판정
- dirty 26 repo의 변경 owner와 목적 기록
- commit 0 repo 두 개를 테스트 후 baseline commit
- 성공 기준: 미분류 repo 0개

### Phase 1 — Zero-risk Hygiene

- benchmark/archive nested `.git` inventory 확정
- ignored build·venv·vendor 용량 목록 생성
- 삭제 명령 없이 recoverability와 재생성 명령 검증
- 성공 기준: 각 후보에 keep/delete/rebuild 판정과 예상 회수량 존재

### Phase 2 — SSOT Cuts

- Jekyll vendor를 lockfile build로 재현 — **완료**
- exact-copy grid-cell 파일의 단일 SSOT 지정 — **비교본 소멸로 종료**
- Engram builder를 `engram-core`로 단일화 — **별도 실행안으로 분리**
- 성공 기준: before/after 산출물 hash 또는 semantic diff 동일

### Phase 3 — Dependency Boundaries

- `avllm-symbolic` 설치형 package화 — **완료**
- temporal의 sibling path hack 제거 — **완료, commit 분리 대기**
- behavior 4종은 기존 file-contract 유지
- mobile은 Package dependency만 유지
- 성공 기준: arbitrary CWD에서 import와 smoke test 통과

### Phase 4 — Hotspot Simplification

- Pixiv viewer의 중복 override 병합 — **완료**
- Jinja 전환은 현재 34줄 절감보다 migration 비용이 커 보류
- behavior-lab E2E/Protocol/ABC 단순화 — **consumer audit 후 철회**
- 성공 기준: route snapshot·regression test·E2E 결과 동일

### Phase 5 — Archive and Governance

- dormant repo를 cold archive로 이동
- `/Users/joon/dev/README.md`는 repo pointer만 유지
- portfolio inventory를 정기 생성하되 새 dashboard는 만들지 않음
- 성공 기준: root discovery 결과가 의도된 활성 repo만 표시

## 6. Decision Gates

| 우선순위 | 결정 | 권고 |
|---|---|---|
| 완료 | commit 0 repo 두 개 | 둘 다 보존·baseline commit |
| P0 | dirty 26 repo | 이동·삭제 전 clean snapshot |
| P1 | Jekyll vendor | clean clone build 검증 후 untrack |
| P1 | AVLLM package boundary | 두 repo 유지 + dependency 명시 |
| P1 | Engram generated files | central builder부터, untrack은 후속 |
| P2 | behavior repo merge | 하지 않음 |
| P2 | media-hub split | 하지 않음; hotspot만 축소 |

## 7. Independent Health Signal

- `agent_system_health.py`: 28/29 통과
- 미통과: 2026-07-23 검증 이후 `model-tier-ssot.md`, `skill_families.md`, `universal-flags.md` 수정
- 이 항목은 과설계 감사가 아닌 config drift 검증 작업으로 별도 처리

## Appendix A — 52 Repo Inventory

| Repo | LOC | 판정 |
|---|---:|---|
| BehaviorSplatter | 48,688 | active·유지 |
| FaceLift | 78,507 | reference·유지 |
| MAMMAL_mouse | 38,279 | reference/dormant 재분류 |
| MonoFusion | 27,665 | reference/dormant 재분류 |
| TIL | 752 | dormant 유지 |
| agent/agent-monitor | 16,355 | maintained·유지 |
| agent/agent-tools | 1,766 | maintained·유지 |
| agent/ai-router | 1,297 | experimental·유지 |
| agent/engram-comms | 4,217 | active·builder 통합 |
| agent/engram-core | 970 | active infra·유지 |
| agent/engram-english | 3,612 | active·builder 통합 |
| agent/engram-learn | 2,320 | active·builder 통합 |
| agent/engram-routine | 2,350 | active·builder 통합 |
| agent/engram-writing | 697 | experimental·builder 통합 |
| agent/gyeol | 1,705 | active SSOT·유지 |
| agent/magi-researchers | 801 | experimental 재평가 |
| agent/mcp-servers/github | 69,949 | external-like maintained·fork 경계 유지 |
| agent/plugins/ponytail | 5,308 | plugin source·유지 |
| ai-engram | 5,717 | active research package·유지 |
| avllm-symbolic | 4,353 | active·package화 |
| avllm-temporal | 3,162 | active·symbolic dependency 명시 |
| behavior-lab | 25,162 | active·유지 |
| behavior-tools | 3,543 | maintained·유지 |
| brain-inspired-podcast | 2,182 | maintained·유지 |
| curation-hub | baseline committed | active PoC·유지 |
| learning-lab | 19,626 | learning hub·경계 정리 |
| learning-lab/research/grid-cells | 1,034 | dirty 연구 작업 보존 경계 후 재판정 |
| learning-lab/research/snn_playground | 16,462 | research snapshot·유지 |
| llm-lab | 14,640 | learning hub·경계 정리 |
| llm-lab/vision-foundation-framework | 2,390 | nested repo 필요성 재평가 |
| media-hub | 92,094 | active monorepo·hotspot 축소 |
| mobile/Packages | 313 | shared package·유지 |
| mobile/apps/Game | 565 | active app·유지 |
| mobile/apps/Learn | 1,919 | active app·유지 |
| mobile/apps/Life | 4,138 | active app·유지 |
| mobile/apps/MediaViewer | 2,967 | active app·유지 |
| mobile/apps/habitflow | baseline committed | 완성 Expo app·유지 |
| mobile/experiments/playground | 1,480 | archive 후보 |
| multimodal-embedding-poc | 1,260 | active PoC·milestone 후 archive |
| neural_information_processing | 12,635 | research hub·유지 |
| pages_repo/cv-academic | 0 | active document source·유지 |
| pages_repo/kafkapple.github.io | 12,816 first-party | vendor untrack |
| pages_repo/markdown-cv | 1,223 | archive 후보 |
| pages_repo/pages | 6,819 | 목적/배포 consumer 재평가 |
| papers/ibs2026-neuroai-plan | 0 | paper·유지 |
| papers/icml2026-ai4science-3dgs-mouse | 0 | paper·유지 |
| papers/ksbns2026-k-brain-mouse | 181 | paper·유지 |
| papers/neurips2026-adaptive-intelligence | 0 | paper·유지 |
| poc/2605_WM_VLA | 121 | dormant/archive 후보 |
| poc/lpwm | 24,071 | upstream-heavy PoC·snapshot 분리 |
| poc/qwen-multimodal | 8,839 | dormant/archive 후보 |
| sdannce-poc | 10,378 | active·deprecated archive 축소 |

---

## 2026-07-31 Resume 결과 (핸드오프 흡수 후 원본 삭제)

`02_INBOX/Handoff/System/260731_Handoff_Repo_Portfolio.md`를 이 문서로 흡수하고 사용자 승인 하에 삭제했다. 아래는 그 핸드오프에만 있던 내용이다. socialmapper 조사 결과는 `sdannce-poc` 저장소 문서·커밋(`ca61b3c`·`8563a24`·`a8c9fe3`·`31b70fd`)에 이미 있어 옮기지 않는다.

**P1 off-device backup — 보류 결정 (2026-07-31)**
- `curation-hub`(`220fa57`)·`habitflow`(`472426c`)에 remote 없음. 사용자 결정 = 지금은 private remote 생성하지 않고 단기 PoC로 로컬 유지.
- 재개 조건: 둘 중 하나라도 배포·협업·장기 보존으로 전환할 때.

**gitignore 노출 전역 스캔 — 무해 확인**
- 관측수단: portfolio 7개 repo에 `git ls-files --others --ignored --exclude-standard --directory` 후 build/vendor/data 제외.
- 무해 확정: `curation.db`(재생성 가능), habitflow `ios/`(expo prebuild), `pages_repo/_obsidian/`(vault 심링크 — `find -type f`가 0을 반환해 오판 직전이었음).
- 잔여 🟡 1건: `media-hub/media-toolkit/_archive/260607_grid_removal/{grid.py,grid_adapter.py}` 248줄이 tracked 조상 `f1b3d9d`와 내용이 달라 델타가 git에 없음. 의도적 폐기본이라 미조치.
- 미스캔: Engram 5종·grid-cells·mobile playground (별도 handoff 소관).

**감사 표 정정 2건**
- `media-hub`의 "clean"은 시점 오기 — 스냅샷과 동시간대에 타 세션이 작업 중이었고 22:42에 그 세션이 직접 커밋(`a4069aa`)·push해 해소. blast radius 회피로 무변경 유지한 판단이 맞았다.
- `kafkapple.github.io` 실제 경로 = `dev/pages_repo/kafkapple.github.io`.

**최종 상태**: portfolio 7개 repo 전부 clean·origin 동기화. sdannce-poc `31b70fd`, mac·gpu03 일치, `pytest tests/` 15 passed 2 skipped.

**유일한 open**: `sdannce.socialmapper` 26파일 재포팅 여부. 권고 = 지금 하지 말 것(8/11 엔트리포인트 정상, 차단 사실이 코드·문서에 정확히 기재됨). 필요해지는 시점에 `integrated_viewer`부터 착수.
