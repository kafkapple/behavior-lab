---
date: 2026-11-12
source: agent
---

# ~/dev · ~/data 레이아웃 정리 — 완료 레포트

**PRD**: [dev_디렉터리_정리 plan](~/.cursor/plans/dev_디렉터리_정리_28752fe6.plan.md) · MoC SSOT = [README §6](../README.md) · data = [DATA_RESULTS_LAYOUT.md](../DATA_RESULTS_LAYOUT.md)

---

## TL;DR

`~/dev` 평면 70+ → **4계층 MoC**(agent/platform·modules, poc/llm-study, archive, core flat) + 구경로 symlink. `~/dev/docs`·`output` 폐기. `~/data`는 alias 스키마 유지·문서화만; 대용량 디스크 이동·scratch·Win junction은 잔여.

---

## 단계별 (목표 → 전후 → 잔여)

| Phase | 목표 | 전 → 후 | 잔여 |
|---|---|---|---|
| **P0** | consumer·dirty 파악 | 미분류 루트 혼재 → 인벤토리 `scratch/261112_dev_layout/` | repo별 dirty 일괄 커밋 미실시 |
| **P1** | 제로리스크 삭제 | `output/`·`__pycache__`·`dev/docs/`·중복 `poc/multimodal-*` | stale `rg '~/dev/docs'` 메모리 일부 |
| **P2** | platform·llm-study | 루트 4 repo → `agent/platform`·`poc/llm-study` + symlink 7 | Win `mklink /J` 수동 |
| **P3** | wsl git | nogit 60KB → `kafkapple/wsl-migrate` `d943482` | migrate.conf gpu03 문구 점진 정리 |
| **P4** | scratch | — | usagi·2.3GB **미착수** |
| **P5** | archive·beast | avllm-archive·flux → `_archive/dev-repos` | symbolic/temporal **루트 유지**; Drive zip 미실시 |
| **P6** | nikke | 루트 repo → `mobile/tools/` + symlink | mobile README 커밋 미분리 |
| **P7** | hubs | → `agent/modules/` + symlink | — |
| **P8** | data/results | flat 실사 → DATA_RESULTS_LAYOUT §mac | `~/data/results` 6M 통합·INDEX diff 자동화 차기 |

---

## ~/dev 목표 트리 (간결)

| 계층 | 경로 | 역할 |
|---|---|---|
| **MoC** | `behavior-lab/docs/dev/` | README · playbooks · audits · data layout |
| **핵심 연구** | 루트 BS·behavior-*·FaceLift·sdannce·pose·beast·papers… | merge 금지, §1 표 |
| **agent** | `agent/` | runtime nested git |
| └ platform | `win-dashboard`, `wsl_migrate` | 엔드포인트 |
| └ modules | `research-hub`, `curation-hub` | 허브 PoC |
| **poc** | `poc/` + `poc/llm-study/` | 양자·로컬LLM·기존 poc git |
| **experiments** | `experiments/` | karpathy 벤치 (poc와 분리) |
| **archive** | `_archive/dev-repos/` | avllm-archive, flux_cond_img (+ 루트 symlink) |
| **mobile** | `mobile/` | `apps/Nikke` + `tools/nikke-spine-viewer` |
| **임시** | `scratch/`, `_scratch/` | 날짜 prefix; P4 잔여 |
| **메타** | papers, pages_repo, external, tools, **TIL** | 평면 유지 |

루트 **호환 symlink** (삭제 금지): `win-dashboard`, `wsl_migrate`, `quantumai-tech-poc`, `localLLM-benchmark`, `research-hub`, `curation-hub`, `nikke-spine-viewer`, `avllm-archive`, `flux_cond_img`.

---

## ~/data · ~/results (목표만 · 본 세션)

| | 목표 | 현황 |
|---|---|---|
| **data** | `<alias>/{raw,derived,processed}` + `_LAYOUT_MOVES.tsv` | alias·stub symlink **정상**; `BehaviorSplatter/` = mac viewer 캐시 유지 |
| **results** | `~/results/{project}/` | `~/data/results` 6M legacy **유지** |
| **정본** | BS `dataset_naming` · `DATASET_INDEX.yaml` | INDEX↔mac diff **문서만** (README § known gap) |

---

## 커밋·산출물

| Repo | SHA | 내용 |
|---|---|---|
| behavior-lab | `b97ab4e`, `84b8038` | taxonomy, playbooks, audits, DATA_RESULTS_LAYOUT |
| wsl-migrate | `ccc09eb`, `d943482` | 신규 repo + README |

**로컬 only**: `agent/platform|modules/README.md`, `experiments/README.md`, `poc/llm-study/README.md`, `scratch/261112_dev_layout/INVENTORY.md`

---

## 잔여 이슈 (우선순위)

1. Win junction (win-dashboard, wsl_migrate)
2. `scratch/`/`_scratch` 용량 (usagi grep 선행)
3. `~/data/results` → `~/results` 정리
4. avllm-symbolic/temporal archive 여부 (현재 **active** 판정)
5. behavior-lab·mobile 기타 dirty — 스코프 밖 커밋

**검증 한 줄**: `ls -la ~/dev/{win-dashboard,beast}` · `behavior-lab/docs/dev/README.md` §6
