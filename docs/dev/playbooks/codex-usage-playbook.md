# Codex Usage Playbook for Joon

작성일: 2026-06-12  
범위: Codex CLI / IDE / App / Web, AGENTS.md, skills, MCP, subagents, memories, 연구/코딩/문서화 워크플로

## BLUF

Joon에게 Codex를 가장 잘 쓰는 방법은 "대화형 코딩 도우미"가 아니라 "검증 가능한 작업 단위 실행기"로 쓰는 것이다.

우선순위는 다음 순서다.

1. 모든 주요 repo에 짧고 구체적인 `AGENTS.md`를 둔다.
2. 요청을 "재현 방법 + 성공 조건 + 검증 명령 + 금지사항"으로 쪼갠다.
3. 반복 작업은 skill로 만든다. 특히 논문/노트북/실험/배포/리뷰/이메일/문서 변환.
4. 큰 작업은 `plan -> implement -> verify`로 분리하고, 중요한 작업은 verifier 시점으로 다시 본다.
5. 병렬 조사는 subagents로 시키되 같은 파일을 동시에 수정시키지 않는다.
6. MCP는 Figma, GitHub, OpenAI docs, Notion, Linear처럼 "외부 시스템 상태"가 필요한 곳에만 붙인다.
7. 메모리는 보조 회상층으로 쓰고, 필수 규칙은 `AGENTS.md`나 repo 문서에 둔다.

## Source Basis

공식 문서 기준:

- OpenAI Codex overview: Codex는 코드 작성, 코드베이스 이해, 리뷰, 디버깅, 반복 개발 작업 자동화를 지원한다. <https://developers.openai.com/codex>
- Prompting: Codex는 file read/edit/tool call 루프로 작업하며, 검증 가능한 재현/테스트/린트 정보를 줄수록 품질이 좋아진다. 복잡한 작업은 작은 단계로 나누는 것을 권장한다. <https://developers.openai.com/codex/prompting>
- Workflows: Codex를 명확한 context와 definition of done을 가진 teammate처럼 다루라고 권장한다. IDE/CLI/cloud별로 코드 설명, 버그 수정, 테스트 작성, UI 프로토타입 workflow를 제시한다. <https://developers.openai.com/codex/workflows>
- Customization: Codex customization layer는 `AGENTS.md`, memories, skills, MCP, subagents로 구성된다. <https://developers.openai.com/codex/concepts/customization>
- AGENTS.md: Codex는 작업 전 `AGENTS.md`를 읽고, global -> repo root -> current directory 순으로 instruction chain을 만든다. 가까운 디렉터리 지침이 더 나중에 들어가므로 우선한다. <https://developers.openai.com/codex/guides/agents-md>
- Skills: skill은 `SKILL.md`와 선택적 scripts/references/assets로 구성되고, progressive disclosure로 필요할 때만 전체 지침을 읽는다. <https://developers.openai.com/codex/skills>
- MCP: MCP는 Codex에 외부 문서, 브라우저, Figma 같은 third-party tools/context를 연결한다. CLI/IDE에서 지원된다. <https://developers.openai.com/codex/mcp>
- Subagents: Codex는 명시적으로 요청할 때 병렬 specialized agents를 spawn하고 결과를 취합할 수 있다. 토큰 비용이 커지므로 복잡하고 병렬화 가능한 작업에 쓴다. <https://developers.openai.com/codex/subagents>
- Sandbox: sandbox는 Codex command 실행의 파일/네트워크 경계를 정하고, approval policy와 함께 동작한다. <https://developers.openai.com/codex/concepts/sandboxing>
- Memories: memories는 기본 off이고, 안정적 선호/워크플로/관례/known pitfalls를 다음 thread에 전달한다. 필수 팀 규칙은 memory가 아니라 `AGENTS.md`에 둔다. <https://developers.openai.com/codex/memories>

연구/실사용 근거:

- AGENTS.md 연구: 10개 repo, 124개 PR에서 `AGENTS.md` 존재가 median runtime 28.64% 감소, output token 16.58% 감소와 연관되었다. <https://arxiv.org/abs/2601.20404>
- coding-agent misalignment 연구: 20,574 세션에서 agent 실패는 intent 해석, rule 준수, scope 제한, 실행/보고 오류 등으로 나타났고, visible resolution의 91.49%가 사용자 명시 교정을 필요로 했다. <https://arxiv.org/abs/2605.29442>
- action bias 연구: stale issue처럼 "고치지 않는 것이 정답"인 200개 task에서 최신 agent도 35-65% undesirable code change를 제안했다. "수정하지 않아도 성공"을 명시해야 한다. <https://arxiv.org/abs/2605.07769>
- vibe coding 연구: AI coding은 전문성을 없애지 않고 context 관리, 빠른 평가, manual/AI 전환 판단으로 재배치한다. <https://arxiv.org/abs/2506.23253>
- 전문 개발자 연구: 숙련자는 AI agent를 생산성 도구로 쓰되 설계/품질 agency를 유지한다. <https://arxiv.org/abs/2512.14012>

## Mental Model

Codex는 "내가 직접 치는 코드를 대신 타이핑하는 autocomplete"가 아니다. 더 좋은 모델은 다음이다.

- 작은 작업: 빠른 pair programmer.
- 중간 작업: repo-aware executor.
- 큰 작업: plan-run-verify loop.
- 불확실한 작업: 조사자 + 비판자 + 구현자를 분리한 orchestration.
- 반복 작업: skill 또는 MCP-backed workflow.

가장 흔한 실패는 "너무 큰 요청", "검증 불가능한 요청", "파일/명령을 안 알려준 요청", "수정하지 않아야 할 때도 뭔가 고치는 action bias"다.

## Priority 1: AGENTS.md를 작업 단위별로 정리

`AGENTS.md`는 ROI가 가장 높다. 공식 문서와 연구가 모두 같은 방향을 가리킨다.

Joon의 현 상태:

- `/Users/joon/dev` 루트는 Gyeol/메모리 중심 instruction이 강하게 들어가 있다.
- 2026-06-13 현재 `behavior-lab`, `BehaviorSplatter`, `papers/cvpr2026`에 repo-scoped `.agents/skills`가 추가되었다.
- 발견된 project `AGENTS.md`: `/Users/joon/dev/agent/gyeol/AGENTS.md`, `/Users/joon/dev/external/NEMS/AGENTS.md`.

추천:

- 각 주요 repo의 root `AGENTS.md`는 80-150줄 이하로 유지한다.
- repo별 build/test/lint 명령, 데이터 위치, 절대 금지사항, 결과물 경로 규칙만 둔다.
- 하위 폴더별 규칙이 다르면 nested `AGENTS.md` 또는 `AGENTS.override.md`를 둔다.
- 반복 교정이 2번 발생하면 `AGENTS.md`에 한 줄 추가한다.

Joon용 기본 템플릿:

```markdown
# AGENTS.md

## Scope

이 파일은 이 repo에서 Codex가 항상 따라야 할 규칙이다.

## Commands

- Smoke: `<command>`
- Unit tests: `<command>`
- Lint/typecheck: `<command>`
- Long-running jobs: 먼저 사용자 확인.

## Project Rules

- 데이터/결과물 기본 경로:
- 수정 금지 경로:
- 외부 다운로드/대용량 실행은 승인 후 진행.
- 실험 결과는 `reports/` 또는 `notebooks/`에 요약과 command를 함께 남긴다.

## Verification

- 코드 변경 후 최소 관련 테스트를 실행한다.
- 테스트를 못 돌리면 이유와 대체 검증을 보고한다.
- "코드 변경이 필요 없음"도 성공 결과일 수 있다.
```

## Priority 2: Prompt를 작업 계약서처럼 작성

공식 prompt guidance의 핵심은 "검증 가능한 context"다. 좋은 prompt는 결과물이 아니라 작업 계약을 준다.

기본 구조:

```text
Task:
<무엇을 바꿀지>

Context:
- Relevant files: @path1 @path2
- Current behavior:
- Desired behavior:

Constraints:
- Do not change:
- Keep API/format compatible:
- Smallest reasonable diff:

Verification:
- First reproduce/inspect:
- Then run:
- Report commands and results.

Definition of done:
- ...
```

버그 수정 prompt:

```text
Bug: <증상>

Repro:
1. <명령/화면/입력>
2. <관찰되는 실패>

Constraints:
- 원인 확인 전 patch하지 마.
- 이미 고쳐져 있으면 코드 변경 없이 "no code change needed"로 끝내.
- public API shape은 유지.

Start by reproducing or proving the failure path. Then patch the smallest scope and run the smallest relevant test.
```

연구/노트북 prompt:

```text
Read @README.md and @notebooks/<name>.ipynb.
Goal: <분석 질문>

Produce:
- notebook/code change if needed
- one markdown summary under reports/
- exact commands used
- limitations and next experiments

Verification:
- Run the notebook/script enough to validate outputs.
- Do not invent results if data is missing.
```

리뷰 prompt:

```text
Review this branch vs main.
Prioritize findings over summary.
Focus on: correctness, data loss, security, reproducibility, missing tests.
Spawn separate subagents for security, tests, and maintainability if useful.
Do not edit files unless I explicitly ask for fixes.
```

## Priority 3: 네 로컬 스킬 세트를 적극 사용

현재 전역 스킬은 이미 생산성 높은 형태다. 특히 다음을 우선적으로 호출하면 좋다.

| Use case | Skill | 사용 기준 |
|---|---|---|
| 중요한 구현/불확실한 변경 | `orchestrate-plan-verify` | plan/implement/verify 분리, production/security/data 변경 |
| 구조/전략 판단 | `deliberate-lite` | 아키텍처, 연구 방향, 여러 관점 필요 |
| 코드 리뷰 | `review-modes` | quick/deep/audit/fact/devil 선택 |
| GitHub PR 댓글 대응 | `github:gh-address-comments` 또는 `gh-address-comments` | unresolved review thread 처리 |
| CI 실패 분석 | `github:gh-fix-ci` 또는 `gh-fix-ci` | GitHub Actions 로그 기반 |
| 논문/문서 파일 | `pdf`, `doc`, `spreadsheet` | 파일 포맷 보존/렌더링 검증 |
| 노트북 생성/수정 | `jupyter-notebook` | `.ipynb` scaffold/clean template |
| UI 검증 | `playwright` | real browser screenshot/flow |
| Figma 구현 | `figma`, `figma-implement-design` | Figma URL/node 기반 |
| 배포 | `vercel-deploy`, `netlify-deploy`, `cloudflare-deploy`, `render-deploy` | 배포 대상별 |
| 보안 | `security-threat-model`, `security-best-practices`, `security-ownership-map` | 명시적 보안 요청 |
| Notion/Linear | `notion-*`, `linear` | 외부 업무 시스템 연동 |

Joon에게 특히 중요한 조합:

- `deliberate-lite` -> `orchestrate-plan-verify`: 연구/아키텍처 결정 후 구현.
- `review-modes deep + fact + verify`: 논문 수치, API, 최신 모델/문서가 섞인 작업.
- `playwright` + `figma-implement-design`: frontend 또는 시각 결과물.
- `pdf`/`doc`/`spreadsheet`: AMILab/계획서/제출 문서.

## Priority 4: Custom Skills를 repo별로 추가

전역 스킬은 많지만, Joon의 실제 반복 작업은 repo/context가 강하다. 아래는 repo-scoped skill로 만들 가치가 있다.

### behavior-lab-analysis

위치 후보: `/Users/joon/dev/behavior-lab/.agents/skills/behavior-lab-analysis/SKILL.md`

트리거:

- CALMS21, CEBRA, behavior-time-series, notebook, visualization, ethology analysis.

내용:

- 관련 README/노트북 우선순위.
- 결과 저장 경로.
- 그림/표 검증 기준.
- "데이터 없으면 가짜 결과 금지".

### behaviorsplatter-experiment

위치 후보: `/Users/joon/dev/BehaviorSplatter/.agents/skills/behaviorsplatter-experiment/SKILL.md`

트리거:

- Rat7M, M5f1, 4D deform, canonical midpoint, MultiSwings, Plucker OOD.

내용:

- 실험 command template.
- long-running/GPU approval 규칙.
- 결과 확인 checklist.
- 실패 로그 요약 형식.

### paper-writing-factcheck

전역 또는 repo별.

트리거:

- 계획서, 논문 문단, citation, related work, fact-check.

내용:

- 최신 사실은 browse.
- citation claim은 source line/URL로 검증.
- final output은 claim table + corrected prose.

### email-comm-explain

전역.

트리거:

- `/comm --explain`, 이메일 초안/해석/응답.

내용:

- 요청 사항 번호화.
- 제목/내용 불일치 탐지.
- 날짜는 절대 날짜로 명시.
- 한국어 톤: 간결/공손/실무형.

### experiment-launch-audit

전역.

트리거:

- "야간 launch", "batch", "GPU", "multi run", "sweep".

내용:

- dry-run.
- output dir collision check.
- resume/restart semantics.
- command logging.
- expected artifacts.

Skill skeleton:

```markdown
---
name: behavior-lab-analysis
description: Use for behavior-lab notebooks, CALMS21/CEBRA behavioral time-series analysis, visualization, and report generation. Trigger on behavior-lab, CALMS21, CEBRA, ethology, or behavior-time-series.
---

# Behavior Lab Analysis

1. Read repo README and the relevant notebook README first.
2. Identify input data paths before editing code.
3. Preserve existing notebook/report style.
4. Run the smallest executable check.
5. Save outputs under `reports/` or the notebook's established output folder.
6. Never invent quantitative results; mark missing data explicitly.
```

## Priority 5: MCP는 "현재 상태가 외부에 있는 문제"에만

MCP는 강력하지만, 무조건 많이 붙이면 context와 보안면이 넓어진다.

추천 연결:

- OpenAI Docs MCP: 이미 `openaiDeveloperDocs`를 등록했다. 다음 Codex 재시작부터 사용 가능.
- GitHub: PR, issue, review, CI, repo metadata.
- Figma: design-to-code, screenshot/asset/spec 검증.
- Notion: meeting/spec/research knowledge.
- Linear: issue/task sync.
- Sentry: production error triage.

비추천:

- 일반 웹 검색으로 충분한 단발성 문서.
- 인증 범위가 과도한 사내 시스템.
- 읽기만 필요한데 write 권한이 열리는 tool.

## Priority 6: Subagents는 "읽기/검토 병렬화"에 먼저 사용

Subagents는 명시적으로 요청해야 하며 비용이 늘어난다. 쓰기 작업에서는 파일 충돌이 생기기 쉽다.

좋은 용도:

- PR 리뷰: security / correctness / tests / maintainability 분리.
- 큰 코드베이스 온보딩: data flow / API / training pipeline / UI 분리.
- 논문/계획서 fact-check: claims / citations / structure 분리.
- migration 검토: schema / callers / tests / rollback 분리.

나쁜 용도:

- 같은 파일을 여러 agent에게 동시에 수정.
- 작은 버그 한 개.
- 외부 사실 확인 없이 상상으로 비교해야 하는 작업.

Prompt:

```text
Spawn one subagent per perspective: security, correctness, tests, maintainability.
Each subagent should read only what it needs and return findings with file/line references.
Do not edit files. Wait for all results, deduplicate, then rank by severity.
```

## Priority 7: Verification을 기본 성공 조건으로 둔다

Codex의 강점은 코드를 쓰는 것보다 "읽고, 수정하고, 명령을 돌리고, 다시 고치는 loop"에 있다.

항상 넣을 문장:

```text
Run the smallest relevant check. If the check cannot run, explain why and provide the strongest alternative verification.
```

중요 작업에는 추가:

```text
After implementing, switch to verifier mode. Criticize the diff as if it may be wrong. Check edge cases, tests, backward compatibility, and whether no change would have been better.
```

Action bias 방지:

```text
It is acceptable to make no code changes if the issue is already fixed, cannot be reproduced, or is caused by configuration/data. Treat "no change needed" as a valid successful outcome if evidence supports it.
```

## Use-Case Playbooks

### 1. 새 repo 이해

Best surface: IDE or CLI.

Prompt:

```text
Read @README.md and inspect the repo structure.
Explain:
1. Main entry points
2. Data/config flow
3. Test/build commands
4. Risky or fragile areas
5. What AGENTS.md should contain for future work
Do not edit files yet.
```

### 2. 버그 수정

Best surface: CLI local thread.

Prompt:

```text
Bug: <symptom>
Repro: <steps/command/log>
Suspected files: @...

First reproduce or trace the failure. If it is already fixed or not a code bug, stop and explain.
Otherwise make the smallest patch, add a regression test if feasible, and run the smallest relevant checks.
```

### 3. 실험 batch launch

Best surface: CLI, with approval for long jobs/GPU/network.

Prompt:

```text
Prepare a batch launch plan for <experiment>.
Before running:
- verify input paths
- check output directory collision
- show exact commands
- estimate runtime/resource risk
- identify resume behavior
Ask before launching long-running jobs.
```

### 4. 논문/계획서 fact-check

Best surface: CLI/App with web.

Prompt:

```text
Fact-check this section.
For every factual claim, classify:
- supported
- unsupported
- stale/needs update
- wrong
Use official/primary sources when possible.
Then rewrite only the unsupported/wrong claims.
```

### 5. UI 구현

Best surface: IDE/CLI + Playwright + Figma MCP if available.

Prompt:

```text
Implement this UI from @screenshot.png / Figma node <...>.
Match layout/spacing/typography.
After implementation, run the app, take desktop and mobile screenshots with Playwright, inspect console errors, and fix visible overlap or broken layout.
```

### 6. PR review

Best surface: CLI or GitHub app.

Prompt:

```text
Review this PR as a code reviewer.
Findings first, ordered by severity.
Check correctness, edge cases, security/privacy, migrations, tests.
Do not summarize until after findings.
If no issues, say so and list residual risk.
```

### 7. CI failure

Best surface: GitHub skill.

Prompt:

```text
Use github:gh-fix-ci.
Inspect failing GitHub Actions checks for this PR.
Summarize the exact failing job/log lines and propose a fix plan.
Do not implement until I approve.
```

## Famous / Public Workflow Lessons

### Karpathy-style vibe coding

Karpathy popularized "vibe coding" as a conversational, high-delegation style: describe what you want, run it, feed errors back, iterate quickly. This is powerful for throwaway prototypes and UI exploration. It is weak for long-lived research code unless you add tests, constraints, and review.

Joon recommendation: use vibe coding only for prototypes, demos, one-off dashboards, and visual exploration. For `BehaviorSplatter`, paper claims, GPU experiments, or repo infrastructure, use agentic coding with verification.

### Simon Willison-style distinction

Willison's useful line is the distinction between unreviewed vibe coding and AI-assisted engineering: if you review, test, and understand the code, it is not "just vibes"; it is an acceleration tool. This maps well to Joon's work because many artifacts are research claims or reproducibility-sensitive.

Joon recommendation: accept AI-generated code only after one of these holds:

- Tests pass.
- You inspected the diff.
- The result is disposable.
- A verifier pass found no blocker.

### Professional developer pattern

Empirical studies say experienced developers keep control over architecture and quality while delegating bounded tasks. That should be the default. Let Codex do search, boilerplate, refactors, tests, summaries, and first-pass fixes. Keep human control over problem framing, scientific interpretation, security, data provenance, and final claims.

## Anti-Patterns

- "Fix this" with no repro.
- "Make it better" with no definition of done.
- Letting Codex modify broad directories without a file scope.
- Running multiple threads that edit the same files.
- Treating memory as a rule system.
- Letting external tools have write scope when read scope is enough.
- Trusting generated citations, metrics, or experiment results without checking artifacts.
- Asking for huge rewrites when a small patch plus test is sufficient.
- Forgetting to say "no code change is acceptable".

## Joon-Specific Recommendations

### This week

1. Add repo `AGENTS.md` to `behavior-lab`, `BehaviorSplatter`, `poc/lpwm`, and `papers/cvpr2026`.
2. Create `behavior-lab-analysis` and `experiment-launch-audit` skills.
3. Enable or review Codex memories only after deciding whether external web/MCP-derived sessions should be excluded.
4. Restart Codex once so the newly registered `openaiDeveloperDocs` MCP becomes available.

### Default command vocabulary

Use these prefixes in prompts:

- "Plan only, no edits."
- "Review only, findings first, no edits."
- "Implement, then verify."
- "No code change is a valid outcome."
- "Use official/current sources."
- "Spawn subagents for independent review."
- "Update AGENTS.md if this correction should persist."

### High-value custom skill backlog

1. `behavior-lab-analysis`
2. `experiment-launch-audit`
3. `paper-writing-factcheck`
4. `email-comm-explain`
5. `behaviorsplatter-experiment`
6. `gpu-remote-runner`
7. `academic-cv-materials`

## Final Operating Rule

Codex output quality is bounded less by model intelligence than by task shape. The best results come from small, verifiable tasks with persistent project guidance and a clear definition of done. Use Codex aggressively for execution and exploration, but keep Joon's judgment in charge of scope, evidence, and final interpretation.

## Fact-Check Addendum: 2026-06-12

이 섹션은 위 문서 작성 후 공식 문서와 로컬 파일을 다시 대조한 정정/확인 내용이다.

### What Was Correct

- `AGENTS.md` 우선순위는 맞다. 공식 문서 기준으로 Codex는 작업 전 `AGENTS.md`를 읽고, global -> project root -> current directory 순서로 instruction chain을 만든다. 가까운 디렉터리 지침이 나중에 들어가므로 더 우선한다.
- skill 설명도 대체로 맞다. Codex skill은 `SKILL.md`가 필수이고, scripts/references/assets를 선택적으로 포함한다. Codex는 먼저 name/description/path만 보고, 선택된 skill의 전체 `SKILL.md`를 나중에 읽는 progressive disclosure 방식을 쓴다.
- MCP 설명도 맞다. MCP는 Codex에 third-party tools/context를 연결하는 계층이고, CLI와 IDE extension에서 지원된다.
- `openaiDeveloperDocs` MCP는 로컬 `/Users/joon/.codex/config.toml`에 등록되어 있다. 이번 세션의 tool surface에는 즉시 노출되지 않았으므로, 다음 Codex 재시작 후 사용하는 것이 맞다.

### Corrections / Nuance

- "Subagents는 읽기/검토 병렬화에 먼저 사용"은 보수적 추천이지 공식 제한은 아니다. 공식 문서 예시는 PR review용 read-only agents뿐 아니라 frontend/theme/testing처럼 disjoint write scope를 나눠 구현하는 custom agents도 제시한다.
- "repo-scoped custom skills 후보" 중 `behavior-lab-analysis`, `behaviorsplatter-experiment`, `paper-factcheck`는 2026-06-13에 구현되었다. `experiment-launch-audit`, `email-comm-explain`은 아직 전역 또는 repo-scoped skill로 만들지 않은 후보로 남아 있다.
- custom subagent는 skill과 다르다. skill은 `.agents/skills/<name>/SKILL.md` 또는 사용자 skill 디렉터리의 workflow 지식이고, custom agent/subagent는 `.codex/agents/<agent>.toml` + `.codex/config.toml`의 `[agents]` 설정으로 정의하는 별도 실행자다.
- GitHub 관련 skill은 전역 skill과 GitHub plugin skill이 중복으로 존재한다. plugin 쪽 `github:*` skill은 GitHub app connector를 우선 사용하므로, PR/issue/CI 작업에는 plugin skill을 우선하는 것이 더 정확하다.

### Actually Available Skills on This Machine

System/user skills currently installed under `/Users/joon/.codex/skills`:

- Core/meta: `openai-docs`, `skill-creator`, `skill-installer`, `plugin-creator`
- Planning/review: `deliberate-lite`, `orchestrate-plan-verify`, `review-modes`
- GitHub standalone: `gh-address-comments`, `gh-fix-ci`, `yeet`
- Documents/data: `pdf`, `doc`, `spreadsheet`, `jupyter-notebook`
- Browser/UI/design: `playwright`, `figma`, `figma-implement-design`, `screenshot`, `develop-web-game`
- Deploy: `vercel-deploy`, `netlify-deploy`, `cloudflare-deploy`, `render-deploy`
- Security: `security-best-practices`, `security-threat-model`, `security-ownership-map`
- Media/API utilities: `imagegen`, `sora`, `speech`, `transcribe`
- Ops/PM/knowledge: `linear`, `sentry`, `notion-knowledge-capture`, `notion-meeting-intelligence`, `notion-research-documentation`, `notion-spec-to-implementation`

GitHub plugin skills currently installed under `/Users/joon/.codex/plugins/cache/openai-curated-remote/github/0.1.2/skills`:

- `github`
- `gh-address-comments`
- `gh-fix-ci`
- `yeet`

Recommended invocation style:

```text
Use $review-modes deep + fact. Review only, findings first, no edits.
```

```text
Use $orchestrate-plan-verify. Implement the smallest patch, run checks, then switch to verifier mode.
```

```text
Use github:gh-fix-ci. Inspect failing GitHub Actions checks and propose a fix plan. Do not implement until approved.
```

### Codex Subagent Explanation for Joon

Codex subagents are separate agents spawned by the parent Codex session for bounded parallel work. Use them when a task naturally splits into independent questions or disjoint file ownership.

Good subagent jobs:

- `explorer`: read-only codebase mapping, dependency tracing, finding call sites.
- `reviewer`: correctness/security/test-risk review with file/line findings.
- `docs_researcher`: verifies official docs/API behavior through MCP or web.
- `worker`: implements one isolated slice with explicit file/module ownership.
- `verifier`: checks a completed diff while the main agent continues non-overlapping work.

Bad subagent jobs:

- Vague "look around and improve it".
- Multiple agents editing the same files.
- Blocking next-step work that the parent must wait for immediately.
- Anything requiring shared mutable state without clear integration ownership.

Prompt pattern:

```text
Use subagents for independent review only.
Spawn:
- explorer: trace affected code paths and cite files/symbols.
- reviewer: find correctness/security/test risks, no style-only comments.
- docs_researcher: verify external API assumptions with official docs.

No subagent should edit files. Parent agent should deduplicate findings and rank by severity.
```

For implementation:

```text
Use subagents only if write scopes are disjoint.
Worker A owns <module/path A>.
Worker B owns <module/path B>.
They must not modify each other's files or revert unrelated changes.
Parent agent integrates, reviews the combined diff, and runs final checks.
```

Joon-specific rule: default subagents to read-only review/exploration. Use worker subagents only for large mechanical changes where file ownership can be cleanly split.

## Implementation Status: 2026-06-13

Implemented repo guidance and scoped skills:

- `behavior-lab/AGENTS.md`
- `BehaviorSplatter/AGENTS.md`
- `poc/lpwm/AGENTS.md`
- `papers/cvpr2026/AGENTS.md`
- `behavior-lab/.agents/skills/behavior-lab-analysis/SKILL.md`
- `BehaviorSplatter/.agents/skills/behaviorsplatter-experiment/SKILL.md`
- `papers/cvpr2026/.agents/skills/paper-factcheck/SKILL.md`
- `CLAUDE.md` and `GEMINI.md` bridge files for the same repos so Claude/Gemini-style agents read `AGENTS.md` and the matching repo skill when relevant.

Fact-check note: official Codex skills docs confirm repository skills are discovered from `.agents/skills` folders from the current working directory up to the repository root. Official subagent docs confirm subagents spawn only when explicitly requested and support built-in `default`, `worker`, and `explorer` agents plus custom TOML agents.

## Approval Modes & 무중단(YOLO) 실행: 2026-07-02

목표: 매번 승인 프롬프트 없이 Codex를 최종 목표까지 돌리기. 두 축의 조합으로 제어.

- `approval_policy` = 얼마나 자주 승인 요청 (`untrusted` / `on-request` / `on-failure` / `never`)
- `sandbox_mode` = 파일·네트워크 접근 범위 (`read-only` / `workspace-write` / `danger-full-access`)
- 우선순위: CLI 플래그 > profile > `~/.codex/config.toml` > default

### 두 가지 방식

| | (A) config/profile 지속 | (B) per-run 플래그 |
|---|---|---|
| 명령 | `codex --profile yolo "..."` | `codex --yolo "..."` |
| 실체 | `approval_policy="never"` + `sandbox_mode="workspace-write"` | `--dangerously-bypass-approvals-and-sandbox` (승인·샌드박스 **모두** 해제, 시스템 전체 접근) |
| 마찰 | 낮음 (한 번 정의) | 매번 플래그 |
| blast radius | workspace 폴더 + tmp, 네트워크 기본 차단 | 무제한 |

`--full-auto`는 소스별로 "workspace-write 편의 플래그" vs "deprecated 경고" 로 엇갈림(🟡) → 명시적 config/`--yolo`로 회피.

### 권장 셋업 (MAGI 3-vote 결론: 글로벌 영구 금지, 명시 opt-in)

```toml
# ~/.codex/config.toml
approval_policy = "on-request"       # 글로벌 기본 = 평소 그대로 (안전)
sandbox_mode    = "workspace-write"

[profiles.yolo]                       # 무중단 원할 때만 명시 호출
approval_policy = "never"
sandbox_mode    = "workspace-write"   # 시스템 전체 필요 시만 danger-full-access
```
```bash
codex --profile yolo "최종 목표까지 알아서 진행"
```

shell alias (`~/dotfiles/shell/common.sh`, 동기화됨):
- `x` = `codex` — 기존 c/a/x SSOT, 대화형 TUI (yolo 아님). `bs && x` 대화형.
- `xy` = `codex exec --profile yolo` — **비대화형(headless)·완료 후 exit → 체이닝**. `bs && xy "..."`.
- `yl` = `codex --profile yolo` — yolo 대화형 TUI(지켜보며, 체인 X).

`approval_policy="never"`를 글로벌 최상위에 박으면 **미래의 모든 프로젝트(민감한 것 포함)에 조용히 적용**되므로 금지. profile 분리 = 마찰도 낮고 위험도 낮은 상위 호환.

### workspace-write = 사고 방지 O + 상당한 격리, 단 완전하진 않음

| 위협 | `--yolo`(full access) | `workspace-write` |
|---|:---:|:---:|
| repo 밖 write (`rm`, `~/.ssh`) | ❌ | ✅ 차단 |
| 네트워크 exfil / `git push` | ❌ | ✅ 기본 차단 (network off) |
| `.git`/`.codex`/`.agents` 수정 | ❌ | ✅ 차단 (writable root 내에서도 read-only 보호, recursive) |
| repo **안** `.env`/secret **읽기** | ❌ | ❌ 못 막음 |
| repo **밖** 파일 읽기 | ❌ | 🟡 대체로 허용 |

→ **쓰기·네트워크·git·앱 config는 기본 차단** → 사고 + 상당수 공격 방어됨. 남는 실질 구멍 = **workspace 안에 있는 secret 읽기**(+ `[sandbox_workspace_write] network_access=true`로 켜면 exfil 가능). 완전 격리는 컨테이너/VM(`--yolo` 전용 무대)만.

전제조건: secret은 workspace 밖(`~/.config/secrets/` — 이미 충족)에 두면 위 "읽기" 구멍도 닫힘. `network_access`는 기본 false 유지.

근거: workspace-write는 network 기본 disabled, write는 workspace+tmp로 제한, `.git`/`.codex`/`.agents`는 recursive read-only 보호. 출처: [Sandboxing](https://developers.openai.com/codex/concepts/sandboxing), [Config reference](https://developers.openai.com/codex/config-reference), [Agent approvals & security](https://developers.openai.com/codex/agent-approvals-security). /fact 2-model + web 재검증(2026-07-02).
