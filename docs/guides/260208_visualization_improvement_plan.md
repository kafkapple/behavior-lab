# Plan: 시각화 모듈 개선 — Multi-Person Color + HTML Report

Created: 2026-02-08

## Context

E2E 검증으로 `animate_skeleton()` GIF 생성은 동작 확인했으나, 시각적 품질이 기초적.
- **문제 1**: 모든 skeleton에 `joint_colors`/`limb_colors`가 `None` → 단일 색상 렌더링
- **문제 2**: CalMS21 (2 mice), NTU (2 persons) 등 multi-person 구분 불가
- **문제 3**: HTML 리포트 없음 (MD/JSON만) — behavior-tools에는 `HTMLReporter` 패턴 존재

**목표**: body_parts 기반 색상 사전 + multi-person 색 구분 + self-contained HTML 리포트

---

## Step 1: Body-Part 색상 사전 추가

**파일**: `src/behavior_lab/visualization/colors.py` (새 파일)

body_parts 의미에 따른 범용 색상 팔레트. skeleton마다 동일한 의미 부위는 같은 색 계열.

### 1a. 색상 팔레트 정의

```python
# Semantic body-part color palette
BODY_PART_COLORS = {
    # Head/Face
    "head": "#E74C3C",       # Red
    "face": "#E74C3C",
    # Torso/Spine
    "torso": "#3498DB",      # Blue
    "spine": "#3498DB",
    "body": "#3498DB",
    "neck": "#2980B9",
    # Arms
    "left_arm": "#2ECC71",   # Green
    "right_arm": "#F39C12",  # Orange
    # Legs
    "left_leg": "#9B59B6",   # Purple
    "right_leg": "#E67E22",  # Dark orange
    # Animal-specific
    "tail": "#95A5A6",       # Gray
    "left_side": "#2ECC71",
    "right_side": "#F39C12",
    # Limbs (generic quadruped)
    "front_left": "#2ECC71",
    "front_right": "#F39C12",
    "hind_left": "#9B59B6",
    "hind_right": "#E67E22",
}

# Multi-person palette (maximally distinct)
PERSON_COLORS = [
    "#1ABC9C",  # Person 0: Teal
    "#E74C3C",  # Person 1: Red
    "#3498DB",  # Person 2: Blue
    "#F39C12",  # Person 3: Orange
]
```

### 1b. 색상 해석 함수

```python
def get_joint_colors(skeleton) -> list[str]:
    """body_parts → 각 joint에 색상 할당. 모든 skeleton 범용."""

def get_limb_colors(skeleton) -> list[str]:
    """각 edge에 소속 body_part 색상 할당."""

def get_person_colors(n_persons: int) -> list[str]:
    """Multi-person 구분 색상."""

def get_joint_labels(skeleton) -> list[str]:
    """joint_names 기반 표시용 약어 (선택적 overlay용)."""
```

**핵심 로직**: `skeleton.body_parts` 순회 → joint index → 색상 매핑.
등록되지 않은 body_part는 fallback 회색.

---

## Step 2: 렌더러 개선

**파일**: `src/behavior_lab/visualization/skeleton.py` (기존 수정)

### 2a. `plot_skeleton()` 개선

현재: 모든 joint 파란색, 모든 limb 회색
개선:
- `joint_colors` 매개변수 추가 (기본값: `colors.get_joint_colors(skeleton)`)
- `limb_colors` 매개변수 추가
- `show_labels` 매개변수: joint 이름/번호 텍스트 overlay
- `person_idx` 매개변수: multi-person 시 어떤 person인지 (색상 오프셋)

**Multi-person 렌더링 로직** (CalMS21 14 joints = 7×2 mice):
```python
if skeleton.num_persons > 1:
    joints_per_person = skeleton.num_joints  # 7
    for p in range(num_persons):
        start = p * joints_per_person
        end = start + joints_per_person
        # person p의 keypoints[start:end]를 PERSON_COLORS[p] 계열로 렌더링
```

### 2b. `animate_skeleton()` 개선

- 동일한 color/label 매개변수 전달
- multi-person 구분 유지
- 프레임 카운터 + 라벨 텍스트 (선택) 오버레이

### 2c. 새 함수: `plot_skeleton_comparison()`

```python
def plot_skeleton_comparison(
    keypoints_list: list[np.ndarray],
    titles: list[str],
    skeleton=None,
    frame: int = 0,
    save_path=None,
) -> tuple:
    """Side-by-side skeleton 비교 (raw vs preprocessed 등)."""
```

---

## Step 3: HTML 리포트 생성기

**파일**: `src/behavior_lab/visualization/html_report.py` (새 파일)

behavior-tools의 `HTMLReporter` 패턴을 활용하되, behavior-lab 파이프라인에 맞게 재설계.

### 3a. 핵심 유틸리티

```python
def fig_to_base64(fig) -> str:
    """matplotlib Figure → base64 data URI (PNG)."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

def image_to_base64(path: str | Path, max_size: int = 600) -> str:
    """이미지 파일 → base64 data URI."""
```

### 3b. `generate_pipeline_report()` 함수

```python
def generate_pipeline_report(
    report_data: dict,
    output_path: str | Path,
    title: str = "Behavior Analysis Report",
) -> Path:
```

**HTML 구조** (self-contained, 외부 의존성 없음):

```
┌─────────────────────────────────────┐
│ Header (그라디언트 배너)             │
│  제목 + 날짜 + 요약 통계            │
├─────────────────────────────────────┤
│ Tab Navigation                      │
│ [Overview] [CalMS21] [NTU] [UCLA]   │
├─────────────────────────────────────┤
│ Overview Tab                        │
│  ├ Dataset summary table            │
│  └ Key metrics cards                │
├─────────────────────────────────────┤
│ Dataset Tab (per dataset)           │
│  ├ Data summary (shape, labels)     │
│  ├ Skeleton sample (static + GIF)   │
│  ├ Preprocessing comparison         │
│  ├ Embedding plot                   │
│  ├ Transition matrix                │
│  ├ Ethogram                         │
│  └ Metrics table                    │
└─────────────────────────────────────┘
```

**CSS**: 반응형 그리드, 호버 효과, 탭 네비게이션 (JS 포함)
**이미지**: 모두 base64 임베딩 → 단일 HTML 파일로 완결

### 3c. report_data 구조

기존 `test_e2e.py`의 `report` dict를 확장:
```python
report_data = {
    "title": str,
    "timestamp": str,
    "datasets": {
        "calms21": {
            "data": {...},
            "figures": {          # NEW: matplotlib Figure 객체 or base64
                "skeleton_static": fig,
                "skeleton_gif": "path.gif",  # GIF는 경로로
                "preprocessing": fig,
                "embedding": fig,
                "transition": fig,
                "ethogram": fig,
            },
            "metrics": {...},
        },
        ...
    }
}
```

---

## Step 4: test_e2e.py 업데이트

**파일**: `scripts/test_e2e.py` (기존 수정)

### 변경사항
- 각 데이터셋에 skeleton sample GIF 생성 추가
- color-aware `plot_skeleton()` 사용
- HTML 리포트 생성 호출 추가
- `plt.close()` 전에 figure를 report_data에 저장

### 출력 추가

```
outputs/e2e_test/
├── report.html          # NEW: self-contained HTML
├── report.json
├── report.md
├── calms21/
│   ├── ... (기존)
│   ├── sample_skeleton.png    # NEW: colored
│   └── sample_animation.gif   # NEW: colored
├── ntu/
│   └── sample_skeleton.png    # NEW
└── nwucla/
    ├── sample_skeleton.png    # NEW
    └── sample_animation.gif   # NEW
```

---

## Step 5: `__init__.py` 업데이트

**파일**: `src/behavior_lab/visualization/__init__.py`

```python
from .colors import get_joint_colors, get_limb_colors, get_person_colors, BODY_PART_COLORS
from .html_report import generate_pipeline_report, fig_to_base64
```

---

## 수정 파일 요약

| 파일 | 작업 | 신규/수정 |
|------|------|-----------|
| `visualization/colors.py` | Body-part 색상 사전 + 해석 함수 | **신규** |
| `visualization/skeleton.py` | Multi-person color, labels, comparison | 수정 |
| `visualization/html_report.py` | Self-contained HTML 리포트 생성기 | **신규** |
| `visualization/__init__.py` | 새 모듈 export | 수정 |
| `scripts/test_e2e.py` | Colored skeleton + GIF + HTML 리포트 | 수정 |

---

## 검증 기준

| 항목 | 기준 |
|------|------|
| CalMS21 skeleton | 2마리 마우스가 **다른 색**으로 표시 |
| NTU skeleton | body_parts(torso/arm/leg)별 색 구분 |
| NW-UCLA skeleton | 20 joints 색상 렌더링 |
| Joint labels | `show_labels=True` 시 관절명 표시 |
| GIF | 3개 데이터셋 모두 colored GIF 생성 |
| HTML report | `report.html` 단일 파일, 브라우저에서 열림 |
| HTML 내용 | 모든 PNG/GIF 이미지 base64 임베딩 |
| 탭 네비게이션 | 데이터셋별 탭 전환 동작 |
