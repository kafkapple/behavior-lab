# MAMMAL to DualPM Converter

MAMMAL_mouse 프로젝트의 출력을 DualPM 학습 데이터셋으로 변환하는 독립 모듈입니다.

## 개요

```
MAMMAL_mouse 출력          이 모듈           DualPM 학습 데이터
┌──────────────────┐      ┌──────┐      ┌──────────────────┐
│ mesh_*.obj       │ ──▶  │      │ ──▶  │ shapes/          │
│ param*.pkl       │      │ 변환  │      │   mouse_shape.gltf│
│ mouse.pkl        │      │      │      │ poses/           │
│ (템플릿+스켈레톤) │      └──────┘      │   *_pose.npz     │
└──────────────────┘                     └──────────────────┘
```

## 빠른 시작

```bash
# 1. 환경 설정
cd /path/to/DualPM_Paper/tools/mammal_converter
pip install -r requirements.txt

# 2. 변환 실행
python convert.py \
    --mammal_dir /home/joon/dev/MAMMAL_mouse \
    --fitting_result markerless_mouse_1_nerf_20251126_234835 \
    --output_dir ./output/dualpm_mouse_dataset
```

## 입력 데이터 구조

### MAMMAL_mouse 필수 파일

```
MAMMAL_mouse/
├── mouse_model/
│   ├── mouse.pkl                    # 템플릿 모델 (정점, 스켈레톤, 스킨웨이트)
│   ├── keypoint22_mapper.json       # 키포인트-조인트 매핑
│   └── mouse_txt/
│       ├── parents.pkl              # 스켈레톤 계층 구조
│       ├── id_to_names.pkl          # 조인트 이름
│       └── ...
│
├── results/fitting/{fitting_name}/
│   ├── obj/
│   │   ├── mesh_000000.obj          # 프레임별 변형된 메시
│   │   └── ...
│   └── params/
│       ├── param0.pkl               # 프레임별 포즈 파라미터
│       └── ...
│
└── data/examples/{dataset_name}/    # 원본 이미지 데이터
    └── ...
```

## 출력 데이터 구조

```
output_dir/
├── shapes/
│   └── mouse/
│       └── mouse_shape.gltf         # 리깅된 템플릿 메시 (GLTF 2.0)
│
├── poses/
│   ├── 000000_pose.npz              # 포즈 파라미터
│   ├── 000001_pose.npz              # shape: (num_joints, 7)
│   └── ...                          # [quaternion(4), position(3)]
│
├── masks/
│   ├── 000000_mask.png              # 세그멘테이션 마스크 (160x160)
│   └── ...
│
├── renders/
│   ├── 000000_rgb.png               # RGB 렌더링 (160x160)
│   └── ...
│
├── features/                        # (옵션) DINOv2 특징
│   └── ...
│
├── cameras/
│   ├── 000000_camera.txt            # 카메라 파라미터
│   └── ...
│
├── metadata/
│   ├── 000000_metadata.txt          # 메타데이터
│   └── ...
│
└── train_benchmark.txt              # 학습/검증 분할
```

## 상세 사용법

### 기본 변환

```bash
python convert.py \
    --mammal_dir /home/joon/dev/MAMMAL_mouse \
    --fitting_result markerless_mouse_1_nerf_20251126_234835 \
    --output_dir ./output/dualpm_mouse
```

### 고급 옵션

```bash
python convert.py \
    --mammal_dir /home/joon/dev/MAMMAL_mouse \
    --fitting_result markerless_mouse_1_nerf_20251126_234835 \
    --output_dir ./output/dualpm_mouse \
    --resolution 160 \
    --train_split 0.9 \
    --with_features \
    --feature_model sd_dino
```

### 옵션 설명

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--mammal_dir` | MAMMAL_mouse 프로젝트 경로 | 필수 |
| `--fitting_result` | 피팅 결과 폴더명 | 필수 |
| `--output_dir` | 출력 경로 | 필수 |
| `--resolution` | 출력 이미지 해상도 | 160 |
| `--train_split` | 학습 데이터 비율 | 0.9 |
| `--with_features` | DINOv2 특징 추출 | False |
| `--camera_view` | 사용할 카메라 뷰 (0-5) | 0 |

## 변환 세부 사항

### 1. 메시 변환 (OBJ → GLTF)

MAMMAL_mouse의 템플릿 메시를 스켈레톤 정보와 함께 GLTF로 변환합니다.

```python
# 포함 정보
- vertices: T-pose 정점 좌표
- faces: 삼각형 인덱스
- skeleton: 조인트 계층 구조
- skinning_weights: 버텍스-조인트 가중치
- inverse_bind_matrices: 바인드 매트릭스
```

### 2. 포즈 변환 (axis-angle → quaternion)

```python
# MAMMAL_mouse (입력)
thetas: (batch, num_joints, 3)  # axis-angle
trans: (batch, 3)               # global position
rotation: (batch, 3)            # global rotation (euler)

# DualPM (출력)
poses: (num_joints, 7)          # [quaternion(4), position(3)]
```

### 3. 카메라 파라미터

6개 뷰 중 선택한 카메라의 내부/외부 파라미터를 추출합니다.

## 워크플로우 예시

### 전체 파이프라인

```bash
# Step 1: MAMMAL_mouse에서 메시 피팅 실행
cd /home/joon/dev/MAMMAL_mouse
./run_mesh_fitting_default.sh 0 100

# Step 2: 변환
cd /home/joon/dev/DualPM_Paper/tools/mammal_converter
python convert.py \
    --mammal_dir /home/joon/dev/MAMMAL_mouse \
    --fitting_result markerless_mouse_1_nerf_20251126_234835 \
    --output_dir /home/joon/data/dualpm_mouse_train

# Step 3: (옵션) DINOv2 특징 추출
python extract_features.py \
    --input_dir /home/joon/data/dualpm_mouse_train/renders \
    --output_dir /home/joon/data/dualpm_mouse_train/features

# Step 4: DualPM 학습
cd /home/joon/dev/DualPM_Paper
python scripts/train.py \
    dataset_root=/home/joon/data/dualpm_mouse_train \
    train_config.save_path=./weights/mouse
```

## 문제 해결

### Q: "KeyError: 'poses'" 오류
A: MAMMAL_mouse param 파일 형식이 다를 수 있습니다. `--param_format legacy` 옵션을 사용하세요.

### Q: GLTF 메시가 렌더링되지 않음
A: GLTF 뷰어(예: https://gltf-viewer.donmccurdy.com/)에서 확인하세요.

### Q: 학습이 발산함
A: 카메라 파라미터와 메시 스케일이 일치하는지 확인하세요.

## 라이선스

이 모듈은 DualPM_Paper 프로젝트의 일부입니다.
BSD-3-Clause License.

## 참고 문서

- [THEORY.md](./THEORY.md): 변환 이론 및 수학적 배경
- [DualPM Paper](https://arxiv.org/abs/2412.04464)
- [MAMMAL_mouse](https://github.com/anl13/MAMMAL_core)
