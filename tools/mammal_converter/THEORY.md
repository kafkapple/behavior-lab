# 변환 이론 및 수학적 배경

## 1. 스켈레톤 기반 메시 변형 (Skeletal Animation)

### 1.1 기본 개념

스켈레톤 애니메이션은 3D 메시를 뼈대(skeleton)를 사용하여 변형하는 기법입니다.

```
        [Root]
           │
       [Spine] ─── [Shoulder] ─── [Elbow] ─── [Wrist]
           │
       [Hip] ─── [Knee] ─── [Ankle]
```

### 1.2 핵심 구성 요소

| 구성 요소 | 설명 | 예시 |
|----------|------|------|
| **Joint (관절)** | 스켈레톤의 노드 | 어깨, 팔꿈치, 손목 |
| **Bone (뼈)** | 두 관절을 연결 | 상완, 전완 |
| **Skinning Weight** | 각 정점에 대한 관절 영향력 | vertex[0]은 shoulder에 70%, elbow에 30% |
| **Bind Pose (T-Pose)** | 기준이 되는 초기 포즈 | 팔을 벌린 자세 |

### 1.3 Linear Blend Skinning (LBS)

정점 v의 최종 위치 v' 계산:

```
v' = Σ(w_i × M_i × v)

where:
  w_i = joint i의 skinning weight (Σw_i = 1)
  M_i = joint i의 skinning matrix
  v   = bind pose에서의 정점 위치
```

**Skinning Matrix 계산:**
```
M_i = G_i × B_i^(-1)

where:
  G_i = joint i의 global transform (world space)
  B_i = joint i의 bind matrix (bind pose에서의 global transform)
  B_i^(-1) = inverse bind matrix
```

---

## 2. 회전 표현 방식

### 2.1 Axis-Angle (MAMMAL_mouse 사용)

3D 벡터로 회전축과 각도를 표현:

```
r = θ × n̂

where:
  θ = 회전 각도 (radians)
  n̂ = 회전축 단위 벡터 (||n̂|| = 1)
  r = axis-angle 벡터 (||r|| = θ)
```

**특징:**
- 3개 값으로 회전 표현 (compact)
- Gimbal lock 없음
- Rodrigues 공식으로 회전 행렬 변환

### 2.2 Quaternion (DualPM 사용)

4D 벡터로 회전 표현:

```
q = (w, x, y, z) = (cos(θ/2), sin(θ/2)×n̂)

where:
  θ = 회전 각도
  n̂ = 회전축 단위 벡터
  ||q|| = 1 (unit quaternion)
```

**특징:**
- 4개 값으로 회전 표현
- Gimbal lock 없음
- 보간(interpolation)에 유리
- 정규화 필요 (||q|| = 1)

### 2.3 변환: Axis-Angle → Quaternion

```python
def axis_angle_to_quaternion(axis_angle):
    """
    axis_angle: (3,) - axis-angle vector
    returns: (4,) - quaternion [x, y, z, w] (scipy convention)
    """
    theta = np.linalg.norm(axis_angle)

    if theta < 1e-8:
        # 회전 없음 → identity quaternion
        return np.array([0, 0, 0, 1])

    axis = axis_angle / theta
    half_theta = theta / 2

    w = np.cos(half_theta)
    xyz = np.sin(half_theta) * axis

    return np.array([xyz[0], xyz[1], xyz[2], w])
```

**주의: Quaternion 순서 컨벤션**
- SciPy: `[x, y, z, w]`
- PyTorch3D: `[w, x, y, z]`
- GLTF: `[x, y, z, w]`
- DualPM: `[w, x, y, z]` (PyTorch3D 기준)

---

## 3. Forward Kinematics (FK)

### 3.1 개념

부모 관절에서 자식 관절로 변환을 전파하여 각 관절의 world space 위치/회전 계산.

```
        [Root] ──Local T──▶ [Child1] ──Local T──▶ [Child2]
           │
     Global T = I    Global T = T_root    Global T = T_root × T_child1
```

### 3.2 알고리즘

```python
def forward_kinematics(local_transforms, parents):
    """
    local_transforms: (num_joints, 4, 4) - 각 관절의 로컬 변환
    parents: (num_joints,) - 부모 관절 인덱스 (-1 = 루트)

    returns: (num_joints, 4, 4) - 글로벌 변환
    """
    global_transforms = np.zeros_like(local_transforms)

    for j in range(num_joints):
        if parents[j] == -1:  # 루트
            global_transforms[j] = local_transforms[j]
        else:
            parent_idx = parents[j]
            global_transforms[j] = global_transforms[parent_idx] @ local_transforms[j]

    return global_transforms
```

### 3.3 관절 위치 추출

```python
# Global transform의 translation 성분 추출
joint_positions = global_transforms[:, :3, 3]  # (num_joints, 3)
```

---

## 4. GLTF 2.0 스켈레톤 표현

### 4.1 GLTF 구조

```json
{
    "nodes": [
        {"name": "Root", "children": [1], "translation": [0,0,0]},
        {"name": "Spine", "children": [2,3], "rotation": [x,y,z,w]},
        ...
    ],
    "skins": [{
        "joints": [0, 1, 2, ...],           // 조인트 노드 인덱스
        "inverseBindMatrices": 0            // accessor 인덱스
    }],
    "meshes": [{
        "primitives": [{
            "attributes": {
                "POSITION": 0,
                "JOINTS_0": 1,              // 각 정점에 영향을 주는 조인트 인덱스
                "WEIGHTS_0": 2              // 각 조인트의 가중치
            }
        }]
    }]
}
```

### 4.2 Inverse Bind Matrix

각 관절에 대해, bind pose에서의 글로벌 변환의 역행렬:

```
inverseBindMatrix[j] = (globalTransform_bindPose[j])^(-1)
```

**용도:** 정점을 world space에서 joint space로 변환

### 4.3 GLTF 컨벤션

- **좌표계:** Y-up, Right-handed
- **단위:** 미터 (m)
- **회전:** Quaternion [x, y, z, w]
- **행렬:** Column-major (OpenGL 스타일)

---

## 5. MAMMAL_mouse → DualPM 변환

### 5.1 전체 파이프라인

```
┌─────────────────────────────────────────────────────────────────┐
│                        MAMMAL_mouse                              │
├─────────────────────────────────────────────────────────────────┤
│  mouse.pkl                                                       │
│  ├── v_template: (V, 3) - T-pose 정점                            │
│  ├── faces: (F, 3) - 삼각형 인덱스                               │
│  ├── skinning_weights: (V, J) - 스킨 가중치                       │
│  ├── t_pose_joints: (J, 3) - T-pose 관절 위치                     │
│  └── parent_indices: (J,) - 부모 관절 인덱스                      │
│                                                                  │
│  param{N}.pkl                                                    │
│  ├── thetas: (B, J, 3) - axis-angle 로컬 회전                     │
│  ├── trans: (B, 3) - 글로벌 위치                                  │
│  ├── rotation: (B, 3) - 글로벌 회전 (euler)                       │
│  └── scale: (B, 1) - 스케일                                       │
└─────────────────────────────────────────────────────────────────┘
                                ↓
                         [변환 프로세스]
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                          DualPM                                  │
├─────────────────────────────────────────────────────────────────┤
│  mouse_shape.gltf                                                │
│  ├── mesh.vertices: (V, 3)                                       │
│  ├── mesh.faces: (F, 3)                                          │
│  ├── skin.joints: [0, 1, ..., J-1]                               │
│  ├── skin.inverseBindMatrices: (J, 4, 4)                         │
│  └── mesh.WEIGHTS_0, mesh.JOINTS_0                               │
│                                                                  │
│  {frame}_pose.npz                                                │
│  └── poses: (J, 7) = [quaternion(4), position(3)]                │
│       - quaternion: [w, x, y, z] (DualPM convention)             │
│       - position: 글로벌 관절 위치                                │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 포즈 변환 상세

```python
def convert_pose(mammal_params, t_pose_joints, parents):
    """
    MAMMAL param → DualPM pose
    """
    # 1. Axis-angle → Local rotation matrix
    local_rotations = []
    for j in range(num_joints):
        axis_angle = mammal_params['thetas'][0, j]  # (3,)
        R = Rotation.from_rotvec(axis_angle).as_matrix()  # (3, 3)
        local_rotations.append(R)

    # 2. Build local transforms
    local_transforms = np.zeros((num_joints, 4, 4))
    for j in range(num_joints):
        local_transforms[j, :3, :3] = local_rotations[j]
        if parents[j] == -1:
            local_transforms[j, :3, 3] = t_pose_joints[j]
        else:
            local_transforms[j, :3, 3] = t_pose_joints[j] - t_pose_joints[parents[j]]
        local_transforms[j, 3, 3] = 1.0

    # 3. Forward Kinematics → Global transforms
    global_transforms = forward_kinematics(local_transforms, parents)

    # 4. Apply global transformation
    global_rotation = Rotation.from_euler('ZYX', mammal_params['rotation'][0]).as_matrix()
    global_translation = mammal_params['trans'][0]
    scale = mammal_params['scale'][0]

    for j in range(num_joints):
        global_transforms[j, :3, :3] = global_rotation @ global_transforms[j, :3, :3]
        global_transforms[j, :3, 3] = global_rotation @ (scale * global_transforms[j, :3, 3]) + global_translation

    # 5. Extract quaternion and position
    poses = np.zeros((num_joints, 7))
    for j in range(num_joints):
        quat = Rotation.from_matrix(global_transforms[j, :3, :3]).as_quat()  # [x,y,z,w]
        quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])  # [w,x,y,z]
        position = global_transforms[j, :3, 3]

        poses[j, :4] = quat_wxyz
        poses[j, 4:] = position

    return poses  # (num_joints, 7)
```

---

## 6. 좌표계 변환

### 6.1 MAMMAL_mouse 좌표계

- **단위:** 밀리미터 (mm)
- **Y축:** 위쪽
- **스케일:** 기본값 115

### 6.2 DualPM 좌표계

- **단위:** 미터 (m) 또는 정규화
- **좌표계:** 카메라 공간 기준

### 6.3 변환

```python
# mm → m 변환
position_m = position_mm / 1000.0

# 또는 정규화 (bbox 기준)
position_normalized = (position - center) / extent
```

---

## 7. 검증 방법

### 7.1 시각적 검증

1. GLTF 뷰어에서 메시 렌더링
2. 스켈레톤 시각화
3. 포즈 적용 후 비교

### 7.2 수치적 검증

```python
# 1. Skinning 결과 비교
mammal_posed_mesh = mammal_model.forward(params)
dualpm_posed_mesh = apply_skinning(template, pose)
assert np.allclose(mammal_posed_mesh, dualpm_posed_mesh, atol=1e-3)

# 2. 관절 위치 비교
mammal_joints = mammal_model.get_joints()
dualpm_joints = pose[:, 4:]
assert np.allclose(mammal_joints, dualpm_joints, atol=1e-3)
```

---

## 8. 참고 자료

1. [GLTF 2.0 Specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html)
2. [GLTF Tutorial - Skins](https://github.khronos.org/glTF-Tutorials/gltfTutorial/gltfTutorial_020_Skins.html)
3. [Rodrigues' rotation formula](https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula)
4. [Quaternion and spatial rotation](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation)
5. [Linear Blend Skinning](https://www.cs.cmu.edu/~barbic/skinning.html)
