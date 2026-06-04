# Triangulation Methodology — kp_benchmark v0.1.3

> 3D keypoint pipeline의 algorithm theory, formulas, code 정리.
> 이전 산재된 정보를 단일 SSOT로 통합.

## Pipeline 개관

```
Raw video (6 cams, 18000 fr, 1152×1024)
        │
        ▼  Stage 1
DLC ResNet50 analyze_videos per cam
        │  K_orig, R, t from label3d_dannce.mat
        ▼
6 × h5 file: (T, 22 kp, x/y/likelihood) — 2D distorted? No: undistorted
        │
        ▼  Stage 2
Custom Linear DLT (this doc, §1)
        │
        ▼
3D world coords (T, 22, 3)
        │
        ▼  Stage 3
Temporal smoothing (§2): Savitzky-Golay window=15 OR Kalman+RTS
        │
        ▼
Final 3D KP dataset
```

## §1. Direct Linear Transform (DLT)

### 1.1 Theory

3D world point **X** = (X, Y, Z, 1) (homogeneous). Camera *i* projection:
$$\mathbf{x}_i = P_i \mathbf{X}, \quad P_i = K_i [R_i | t_i] \in \mathbb{R}^{3 \times 4}$$

여기서 (x_i, y_i, w_i) = P_i X. Pixel = (x_i/w_i, y_i/w_i).

Cross-product 형태로 (homogeneous equation):
$$\mathbf{x}_i \times P_i \mathbf{X} = \mathbf{0}$$

3개 방정식 중 2개만 독립:
$$x_i \cdot P_i^{(3)} \mathbf{X} - P_i^{(1)} \mathbf{X} = 0$$
$$y_i \cdot P_i^{(3)} \mathbf{X} - P_i^{(2)} \mathbf{X} = 0$$

P_i^(j)는 j번째 row (1-indexed).

### 1.2 6-view stacking

6 cam × 2 eq = 12 equations on 4 unknowns. Over-determined.

Matrix form: $A \mathbf{X} = \mathbf{0}$, $A \in \mathbb{R}^{12 \times 4}$.

### 1.3 SVD solution

$$A = U \Sigma V^T$$
$$\mathbf{X}^* = V^T_{\text{last row}}$$

$\Sigma$의 smallest singular value에 해당하는 right singular vector.

Cartesian 변환:
$$(X, Y, Z) = \mathbf{X}^*[0:3] / \mathbf{X}^*[3]$$

### 1.4 Pseudo-code

```python
def triangulate(pts_2d, likelihoods, Ps, prob_min=0.05):
    """Linear DLT triangulation.

    pts_2d: (6, 2) per-cam 2D pixels
    likelihoods: (6,) per-cam confidence
    Ps: list of 6 (3, 4) projection matrices
    """
    valid = [i for i in range(6)
             if likelihoods[i] >= prob_min
             and not np.isnan(pts_2d[i]).any()]
    if len(valid) < 2:
        return np.full(3, np.nan)  # 최소 2 view

    A = []
    for i in valid:
        x, y = pts_2d[i]
        A.append(x * Ps[i][2] - Ps[i][0])
        A.append(y * Ps[i][2] - Ps[i][1])
    A = np.stack(A)              # (2n_valid, 4)
    _, _, Vt = np.linalg.svd(A)
    X_homo = Vt[-1]
    return X_homo[:3] / X_homo[3] if abs(X_homo[3]) > 1e-9 else np.full(3, np.nan)
```

### 1.5 Hyperparameters (v0.1.3)

| | Value | 이유 |
|---|---|---|
| `prob_min` | **0.05** | sweep으로 0.10 vs 0.05 비교, OOD 0.27 mm 개선 |
| Min views | 2 | DLT 최소 요구사항 (4 equations on 4 unknowns) |
| K | K_orig (label3d) | K_new(α=0)과 통계적 tied — upstream도 K_orig 사용 추정 |
| skew K[0,1] | 포함 | cam1/4/6의 1-3 px overlay 정확도 향상 |
| Distortion | OFF | videos_undist 이미 undistorted (double-correction 회피) |

### 1.6 Reference

- Hartley & Zisserman, *Multiple View Geometry*, Ch. 12 (Triangulation)
- aniposelib `CameraGroup.triangulate()` (수학적 동등성 확인 ✓)

---

## §2. Temporal Smoothing

### 2.1 Savitzky-Golay (현재 default)

#### 2.1.1 핵심 아이디어
Sliding window 안의 데이터에 **다항식 fit**, window 중심점을 그 다항식 값으로 대체.
이동 평균(moving average)과 달리 **peak/local feature를 보존**.

#### 2.1.2 알고리즘
1. Window 크기 N = 2m+1 (우리는 15 = 2·7+1)
2. 각 중심 인덱스 k에 대해, 주변 [-m, m] 구간 데이터에 차수 p 다항식 fit (우리는 p=3):
   $$f(x) = a_0 + a_1 x + a_2 x^2 + a_3 x^3$$
3. Least squares: $\min \sum_{i=-m}^{m} (y_{k+i} - f(i))^2$
4. Smoothed 값 = $a_0$ (다항식 중심에서의 값 = constant term)
5. **수학적으로 동등**: pre-computed FIR convolution
   $$\hat{y}_k = \sum_{i=-m}^{m} c_i \cdot y_{k+i}$$
   coefficients $c_i$는 window/order로 결정되는 고정 vector.

#### 2.1.3 구현
```python
from scipy.signal import savgol_filter
smoothed = savgol_filter(series, window_length=15, polyorder=3)
# Internally: convolve with pre-computed Savitzky-Golay coefficients
```

NaN 처리: linear interp → filter → 원래 NaN 위치 복원.

#### 2.1.4 장단점
| 장점 | 단점 |
|---|---|
| ✅ 단순 (scipy 1-line) | ❌ noise model 없음 (uniform smoothing) |
| ✅ Peak/feature 보존 | ❌ Edge effects (window 끝부분) |
| ✅ 결정론적 (state 없음) | ❌ window 크기 선택 ad-hoc |
| ✅ O(N) 연산 | ❌ per-frame uncertainty quantification 없음 |

### 2.2 Kalman Filter + RTS Smoother (이번 추가)

#### 2.2.1 핵심 아이디어
**State-space framework**: 시스템이 hidden state를 가지고 dynamics에 따라 evolve.
관측은 noisy. Bayesian recursive estimation으로 state 추정.
**Forward (Kalman)** = causal (past only).
**Backward (RTS)** = offline smoothing (past + future) = 더 정확.

#### 2.2.2 모델

State: $\mathbf{s}_k = [p_k, v_k]^T$ (position, velocity).

Transition (constant velocity):
$$\mathbf{s}_k = F \mathbf{s}_{k-1} + \mathbf{w}_k, \quad F = \begin{bmatrix} 1 & dt \\ 0 & 1 \end{bmatrix}$$

Process noise $\mathbf{w}_k \sim \mathcal{N}(0, Q)$:
$$Q = q \begin{bmatrix} dt^4/4 & dt^3/2 \\ dt^3/2 & dt^2 \end{bmatrix}$$

Observation: $z_k = H \mathbf{s}_k + v_k$, $H = [1, 0]$, $v_k \sim \mathcal{N}(0, R)$.

**Forward Kalman**:
- Predict: $\hat{s}_{k|k-1} = F \hat{s}_{k-1|k-1}$, $P_{k|k-1} = F P_{k-1|k-1} F^T + Q$
- Update (z available): $K_k = P_{k|k-1} H^T / (H P_{k|k-1} H^T + R)$,
  $\hat{s}_{k|k} = \hat{s}_{k|k-1} + K_k (z_k - H \hat{s}_{k|k-1})$,
  $P_{k|k} = (I - K_k H) P_{k|k-1}$
- Skip update at NaN (prediction-only)

**Backward RTS smoother**:
$$C_k = P_{k|k} F^T P_{k+1|k}^{-1}$$
$$\hat{s}_{k|N} = \hat{s}_{k|k} + C_k (\hat{s}_{k+1|N} - \hat{s}_{k+1|k})$$

Hyperparameters (v0.1.3):
- $q = 1.0$ (process noise scale, velocity volatility) — 클수록 less smoothing
- $R = 4.0$ (measurement variance, ~2 mm std) — 작을수록 trust measurement more

#### 2.2.4 알고리즘 흐름

**Forward Kalman pass (k=1, 2, ..., N)**:
```
1. Predict:
     ŝ_{k|k-1} = F · ŝ_{k-1|k-1}                    (state extrapolation)
     P_{k|k-1} = F · P_{k-1|k-1} · F^T + Q          (uncertainty grows)

2. Update (when z_k available):
     y_k = z_k - H · ŝ_{k|k-1}                      (innovation, residual)
     S_k = H · P_{k|k-1} · H^T + R                  (innovation covariance)
     K_k = P_{k|k-1} · H^T · S_k^{-1}               (Kalman gain)
     ŝ_{k|k} = ŝ_{k|k-1} + K_k · y_k                (state update)
     P_{k|k} = (I - K_k · H) · P_{k|k-1}            (uncertainty reduces)

     If z_k missing (NaN): prediction-only, skip update.
```

**RTS backward pass (k=N-1, N-2, ..., 1)**:
```
   C_k = P_{k|k} · F^T · P_{k+1|k}^{-1}             (smoother gain)
   ŝ_{k|N} = ŝ_{k|k} + C_k · (ŝ_{k+1|N} - ŝ_{k+1|k}) (incorporate future)
   P_{k|N} = P_{k|k} + C_k · (P_{k+1|N} - P_{k+1|k}) · C_k^T
```

각 timestep에서 미래 정보를 backward 흐름으로 통합 → **offline smoother**.

#### 2.2.5 장단점
| 장점 | 단점 |
|---|---|
| ✅ Principled noise model (Q, R 명시) | ❌ Q, R tuning 필요 |
| ✅ Per-frame uncertainty (P_{k|N}) | ❌ Linear + Gaussian 가정 |
| ✅ Missing data 자연스럽게 처리 | ❌ 구현 복잡 (forward + backward) |
| ✅ Per-frame measurement weight 가능 | ❌ Dynamics 모델 잘못 선택 시 bias |
| ✅ Extension: bone-length state, 6D pose | ❌ Smoothing strength = f(Q/R), 직관적이지 않음 |

#### 2.2.6 Why both methods are equivalent on our data
**핵심**: 두 알고리즘 모두 essentially **low-pass FIR filter** (~10 Hz cutoff @ 100 fps).
- SavGol convolution kernel ≈ 15-tap FIR with negative side-lobes (Savitzky-Golay 다항식)
- Kalman+RTS의 smoothing kernel ≈ exponential-ish FIR (Q/R ratio가 effective bandwidth 결정)
- 노이즈가 고주파, 신호가 저주파인 경우 → 둘 다 동일한 cutoff 적용 시 동일한 결과

**Kalman이 이길 조건** (현재 미충족):
1. Per-frame R = f(DLC likelihood) — 낮은 신뢰 frame은 큰 R로 자동 down-weight
2. Higher-order dynamics (가속도, jerk 모델) — 갑작스러운 mouse 점프 처리
3. Multi-kp joint state with bone-length constraint
4. Out-of-distribution velocity prediction (long NaN gap 메우기)

→ 이 4가지 모두 v0.2 candidate

### 2.3 Benchmark: SavGol vs Kalman

| Method | mammal_3600 MPJPE | OOD Li MPJPE | jitter reduction |
|---|---|---|---|
| Raw (no smoothing) | 23.10 [22.68, 23.54] | 19.90 [18.47, 21.37] | — |
| **Savitzky-Golay w=15** | **22.33** [21.95, 22.72] | **19.14** [17.73, 20.63] | -63% |
| **Kalman+RTS q=1 R=4** | 22.39 [22.00, 22.79] | 19.21 [17.82, 20.67] | -61% |

**정확한 결론** (Karpathy: don't overclaim):

- **MPJPE 차이 0.07 mm는 bootstrap noise 범위 내** (CI 폭 ~3 mm)
- 95% CI 거의 완전 overlap → **통계적으로 무차별**
- SavGol이 "더 성공"이 아니라 **"통계적 동치"**
- Default를 SavGol로 둔 이유는 **정확도가 아닌 단순성** (scipy 1-line vs 60 lines)

Kalman 장점 (현재 활용 안 함):
- 측정 노이즈 R을 per-frame likelihood로 가중 가능 (extension)
- Out-of-distribution velocity 추정
- Future: bone-length constraint state 추가 가능

### 2.4 미실험 alternatives

- **Anipose `triangulate_optim`**: spatial bone-length prior + temporal
- **Particle filter**: nonlinear dynamics (mouse가 갑자기 점프 등)
- **Bidirectional LSTM denoiser**: data-driven smoothing

---

## §3. Implementation Files (SSOT)

| File | Purpose |
|---|---|
| `scripts/anipose_triangulate.py` | aniposelib API call (validation) |
| `/tmp/full_kp_dataset.py` | canonical DLT triangulation (executable on gpu03) |
| `scripts/smooth_kp_temporal.py` | Savitzky-Golay smoothing |
| `scripts/smooth_kalman_rts.py` | Kalman+RTS smoothing (this doc §2.2) |
| `scripts/render_kp_results_report.py` | HTML report generator |
| `outputs/kp_benchmark/results_smoothing_compare_v2.csv` | benchmark numbers |

## §4. v0.2 후보 (deferred)

1. **Anipose `triangulate_optim`** with bone-length prior (anatomical constraint)
2. **Likelihood-weighted Kalman R** (per-frame measurement noise from DLC likelihood)
3. **Bone-length state augmentation** (Kalman state = position + velocity + segment length)
4. **Particle filter** for occlusion handling
5. **Cam2 deep-dive resolution** (sub-pixel calibration but user perceives offset)
