# gpu03 Storage Management Guide

Server-wide 3-tier storage strategy for `/home/joon/` on gpu03.

---

## 1. Physical Volumes

| Mount | Device | Type | Total | Used | Free | Purpose |
|-------|--------|------|------:|-----:|-----:|---------|
| `/` (HOME) | nvme1n1p2 | NVMe | 879G | 437G | 397G | Code, envs, scripts |
| `/node_data` | nvme0n1p1 | NVMe | 7.0T | 5.3T | 1.4T | Training data & active checkpoints |
| `/node_data_2` | sda1 | HDD | 7.0T | 2.7T | 4.0T | Archive & cache |

> Last audited: 2026-03-26

---

## 2. Tier Definitions

```
┌──────┬──────────────────┬──────────────────────────────────┬──────────────────────────────────────────┐
│ Tier │ Location         │ Criteria                         │ Contents                                 │
├──────┼──────────────────┼──────────────────────────────────┼──────────────────────────────────────────┤
│ HOT  │ /node_data NVMe  │ Training read/write per-batch    │ datasets, active checkpoints              │
│      │                  │                                  │ /node_data/joon/{data,checkpoints}        │
├──────┼──────────────────┼──────────────────────────────────┼──────────────────────────────────────────┤
│ WARM │ /node_data NVMe  │ Frequent reference, not training │ wandb_logs, recent eval results           │
│      │                  │                                  │ /node_data/joon/{wandb_logs,outputs}      │
├──────┼──────────────────┼──────────────────────────────────┼──────────────────────────────────────────┤
│ COLD │ /node_data_2 HDD │ Rarely accessed, archive         │ _archive, old checkpoints, hf_cache       │
│      │                  │ + re-downloadable large caches   │ /node_data_2/joon/{checkpoints,outputs,   │
│      │                  │                                  │  hf_cache,wandb_runs,quarantine}           │
├──────┼──────────────────┼──────────────────────────────────┼──────────────────────────────────────────┤
│ HOME │ / NVMe (879G)    │ Code & config only               │ git repos, conda envs, pip cache          │
│      │                  │ Outputs = archival candidates     │ /home/joon/dev/<project>/                 │
└──────┴──────────────────┴──────────────────────────────────┴──────────────────────────────────────────┘
```

### HOME Tier Rules

- Code repos live at `/home/joon/dev/<project>/`
- Project `outputs/` and `results/` directories are acceptable while active
- Once a project completes → archive outputs to COLD via symlink pattern (see §4)
- `/home/joon/data` is a **permanent symlink** → `/node_data/joon/data` (config path consistency)

### HOT/WARM Boundary

Both on the same NVMe — the distinction is access pattern, not location:
- **HOT**: Read every training step (datasets, active model checkpoint being trained)
- **WARM**: Referenced occasionally (completed experiment results, eval outputs)
- When node_data fills up → move WARM items to COLD first

### COLD Tier Rules

- Archive-only. Symlinked from original location for backward compatibility
- `hf_cache` (127G): Re-downloadable model weights. Lowest preservation priority
- ~~`quarantine/`: Staging area for items pending review (30-day hold → delete or classify)~~ — **폐지 (2026-07-28)**: 마지막 격리분 2건이 예정 삭제일(2026-04-22)을 97일 초과해 방치 → 삭제하고 디렉토리 자체를 없앰. 30-day hold가 실제로 집행되지 않아 무기한 보관소로 변질됨. 재도입 시 만료 강제 수단(cron 등) 동반 필요
- All files default permissions (755/644). No special chmod

---

## 3. Current Inventory (2026-03-25)

### HOME — `/home/joon/dev/` (55G code repos)

| Project | Size | Notes |
|---------|-----:|-------|
| FaceLift | 37G | outputs/_archive → node_data_2 (symlink) |
| sdannce-poc | 7.1G | outputs/ 134M (archival candidate) |
| flux_cond_img | 6.0G | outputs/ 693M (archival candidate) |
| pose-splatter | 3.8G | |
| MAMMAL_mouse | 371M | results/fitting 290M (active job) |
| Others | ~5G | bams, qwen-multimodal, etc. |

### HOT/WARM — `/node_data/joon/` (200G)

| Path | Size | Tier | Notes |
|------|-----:|------|-------|
| checkpoints/FaceLift/gslrm/ | 50G | HOT+WARM | 7 active (8 archived to COLD) |
| checkpoints/FaceLift/mvdiffusion/ | 55G | HOT+WARM | 4 active (mouse_M5t2 + H7v2 8k/9k archived) |
| data/preprocessed/ | 39G | HOT | M5t2 training data |
| data/sdannce/ | 9.2G | HOT | Behavior keypoints |
| data/benchmarks/ | 4.4G | HOT | Evaluation benchmarks |
| data/raw/ | 857M | WARM | Original video/images |
| wandb_logs/ | 238M | WARM | Active W&B logs |
| checkpoints/FaceLift/{eval,sam,deformation} | ~1G | WARM | Utility models |

**Key checkpoints (KEEP HOT):**
- `gslrm/base_uniform_v2_6view_v2` (3.6G) — 6-view best, PSNR=23.84
- `gslrm/M5t2_E0_1_facelift` (7.5G) — 4-view best
- `gslrm/RAT1_rat_ft_v1` (7.1G) — Rat fine-tune
- `gslrm/M5t2_6view_alpha*` (~18G) — Alpha ablation (paper evidence)
- `mvdiffusion/mouse_M5t2_H7v2_spatial_token` (17G) — Latest MVDiff (ckpt-10000)

**Archived to COLD (2026-03-26, 42G total):**
- `gslrm/base_uniform_v2_hp_M5_4` (3.6G) — ✅ symlink
- `gslrm/base_uniform_v2_hp_M5_5` (3.6G) — ✅ symlink
- `gslrm/base_uniform_v2_hp_M0` (3.6G) — ✅ symlink
- `gslrm/base_uniform_v2_4view_ssim03_v2` (3.6G) — ✅ symlink
- `gslrm/base_uniform_v2_4view_ssim05_v2` (3.6G) — ✅ symlink
- `gslrm/base_uniform_v2_4view_ssim10_v2` (3.6G) — ✅ symlink
- `gslrm/base_uniform_v2_domain_adapt_E2_v1` (3.6G) — ✅ symlink
- `gslrm/base_uniform_v2_4view_v2` (3.6G) — ✅ symlink
- `mvdiffusion/mouse_M5t2` (14G) — ✅ symlink

### COLD — `/node_data_2/joon/` (162G)

| Path | Size | Notes |
|------|-----:|-------|
| hf_cache/hub/ | 126G | HuggingFace models (re-downloadable) |
| checkpoints/FaceLift/ | 68G | H7v2 ckpt-8000/9000 + 8 gslrm + mouse_M5t2 (all symlinked) |
| ~~quarantine/~~ | ~~5.4G~~ | **삭제 완료 2026-07-28** (Legacy wandb 2.6G + gaussian samples 2.8G, 예정 삭제일 2026-04-22 경과) |
| wandb_runs/ | 3.2G | Old W&B experiment runs |
| outputs/FaceLift/_archive/ | 463M | Archived FaceLift outputs |

---

## 4. Symlink Patterns

### Pattern A — Data Path Alias (permanent)
```
/home/joon/data → /node_data/joon/data
```
Purpose: Config files reference `/home/joon/data/...`, actual data on NVMe.

### Pattern B — Cold Archive (created when archiving)
```
<project>/outputs/_archive → /node_data_2/joon/outputs/<project>/_archive
node_data/.../checkpoint-NNNN → /node_data_2/joon/checkpoints/.../checkpoint-NNNN
```
Purpose: Archive to HDD, keep backward-compatible paths.

### Archive Procedure
```bash
# 1. rsync with I/O throttling (shared server)
ionice -c 3 nice -n 19 rsync -av --progress <source>/ <node_data_2_dest>/

# 2. Verify file count
find <source> -type f | wc -l
find <dest> -type f | wc -l

# 3. Replace with symlink
rm -rf <source>
ln -s <node_data_2_dest> <source>

# 4. Verify symlink works
ls <source>/
```

### Symlink Health Check
```bash
# Detect broken symlinks under /home/joon
find /home/joon/dev -type l ! -exec test -e {} \; -print 2>/dev/null
find /node_data/joon -type l ! -exec test -e {} \; -print 2>/dev/null
```

---

## 5. VS Code Workspace Settings

### 5.1 Workspace File (`server.code-workspace`)

Only code/config folders as workspace roots. **Data and archive folders must NOT be workspace roots.**

```
✅ Workspace roots: dotfiles, dev/, .agent
❌ Never add: /home/joon/data, /node_data_2/joon/, individual project duplicates
```

Workspace-level settings provide defaults; project-level `.vscode/settings.json` overrides.

### 5.2 Project-Level Settings (all 7 projects, unified)

```json
{
    "files.watcherExclude": {
        "**/outputs/**": true, "**/results/**": true,
        "**/datasets/**": true, "**/data/**": true,
        "**/checkpoints/**": true, "**/logs/**": true,
        "**/wandb/**": true, "**/tensorboard/**": true,
        "**/.git/**": true
    },
    "search.exclude": { "...same patterns..." },
    "python.analysis.exclude": [
        "**/outputs", "**/checkpoints", "**/data", "**/datasets",
        "**/logs", "**/wandb", "**/tensorboard", "**/.venv", "**/venv"
    ],
    "git.autorefresh": false,
    "search.followSymlinks": false
}
```

NO `files.exclude` — all folders visible in Explorer for easy access.
Protected from background I/O via `watcherExclude` + `python.analysis.exclude`.

### 5.3 Settings Quick Reference

| Setting | What it does | Why needed |
|---------|-------------|------------|
| `files.watcherExclude` | Stop file change monitoring | **Most important — saves server I/O** |
| `search.exclude` | Skip from Ctrl+Shift+F | Saves CPU |
| `python.analysis.exclude` | Pylance ignores these dirs | **Saves ~1GB RAM** |
| `git.autorefresh: false` | No auto git status check | Saves I/O on 4.8G .git |
| `search.followSymlinks: false` | Don't follow symlinks when searching | Prevents I/O storms |

### 5.4 Accessing Excluded Files

Excluded files are **still fully accessible** via VS Code integrated terminal:
```bash
ls checkpoints/              # browse
cat outputs/some_file.json   # read
du -sh outputs/              # check size
```

> **Key**: `files.watcherExclude` (singular), NOT `files.watcherExcludes` (plural).
> The plural form is silently ignored by VS Code.

---

## 6. Known Issues & TODO

| Issue | Status | Action |
|-------|--------|--------|
| node_data 99%→80% | ✅ Resolved | 42G archived (2026-03-26), 1.4T free |
| wandb split (node_data + node_data_2) | ⚠️ Pending | Consolidate under single WANDB_DIR |
| hf_cache 127G on HDD | ℹ️ Acceptable | 1-time load latency only. Audit unused models |
| quarantine 5.4G | ✅ Resolved (2026-07-28) | 삭제 완료, quarantine 개념 폐지 (§2 참조) |
| `~/data/preprocessed/WK1_v2_sc_fx1000` symlink 오연결 | ✅ Resolved (2026-07-28) | 권장(subject_centered) 이름이 비권장(square_min) 데이터를 가리킴 → 재연결 + `WK1_fx1000_full` symlink 신설. 상세 = `~/dev/BehaviorSplatter/docs/DATASET_LOCATIONS.md` |
| sdannce-poc/flux outputs on HOME | ℹ️ Low priority | Archive when projects complete |

---

## 7. Maintenance Checklist

### Monthly
- [ ] `df -h /home /node_data /node_data_2` — check capacity
- [ ] Broken symlink check (§4 health check command)
- [ ] **Mis-pointed** symlink check — 이름과 대상 basename 불일치 탐지. broken 체크로는 안 잡히는 유형 (260728 WK1 사고). 260728 전수 실행 결과 잔여 0건:

```bash
for base in ~/data ~/data/preprocessed ~/data/shared /node_data/joon/checkpoints/FaceLift/gslrm; do
  for l in "$base"/*; do
    [ -L "$l" ] || continue
    t=$(readlink -f "$l")
    [ "$(basename "$l")" = "$(basename "$t")" ] || echo "MISMATCH: $l -> $t"
  done
done
```

### Per Experiment Completion
- [ ] Move intermediate checkpoints (non-best) → COLD
- [ ] Archive outputs if project phase complete → COLD with symlink

### Quarterly
- [ ] hf_cache audit: `du -sh /node_data_2/joon/hf_cache/hub/models--*` → delete unused
- [ ] wandb old runs cleanup (> 6 months)
- [ ] Verify VS Code settings consistency across projects

---

## 8. Monitoring

Two bash aliases in `~/.bashrc` on gpu03:

```bash
stor           # Quick: df -h for all 3 tiers (no HDD I/O, instant)
stor-detail    # Full: du -sh per tier + broken symlink check (takes seconds)
```

No cron, no VS Code integration, no Python scripts needed.
COLD data is static — on-demand `stor` is sufficient.

---

*Created: 2026-03-25 | Updated: 2026-03-26 | gpu03 Storage Management Guide v1.1*
