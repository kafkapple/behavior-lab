"""Save a reproducible descriptive Stats + DEK bundle for an existing labeled project."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import core  # noqa: E402
import dek  # noqa: E402
import stats  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--project", default="cort_vs_veh")
    p.add_argument("--level", type=int, default=8)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    proj = core.load_project(root, a.project)
    paths = list(dict.fromkeys(path for sessions in proj["groups"].values() for path in sessions))
    ses = {path: core.load(root, path) for path in paths}
    fit_ids = {s["fit_id"] for s in ses.values()}
    if len(fit_ids) != 1 or None in fit_ids:
        raise ValueError("project sessions must have labels from one fit")
    out = a.output or Path("outputs/subtle") / a.project / (datetime.now().strftime("%y%m%d_%H%M%S") + f"_L{a.level}_descriptive")
    out.mkdir(parents=True, exist_ok=True)
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "project": a.project,
                  "project_file": str(root / ".subtle_projects" / f"{a.project}.project.json"),
                  "data_root": str(root), "groups": proj["groups"], "fit_id": next(iter(fit_ids)),
                  "level": f"super L{a.level}", "frames": {path: len(s["xyz"]) for path, s in ses.items()},
                  "code_commit": core.code_commit(), "group_note": "folder-defined assignments are provisional; no treatment inference"}
    print("Stats: animal-level shares", flush=True)
    S = stats.group_stats(root, proj, "super", a.level)
    rows = []
    for pair in S["pairs"]:
        ga, gb = pair["groups"]
        for t in pair["tests"]:
            rows.append({"group_a": ga, "group_b": gb, "cluster": t["cluster"], "mean_a_pct": t["mean"][ga],
                         "sd_a_pct": t["sd"][ga], "animals_a_pct": json.dumps(t["values"][ga]),
                         "mean_b_pct": t["mean"][gb], "sd_b_pct": t["sd"][gb], "animals_b_pct": json.dumps(t["values"][gb]),
                         "hedges_g": t["g"], "p_raw": t["p"], "p_bh": t["p_adj"]})
    with (out / "group_stats.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for i, pair in enumerate(S["pairs"]):
        ga, gb = pair["groups"]
        fig, ax = plt.subplots(figsize=(10, max(4, len(pair["tests"]) * .48)))
        for j, t in enumerate(pair["tests"]):
            for group, offset, color in ((ga, -.13, "#2687b8"), (gb, .13, "#c77712")):
                vals = t["values"][group]; y = np.full(len(vals), j + offset)
                ax.scatter(vals, y, color=color, s=17, alpha=.55)
                ax.errorbar(t["mean"][group], j + offset, xerr=t["sd"][group] or 0, fmt="|", color=color, capsize=3)
            ax.text(101, j, f"pBH={t['p_adj']:.3f}", va="center", fontsize=8)
        ax.set(yticks=range(len(pair["tests"])), yticklabels=[str(t["cluster"]) for t in pair["tests"]],
               xlim=(-2, 124), xlabel="Share of frames per animal (%)", ylabel="Cluster ID",
               title=f"{ga} (n={pair['n'][ga]}) vs {gb} (n={pair['n'][gb]}) · super L{a.level} · fit {provenance['fit_id']}")
        ax.grid(axis="x", alpha=.2); fig.tight_layout(); fig.savefig(out / f"group_stats_pair{i}.png", dpi=160); plt.close(fig)
    print("DEK: kinematics", flush=True)
    D = dek.group_dek(root, proj, "super", a.level, progress=lambda done, total, rel: print(f"  {done}/{total} {rel}", flush=True))
    with (out / "dek.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["group_a", "group_b", "cluster", "n_a", "n_b", "feature", "t", "p_raw", "p_bh", "hedges_g"])
        for pair in D["pairs"]:
            for r, cluster in enumerate(pair["clusters"]):
                for j, feature in enumerate(D["columns"]):
                    w.writerow([*pair["groups"], cluster, *pair["n"][r], feature, pair["t"][r][j], pair["p"][r][j], pair["p_adj"][r][j], pair["g"][r][j]])
    for i, pair in enumerate(D["pairs"]):
        for typ in "VEA":
            for derivative in range(3):
                idx = [j for j, col in enumerate(D["columns"]) if col.startswith(typ + str(derivative))]
                values = np.array([[np.nan if pair["t"][r][j] is None else pair["t"][r][j] for j in idx] for r in range(len(pair["clusters"]))])
                fig, ax = plt.subplots(figsize=(max(7, len(idx) * .42), max(4, len(pair["clusters"]) * .42)))
                finite = values[np.isfinite(values)]; limit = max(2, float(np.max(np.abs(finite))) if finite.size else 2)
                im = ax.imshow(values, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
                ax.set(xticks=range(len(idx)), xticklabels=[D["columns"][j].split(":", 1)[1] for j in idx],
                       yticks=range(len(pair["clusters"])), yticklabels=pair["clusters"], title=f"DEK {typ}{derivative} · {pair['groups'][0]} vs {pair['groups'][1]} · fit {provenance['fit_id']}")
                for r in range(len(pair["clusters"])):
                    for k, j in enumerate(idx):
                        q = pair["p_adj"][r][j]
                        if q is not None and q < .05:
                            ax.text(k, r, "*", ha="center", va="center", color="black", fontsize=15, fontweight="bold")
                ax.tick_params(axis="x", rotation=60); fig.colorbar(im, ax=ax, label="Welch t; + = first group higher")
                fig.tight_layout(); fig.savefig(out / f"dek_pair{i}_{typ}{derivative}.png", dpi=160); plt.close(fig)
    finite_p = [q for pair in D["pairs"] for row in pair["p_adj"] for q in row if q is not None]
    provenance["stats"] = {"one_fit": S["one_fit"], "pairs": len(S["pairs"]), "tests": len(rows),
                            "min_p_bh": min(t["p_adj"] for pair in S["pairs"] for t in pair["tests"]),
                            "n_p_bh_below_005": sum(t["p_adj"] < .05 for pair in S["pairs"] for t in pair["tests"])}
    provenance["dek"] = {"cells_tested": sum(pair["tests"] for pair in D["pairs"]), "min_p_bh": min(finite_p) if finite_p else None,
                          "n_p_bh_below_005": sum(q < .05 for q in finite_p), "min_frames_per_animal_cluster": D["min_frames"]}
    (out / "summary.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    hits = [(pair["groups"], cluster, D["columns"][j], pair["p_adj"][r][j]) for pair in D["pairs"]
            for r, cluster in enumerate(pair["clusters"]) for j in range(len(D["columns"]))
            if pair["p_adj"][r][j] is not None and pair["p_adj"][r][j] < .05]
    note = [f"# {a.project}: descriptive group comparison", "",
            f"Existing fit `{provenance['fit_id']}`, super L{a.level}; {sum(provenance['frames'].values()):,} frames from {len(paths)} sessions.",
            f"Group assignment: {', '.join(f'{g} n={len(s)}' for g, s in proj['groups'].items())}. Folder names only; treatment assignment is unverified.", "",
            f"Cluster share tests: {len(rows)}; minimum BH p={provenance['stats']['min_p_bh']:.4f}; {provenance['stats']['n_p_bh_below_005']} below .05.",
            f"DEK: {provenance['dek']['cells_tested']} tested cells; minimum BH p={provenance['dek']['min_p_bh']:.4f} when available; {len(hits)} below .05." if provenance["dek"]["min_p_bh"] is not None else "DEK: no eligible cells.",
            "DEK cells below .05 (exploratory):"]
    note += [f"- {groups[0]} vs {groups[1]}, cluster {cluster}, {feature}, pBH={q:.4f}" for groups, cluster, feature, q in hits]
    note += ["", "`group_stats.csv` has animal shares and sample SD; `group_stats_pair0.png` shows animals and mean ± SD.",
             "`dek.csv` has all feature cells; `dek_pair0_<V|E|A><0|1|2>.png` shows signed Welch t, with stars for adjusted p<.05.",
             "`summary.json` records the exact input paths, frame counts, fit ID and code version.", "",
             "Limits: this is exploratory; cluster levels and DEK cells were inspected together, folder groups are provisional, and copied 3D points can affect kinematics. No treatment claim follows from these plots."]
    (out / "README.md").write_text("\n".join(note) + "\n")
    print(json.dumps({"output": str(out), "stats": provenance["stats"], "dek": provenance["dek"]}), flush=True)


if __name__ == "__main__":
    main()
