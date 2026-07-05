"""Run the behavior analysis workbench on a single dataset slice.

This is a file-backed driver for methods that spawn subprocesses.
It avoids stdin-based multiprocessing issues when running isolated
datasets from the shell or notebooks.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_behavior_workbench_batch import (
    METHODS,
    OUT_DIR,
    asdict as _asdict,
    error_result,
    load_datasets,
    plot_summary,
    write_html,
)


def run_dataset(name: str) -> pd.DataFrame:
    ds = [d for d in load_datasets() if d.name == name][0]
    results = []
    for method, fn in METHODS.items():
        print(f"  {method}...", flush=True)
        try:
            result = fn(ds)
            print(
                f"    ok: clusters={result.n_clusters}, sil={result.silhouette}, {result.elapsed_sec:.1f}s",
                flush=True,
            )
        except Exception as exc:
            result = error_result(ds, method, exc)
            print(f"    error: {result.error}", flush=True)
        results.append(result)
    return pd.DataFrame([_asdict(r) for r in results])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    df = run_dataset(args.dataset)
    out_dir = OUT_DIR / "single_runs" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / "results.csv"
    df.to_csv(out_path, index=False)
    (out_dir / "results.json").write_text(json.dumps(df.to_dict(orient="records"), indent=2), encoding="utf-8")
    plot_summary(df)
    write_html(df, [d for d in load_datasets() if d.name == args.dataset])
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
