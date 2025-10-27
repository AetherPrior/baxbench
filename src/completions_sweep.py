#!/usr/bin/env python3
"""
Multiprocess batch runner: iterate over (env, scenario, spec_type, safety_prompt) rows
in a collated CSV and invoke your main script once per unique combination, in parallel.

Example:
  LOCAL_API_BASE=http://localhost:8000 \
  python tools/batch_run_from_collated_mp.py \
    --csv /collated_outputs/Qwen/Qwen3-8B-4096_cwes/Qwen/Qwen3-8B-4096_instances_rewrites.csv \
    --models Qwen/Qwen3-8B \
    --mode generate \
    --n-samples 1 \
    --temperature 0.001 \
    --results-dir results_reason_intervention \
    --max-concurrent-runs 1 \
    --workers 4 \
    --main-script src/main.py \
    --pipenv \
    --extra-args --vllm
"""

import argparse
import csv
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

REQUIRED_COLS = ["env", "scenario", "spec_type", "safety_prompt"]


def infer_sep(path: str) -> str:
    for sep in [",", "\t", "|", ";"]:
        try:
            pd.read_csv(path, sep=sep, nrows=5)
            return sep
        except Exception:
            continue
    with open(path, "r", newline="") as f:
        sample = f.read(8192)
    try:
        dialect = csv.Sniffer().sniff(sample)
        return dialect.delimiter
    except Exception:
        return ","


def load_unique_combos(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    sep = infer_sep(csv_path)
    df = pd.read_csv(csv_path, sep=sep)
    df = df[df['passed'] == True]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df[REQUIRED_COLS].dropna().copy()
    for col in REQUIRED_COLS:
        df[col] = df[col].astype(str).str.strip()

    uniq = df.drop_duplicates(subset=REQUIRED_COLS).reset_index(drop=True)
    # filter out CWEs that havent passed 
    return uniq


def slugify(s: str) -> str:
    return (
        s.replace("/", "_")
         .replace("\\", "_")
         .replace(" ", "_")
         .replace(":", "_")
         .replace("|", "_")
         .replace("*", "_")
         .replace("?", "_")
         .replace('"', "_")
         .replace("<", "_")
         .replace(">", "_")
    )


def build_cmd(
    main_script: str,
    models: str,
    mode: str,
    n_samples: int,
    temperature: float,
    results_dir: str,
    max_concurrent_runs: int,
    env_name: str,
    scenario: str,
    spec_type: str,
    safety_prompt: str,
    use_pipenv: bool,
    extra_args: Optional[List[str]] = None,
) -> List[str]:
    base = ["pipenv", "run", "python", main_script] if use_pipenv else ["python", main_script]
    args = [
        "--models", models,
        "--mode", mode,
        "--n_samples", str(n_samples),
        "--temperature", str(temperature),
        "--spec_type", spec_type,
        "--safety_prompt", safety_prompt,
        "--max_concurrent_runs", str(max_concurrent_runs),
        "--envs", env_name,
        "--scenarios", scenario,
        "--results_dir", results_dir,
    ]
    if extra_args:
        args += extra_args
    return base + args


def _worker_run(
    cmd: List[str],
    local_api_base: Optional[str],
    timeout: int,
    marker_path: Optional[str],
    dry_run: bool,
) -> Tuple[int, str]:
    """
    Worker process entry:
      - runs a single command (or dry-runs),
      - creates marker on success if provided,
      - returns (rc, printable_command).
    """
    env = os.environ.copy()
    if local_api_base:
        env["LOCAL_API_BASE"] = local_api_base

    printable = " ".join(shlex.quote(x) for x in cmd)

    if dry_run:
        return 0, "[DRY-RUN] " + printable

    try:
        proc = subprocess.run(cmd, env=env, timeout=timeout)
        rc = proc.returncode
        if rc == 0 and marker_path:
            Path(marker_path).touch()
        return rc, printable
    except subprocess.TimeoutExpired:
        return 124, printable  # 124 commonly used for timeout
    except Exception as e:
        return 1, f"{printable}  # EXC: {e}"


def main():
    p = argparse.ArgumentParser(description="Multiprocess batch runner for src/main.py over collated CSV combos.")
    p.add_argument("--csv", required=True, help="Path to collated *_instances_rewrites.csv")
    p.add_argument("--models", required=True, help="Model id (e.g., Qwen/Qwen3-8B)")
    p.add_argument("--mode", default="generate")
    p.add_argument("--n-samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.001)
    p.add_argument("--results-dir", default="results_reason_intervention")
    p.add_argument("--max-concurrent-runs", type=int, default=1, help="Passed through to your main.py")
    p.add_argument("--workers", type=int, default=2, help="Parallel processes to spawn here in the batch runner")
    p.add_argument("--main-script", default="src/main.py", help="Entry script to invoke")
    p.add_argument("--pipenv", action="store_true", help="Use 'pipenv run python' to invoke")
    p.add_argument("--local-api-base", default=os.getenv("LOCAL_API_BASE", None), help="Value for LOCAL_API_BASE env")
    p.add_argument("--timeout", type=int, default=1200, help="Per-run timeout seconds")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true", help="Skip combos already marked done")
    p.add_argument("--limit", type=int, default=0, help="Run only first N combos (for testing)")
    p.add_argument("--shuffle", action="store_true", help="Shuffle order before dispatch")
    p.add_argument("--extra-args", nargs=argparse.REMAINDER, help="Any extra args to append (e.g. --vllm)")

    args = p.parse_args()

    uniq = load_unique_combos(args.csv)
    if args.shuffle:
        uniq = uniq.sample(frac=1.0, random_state=42).reset_index(drop=True)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    done_dir = results_dir / ".done"
    if args.resume:
        done_dir.mkdir(parents=True, exist_ok=True)

    combos: Iterable[Tuple[str, str, str, str]] = uniq[REQUIRED_COLS].itertuples(index=False, name=None)

    # Pre-build the task list
    tasks = []
    for env_name, scenario, spec_type, safety_prompt in combos:
        if "net-http" in env_name:
            env_name = "Go-net/http"

        slug = f"env={slugify(env_name)}__scenario={slugify(scenario)}__spec={slugify(spec_type)}__safety={slugify(safety_prompt)}.done"
        marker = str(done_dir / slug) if args.resume else None
        if marker and os.path.exists(marker):
            print(f"[skip] already done: {slug}")
            continue

        cmd = build_cmd(
            main_script=args.main_script,
            models=args.models,
            mode=args.mode,
            n_samples=args.n_samples,
            temperature=args.temperature,
            results_dir=args.results_dir,
            max_concurrent_runs=args.max_concurrent_runs,
            env_name=env_name,
            scenario=scenario,
            spec_type=spec_type,
            safety_prompt=safety_prompt,
            use_pipenv=args.pipenv,
            extra_args=args.extra_args,
        )
        tasks.append((cmd, args.local_api_base, args.timeout, marker, args.dry_run))

        if args.limit and len(tasks) >= args.limit:
            break

    if not tasks:
        print("No tasks to run. (All done or filtered out.)")
        return

    print(f"[batch] Using LOCAL_API_BASE={args.local_api_base or os.getenv('LOCAL_API_BASE')}")
    print(f"[batch] Dispatching {len(tasks)} task(s) with --workers={args.workers}\n")

    # Dispatch in parallel
    successes, failures = 0, 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [ex.submit(_worker_run, *t) for t in tasks]
        for fut in as_completed(futures):
            rc, printable = fut.result()
            if rc == 0:
                print(f"[OK] {printable}")
                successes += 1
            else:
                print(f"[ERR {rc}] {printable}", file=sys.stderr)
                failures += 1

    print(f"\nDone. Success: {successes}  |  Fail: {failures}  |  Total: {successes + failures}")


if __name__ == "__main__":
    main()
