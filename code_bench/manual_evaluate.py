#!/usr/bin/env python3
"""
Interactive replacement for evaluate.py.

Goal:
  - Read judge outputs from test_results_*.jsonl (produced by test.py)
  - Let HUMAN extract the final evaluation label: correct / incorrect / no evaluation
  - Save an evaluate_results_*.jsonl-like file for downstream analysis (get_res.py)

This replaces the "evaluator_model" step with a human.

Output record mirrors evaluate.py fields:
  idx, task_title, task_rating, solution_model, solution_occurrence, ci_pair_id,
  prompt_label, judge_model, evaluator_model="human", evaluation, timestamp

Logs:
  all_logs/manual_evaluate_<timestamp>.log
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional


def _safe_model_name(model: str) -> str:
    return model.replace("/", "_").replace("\\", "_")


def _setup_file_logging(log_path: Optional[str] = None) -> str:
    if log_path is None:
        os.makedirs("all_logs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join("all_logs", f"manual_evaluate_{ts}.log")
    else:
        parent = os.path.dirname(log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    return log_path


def _print_and_log(s: str) -> None:
    print(s)
    logging.info(s)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                out.append(rec)
    return out


def _prompt_eval_label() -> str:
    """
    Human extraction of evaluation label.
    """
    while True:
        ans = input(
            "Extract label: [c]orrect / [i]ncorrect / [n]o-eval / [s]kip / [q]uit: "
        ).strip().lower()
        if ans in ("c", "correct"):
            return "correct"
        if ans in ("i", "incorrect"):
            return "incorrect"
        if ans in ("n", "no", "noeval", "no-eval", "no evaluation", "no_evaluation"):
            return "no evaluation"
        if ans in ("s", "skip"):
            return "skip"
        if ans in ("q", "quit"):
            return "quit"


def _filter_records(
    records: List[Dict[str, Any]],
    *,
    only_ci_none: bool,
    prompt_labels: Optional[List[str]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    allowed = set([p.lower() for p in prompt_labels]) if prompt_labels else None
    for r in records or []:
        if only_ci_none and r.get("ci_pair_id") is not None:
            continue
        pl = r.get("prompt_label")
        if allowed is not None and (str(pl).lower() not in allowed):
            continue
        out.append(r)
    return out


def _sample(records: List[Dict[str, Any]], *, k: Optional[int], seed: Optional[int]) -> List[Dict[str, Any]]:
    if k is None:
        return list(records)
    k = int(k)
    if k <= 0:
        return []
    rng = random.Random(seed) if seed is not None else random.Random()
    if len(records) <= k:
        return list(records)
    return rng.sample(records, k=k)


def _list_test_results_files(output_dir: str) -> List[str]:
    """
    Return full paths to all test_results_*.jsonl in output_dir (non-recursive).
    """
    if not os.path.exists(output_dir):
        return []
    try:
        names = os.listdir(output_dir)
    except OSError:
        return []
    out: List[str] = []
    for name in names:
        if not (name.startswith("test_results_") and name.endswith(".jsonl")):
            continue
        out.append(os.path.join(output_dir, name))
    return sorted(out)


def _model_suffix_from_test_results_filename(path: str) -> str:
    """
    test_results_<suffix>.jsonl -> <suffix>
    """
    base = os.path.basename(path)
    if base.startswith("test_results_") and base.endswith(".jsonl"):
        return base[len("test_results_") : -len(".jsonl")]
    return _safe_model_name(base)


def _run_one_file(
    *,
    test_file: str,
    output_dir: str,
    out_file: Optional[str],
    k: int,
    seed: int,
    only_ci_none: bool,
    prompt_labels: Optional[List[str]],
) -> bool:
    """
    Returns False if user chose to quit, True otherwise.
    """
    test_records = _read_jsonl(test_file)
    if not test_records:
        _print_and_log(f"Skip: no records in {test_file}")
        return True

    filtered = _filter_records(test_records, only_ci_none=only_ci_none, prompt_labels=prompt_labels)
    if not filtered:
        _print_and_log(f"Skip: no records after filtering in {test_file}")
        return True

    picked = _sample(filtered, k=k, seed=seed)
    suffix = _model_suffix_from_test_results_filename(test_file)
    os.makedirs(output_dir, exist_ok=True)
    dst = out_file or os.path.join(output_dir, f"evaluate_results_manual_{suffix}.jsonl")
    _print_and_log(f"\n### File: {os.path.basename(test_file)}")
    _print_and_log(f"Picked {len(picked)} records, writing to: {dst}")

    out_rows: List[Dict[str, Any]] = []
    counts = {"correct": 0, "incorrect": 0, "no evaluation": 0, "skip": 0}

    for i, r in enumerate(picked, 1):
        idx = r.get("idx")
        prompt_label = r.get("prompt_label")
        solution_model = r.get("solution_model")
        solution_occurrence = r.get("solution_occurrence")
        ci_pair_id = r.get("ci_pair_id")

        _print_and_log("\n" + "=" * 80)
        _print_and_log(f"[{i}/{len(picked)}] idx={idx} prompt_label={prompt_label} ci_pair_id={ci_pair_id}")
        _print_and_log(f"solution_model={solution_model} solution_occurrence={solution_occurrence}")
        if r.get("task_title"):
            _print_and_log(f"task_title={r.get('task_title')}")
        if r.get("task_rating") is not None:
            _print_and_log(f"task_rating={r.get('task_rating')}")
        _print_and_log("\n--- JUDGE RESPONSE (raw) ---\n" + str(r.get("response") or ""))

        ans = _prompt_eval_label()
        if ans == "quit":
            _print_and_log("Stopping early (user quit).")
            # Save whatever we have for this file before quitting.
            with open(dst, "w", encoding="utf-8") as f:
                for row in out_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            return False
        if ans == "skip":
            counts["skip"] += 1
            continue

        counts[ans] += 1
        out_rows.append(
            {
                "idx": idx,
                "task_title": r.get("task_title"),
                "task_rating": r.get("task_rating"),
                "solution_model": solution_model,
                "solution_occurrence": solution_occurrence,
                "ci_pair_id": ci_pair_id,
                "prompt_label": prompt_label,
                "judge_model": r.get("judge_model"),
                "evaluator_model": "human",
                "evaluation": ans,
                "timestamp": datetime.now().isoformat(),
            }
        )

    with open(dst, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    _print_and_log(f"Done {os.path.basename(test_file)}. Saved {len(out_rows)} labeled records.")
    _print_and_log(f"Counts: {json.dumps(counts, ensure_ascii=False)}")
    return True


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Manual extraction of evaluate.py labels (human evaluator).")
    parser.add_argument(
        "--test-file",
        type=str,
        default=None,
        help="Path to a single test_results_*.jsonl. If omitted, runs on ALL test_results_*.jsonl in --output-dir.",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default=None,
        help="Output file (only used when --test-file is provided). Otherwise output is per model in output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output dir for default out-file (default: outputs)",
    )
    parser.add_argument("--k", type=int, default=15, help="How many records to label (default: 15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--only-ci-none",
        action="store_true",
        help="Only label records with ci_pair_id=None (default: false)",
    )
    parser.add_argument(
        "--prompt-labels",
        nargs="*",
        default=None,
        help="Limit to these prompt labels, e.g. good bad (default: all)",
    )
    parser.add_argument("--log-file", type=str, default=None, help="Write log to this file (default: all_logs/...)")
    args = parser.parse_args()

    log_path = _setup_file_logging(args.log_file)
    _print_and_log(f"Log file: {log_path}")

    if args.test_file:
        ok = _run_one_file(
            test_file=args.test_file,
            output_dir=args.output_dir,
            out_file=args.out_file,
            k=int(args.k),
            seed=int(args.seed),
            only_ci_none=bool(args.only_ci_none),
            prompt_labels=args.prompt_labels,
        )
        if not ok:
            sys.exit(0)
        return

    # Auto-run on all test_results_*.jsonl in output-dir
    files = _list_test_results_files(args.output_dir)
    if not files:
        print(f"Error: no test_results_*.jsonl found in {args.output_dir}", file=sys.stderr)
        sys.exit(1)
    _print_and_log(f"Found {len(files)} test_results files in {args.output_dir}")

    for idx_file, tf in enumerate(files):
        # Derive per-file seed so sampling is stable but differs between files.
        per_seed = int(args.seed) + idx_file * 10007
        ok = _run_one_file(
            test_file=tf,
            output_dir=args.output_dir,
            out_file=None,
            k=int(args.k),
            seed=per_seed,
            only_ci_none=bool(args.only_ci_none),
            prompt_labels=args.prompt_labels,
        )
        if not ok:
            return


if __name__ == "__main__":
    main()

