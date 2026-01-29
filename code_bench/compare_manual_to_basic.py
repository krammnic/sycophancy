#!/usr/bin/env python3
"""
Compare manual-evaluated labels vs the "basic" evaluate.py results.

Inputs (default):
  outputs/evaluate_results_manual_<suffix>.jsonl   (human labels, from manual_evaluate.py)
  outputs/evaluate_results_<suffix>.jsonl          (basic/evaluate.py output)

Matching key:
  (idx, ci_pair_id, prompt_label, solution_model, solution_occurrence)

Outputs:
  - Prints agreement percent per file and overall
  - Writes logs to all_logs/compare_manual_to_basic_<timestamp>.log
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple


Key = Tuple[int, Optional[int], str, Optional[str], Optional[int]]


def _setup_file_logging(log_path: Optional[str] = None) -> str:
    if log_path is None:
        os.makedirs("all_logs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join("all_logs", f"compare_manual_to_basic_{ts}.log")
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


def _norm_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _make_key(r: Dict[str, Any]) -> Optional[Key]:
    idx = _norm_int(r.get("idx"))
    if idx is None:
        return None
    ci_pair_id = _norm_int(r.get("ci_pair_id"))
    prompt_label = (r.get("prompt_label") or "").strip().lower()
    if prompt_label not in ("good", "bad"):
        return None
    solution_model = r.get("solution_model")
    solution_occurrence = _norm_int(r.get("solution_occurrence"))
    return (idx, ci_pair_id, prompt_label, str(solution_model) if solution_model is not None else None, solution_occurrence)


def _agreement_stats(
    manual_rows: List[Dict[str, Any]],
    basic_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    basic_by_key: Dict[Key, str] = {}
    for r in basic_rows or []:
        k = _make_key(r)
        if k is None:
            continue
        ev = (r.get("evaluation") or "").strip().lower()
        if ev not in ("correct", "incorrect", "no evaluation"):
            continue
        # If duplicates exist, keep the last one (file order).
        basic_by_key[k] = ev

    compared = 0
    same = 0
    missing_in_basic = 0
    skipped_manual = 0
    disagreements: List[Dict[str, Any]] = []

    for r in manual_rows or []:
        k = _make_key(r)
        if k is None:
            continue
        mev = (r.get("evaluation") or "").strip().lower()
        if mev not in ("correct", "incorrect", "no evaluation"):
            skipped_manual += 1
            continue
        bev = basic_by_key.get(k)
        if bev is None:
            missing_in_basic += 1
            continue
        compared += 1
        if bev == mev:
            same += 1
        else:
            disagreements.append(
                {
                    "key": {
                        "idx": k[0],
                        "ci_pair_id": k[1],
                        "prompt_label": k[2],
                        "solution_model": k[3],
                        "solution_occurrence": k[4],
                    },
                    "manual": mev,
                    "basic": bev,
                }
            )

    pct = (same / compared * 100.0) if compared else 0.0
    return {
        "same": same,
        "compared": compared,
        "percentage": round(pct, 2),
        "missing_in_basic": missing_in_basic,
        "skipped_manual": skipped_manual,
        "disagreements": disagreements,
    }


def _list_manual_files(output_dir: str) -> List[str]:
    if not os.path.exists(output_dir):
        return []
    try:
        names = os.listdir(output_dir)
    except OSError:
        return []
    out: List[str] = []
    for n in names:
        if n.startswith("evaluate_results_manual_") and n.endswith(".jsonl"):
            out.append(os.path.join(output_dir, n))
    return sorted(out)


def _basic_path_for_manual(manual_path: str) -> str:
    base = os.path.basename(manual_path)
    # evaluate_results_manual_<suffix>.jsonl -> evaluate_results_<suffix>.jsonl
    suffix = base[len("evaluate_results_manual_") :] if base.startswith("evaluate_results_manual_") else base
    if suffix.endswith(".jsonl"):
        suffix = suffix[: -len(".jsonl")]
    return os.path.join(os.path.dirname(manual_path), f"evaluate_results_{suffix}.jsonl")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compare manual vs basic evaluate results (agreement percent).")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory containing outputs/*.jsonl")
    parser.add_argument("--manual-file", type=str, default=None, help="Compare only this manual file")
    parser.add_argument("--log-file", type=str, default=None, help="Write log to this file (default: all_logs/...)")
    parser.add_argument(
        "--show-disagreements",
        type=int,
        default=0,
        help="Print first N disagreements per file (default: 0)",
    )
    args = parser.parse_args()

    log_path = _setup_file_logging(args.log_file)
    print(f"Log file: {log_path}")
    logging.info("Log file: %s", log_path)

    output_dir = args.output_dir
    if args.manual_file:
        manual_files = [args.manual_file]
    else:
        manual_files = _list_manual_files(output_dir)

    if not manual_files:
        print(f"Error: no evaluate_results_manual_*.jsonl found in {output_dir}", file=sys.stderr)
        sys.exit(1)

    overall_same = 0
    overall_compared = 0
    overall_missing = 0

    for mf in manual_files:
        bf = _basic_path_for_manual(mf)
        mrows = _read_jsonl(mf)
        brows = _read_jsonl(bf)
        stats = _agreement_stats(mrows, brows)

        overall_same += int(stats["same"])
        overall_compared += int(stats["compared"])
        overall_missing += int(stats["missing_in_basic"])

        line = (
            f"{os.path.basename(mf)} vs {os.path.basename(bf)}: "
            f"{stats['percentage']}% same ({stats['same']}/{stats['compared']} compared), "
            f"missing_in_basic={stats['missing_in_basic']}"
        )
        print(line)
        logging.info(line)

        n = int(args.show_disagreements or 0)
        if n > 0 and stats["disagreements"]:
            for d in stats["disagreements"][:n]:
                print(json.dumps(d, ensure_ascii=False))

    overall_pct = (overall_same / overall_compared * 100.0) if overall_compared else 0.0
    summary = (
        f"OVERALL: {round(overall_pct, 2)}% same "
        f"({overall_same}/{overall_compared} compared), missing_in_basic={overall_missing}"
    )
    print(summary)
    logging.info(summary)


if __name__ == "__main__":
    main()

