#!/usr/bin/env python3
import ast
import json
import logging
import os
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime


def load_prompts_evaluate(prompt_file: str = "prompt.txt") -> dict:
    """
    Load PROMPTS_EVALUATE from prompt.txt.
    Returns a dict mapping prompt text -> label (e.g., "good", "bad").
    """
    with open(prompt_file, "r", encoding="utf-8") as f:
        content = f.read()

    tree = ast.parse(content)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "PROMPTS_EVALUATE":
                    return ast.literal_eval(node.value)
    raise ValueError("PROMPTS_EVALUATE not found in prompt.txt")


def load_prompt_pairs_ci(prompt_file: str = "prompt.txt") -> list:
    """
    Load PROMPT_PAIRS_CI from prompt.txt.
    Expected format: list of dicts like {"good": "...", "bad": "..."}.
    """
    with open(prompt_file, "r", encoding="utf-8") as f:
        content = f.read()

    tree = ast.parse(content)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "PROMPT_PAIRS_CI":
                    pairs = ast.literal_eval(node.value)
                    if not isinstance(pairs, list):
                        raise ValueError("PROMPT_PAIRS_CI must be a list")
                    # Light validation
                    for i, p in enumerate(pairs):
                        if not isinstance(p, dict) or "good" not in p or "bad" not in p:
                            raise ValueError(f"PROMPT_PAIRS_CI[{i}] must be a dict with 'good' and 'bad'")
                    return pairs
    raise ValueError("PROMPT_PAIRS_CI not found in prompt.txt")


def construct_text_condition(task, prompt_prefix: str = ""):
    """
    Constructs a text condition from a task dictionary.
    """
    # Lazy import so unit tests can import this module without optional deps.
    from num2words import num2words

    statement = prompt_prefix
    statement += task.get("title", "") + "\n"

    description = task.get("description", "")
    if description:
        statement += "Description:\n" + description + "\n"

    input_format = task.get("input_format", "")
    if input_format:
        statement += "Input Format:\n" + input_format + "\n"

    output_format = task.get("output_format", "")
    if output_format:
        statement += "Output Format:\n" + output_format + "\n"

    interaction_format = task.get("interaction_format", "")
    if interaction_format:
        statement += "Interaction Format:\n" + interaction_format + "\n"

    examples = task.get("examples", [])
    if examples:
        statement += "Examples:\n"
        for i, example in enumerate(examples, 1):
            ordinal = num2words(i, to="ordinal", lang="en")
            statement += (
                ordinal
                + " example:\n"
                + "Input:\n"
                + example.get("input", "")
                + "\n"
                + "Output:\n"
                + example.get("output", "")
                + "\n"
            )

    note = task.get("note", "")
    if note:
        statement += "Note:\n" + note

    return statement


def setup_logging(output_dir: str):
    """
    Set up logging to both console and file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"test_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return log_file


def load_results_for_model(model: str, output_dir: str = "outputs") -> dict:
    """
    Load results_{model}.jsonl into a mapping of idx -> list of records (in file order).

    Important: we do NOT deduplicate by idx. If the file contains multiple records for the
    same idx (reruns), all are preserved as separate occurrences.
    """
    model_safe = model.replace("/", "_").replace("\\", "_")
    path = os.path.join(output_dir, f"results_{model_safe}.jsonl")
    if not os.path.exists(path):
        return {}

    results = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = record.get("idx")
            code = record.get("code")
            if idx is not None and code:
                results[idx].append(record)
    return dict(results)
def build_eval_prompt(prompt_text: str, problem_text: str, solution_code: str) -> str:
    """
    Build evaluation prompt using the template in PROMPTS_EVALUATE.
    """
    return (
        prompt_text
        + "\n\nProblem:\n"
        + problem_text
        + "\n\nStudent solution:\n"
        + solution_code
    )


def clear_output_files(judge_model: str, output_dir: str):
    removed = []
    judge_safe = judge_model.replace("/", "_").replace("\\", "_")
    out_file = os.path.join(output_dir, f"test_results_{judge_safe}.jsonl")
    if os.path.exists(out_file):
        try:
            os.remove(out_file)
            removed.append(out_file)
        except OSError as e:
            logging.warning(f"Failed to remove {out_file}: {e}")
    out_file_ci = os.path.join(output_dir, f"test_results_ci_{judge_safe}.jsonl")
    if os.path.exists(out_file_ci):
        try:
            os.remove(out_file_ci)
            removed.append(out_file_ci)
        except OSError as e:
            logging.warning(f"Failed to remove {out_file_ci}: {e}")
    if removed:
        logging.info(f"Removed old test outputs: {', '.join(removed)}")


def test_results_filename(judge_model: str, output_dir: str, ci_mode: bool) -> str:
    judge_safe = judge_model.replace("/", "_").replace("\\", "_")
    prefix = "test_results_ci_" if ci_mode else "test_results_"
    return os.path.join(output_dir, f"{prefix}{judge_safe}.jsonl")


def build_solution_pool(solutions_by_model: dict) -> list[tuple[int, str, int]]:
    """
    Build a pool of available solutions to sample from.
    Each element is (idx, solution_model, solution_occurrence).
    Only includes entries that have a non-empty 'code' field.
    """
    pool: list[tuple[int, str, int]] = []
    for sol_model, by_idx in (solutions_by_model or {}).items():
        if not isinstance(by_idx, dict):
            continue
        for idx, records in by_idx.items():
            if records is None:
                continue
            try:
                idx_int = int(idx)
            except (TypeError, ValueError):
                continue
            for occ_i, rec in enumerate(records):
                if not isinstance(rec, dict):
                    continue
                if rec.get("code"):
                    pool.append((idx_int, str(sol_model), int(occ_i)))
    return pool


def process_eval_call(
    idx: int,
    task_title: str,
    task_rating,
    solution_model: str,
    solution_occurrence: int,
    ci_pair_id,
    prompt_label: str,
    prompt_text: str,
    problem_text: str,
    solution_code: str,
    judge_model: str,
    max_tokens: int,
    reasoning_effort: str,
    output_file: str,
    stats_lock: threading.Lock,
    successful: list,
    failed: list,
):
    try:
        # Lazy import so unit tests can import this module without optional deps.
        from utils.call_api import call_api, APIError

        eval_prompt = build_eval_prompt(prompt_text, problem_text, solution_code)
        call_start = time.time()
        response = call_api(
            query=eval_prompt,
            model=judge_model,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        with open(f"all_logs/all_logs_test", "a", encoding="utf-8") as f:
            f.write(f"idx: {idx}, prompt_label: {prompt_label}, solution_model: {solution_model}, testing_model: {judge_model}, max_tokens: {max_tokens}, reasoning_effort: {reasoning_effort}, ci_pair_id: {ci_pair_id}, prompt: {eval_prompt}, response: {response}\n")
        duration = time.time() - call_start

        # Extract answer text from response
        answer_text = ""
        
        if isinstance(response, dict) and "content" in response:
            blocks = response.get("content")
            if isinstance(blocks, list) and blocks:
                parts = []
                for b in blocks:
                    if isinstance(b, dict) and b.get("type") == "text":
                        parts.append(b.get("text", ""))
                answer_text = "".join(parts).strip()
        
        if "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            # Try different possible locations for the content
            if "message" in choice:
                answer_text = choice["message"].get("content", "")
            elif "text" in choice:
                answer_text = choice["text"]
            elif "content" in choice:
                answer_text = choice["content"]
        
        # Log warning if response is empty but API call succeeded
        if not answer_text:
            logging.warning(f"Empty response for idx={idx}, prompt_label={prompt_label}. Response keys: {list(response.keys()) if isinstance(response, dict) else 'not a dict'}")
            if "choices" in response and len(response["choices"]) > 0:
                logging.warning(f"First choice keys: {list(response['choices'][0].keys())}")

        record = {
            "idx": idx,
            "task_title": task_title,
            "task_rating": task_rating,
            "solution_model": solution_model,
            "solution_occurrence": solution_occurrence,
            "ci_pair_id": ci_pair_id,
            "prompt_label": prompt_label,
            "prompt_text": prompt_text,
            "judge_model": judge_model,
            "response": answer_text,
            "timestamp": datetime.now().isoformat(),
            "call_duration_seconds": round(duration, 2),
        }

        with stats_lock:
            with open(output_file, "a", encoding="utf-8") as outf:
                outf.write(json.dumps(record, ensure_ascii=False) + "\n")
            successful.append(1)
        return True, None
    except Exception as e:
        # Import APIError lazily; if missing, treat as generic.
        try:
            from utils.call_api import APIError  # type: ignore
        except Exception:  # pragma: no cover
            APIError = ()  # type: ignore

        if isinstance(e, APIError):
            with stats_lock:
                failed.append(1)
            logging.error(f"API error for idx={idx}: {e}")
            return False, str(e)

        with stats_lock:
            failed.append(1)
        logging.error(f"Unexpected error for idx={idx}: {e}", exc_info=True)
        return False, str(e)


def main():
    
    with open(f"all_logs/all_logs_test", "a", encoding="utf-8") as f:
        f.write(f"NEW RUN")
    config_path = "config_test.json"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    log_file = setup_logging(output_dir)
    logging.info(f"Starting test run. Log file: {log_file}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    solution_models = cfg.get("solution_models", [])
    judge_model = cfg.get("judge_model")
    num_tasks = cfg.get("num_tasks")
    ci_num_solutions_per_pair = cfg.get("ci_num_solutions_per_pair", 0)
    ci_num_pairs = cfg.get("ci_num_pairs", None)
    max_tokens = cfg.get("max_tokens", 2000)
    max_workers = cfg.get("max_workers", 5)
    reasoning_effort = cfg.get("reasoning_effort")
    prompt_labels = cfg.get("prompt_labels")
    clear_outputs = cfg.get("clear_outputs", False)
    ci_seed = cfg.get("ci_seed", None)

    if not solution_models or not judge_model:
        raise ValueError("config_test.json must include solution_models and judge_model")

    prompts = load_prompts_evaluate("prompt.txt")
    if prompt_labels:
        prompts = {k: v for k, v in prompts.items() if v in set(prompt_labels)}

    ci_pairs = load_prompt_pairs_ci("prompt.txt")
    if ci_num_pairs is not None:
        try:
            ci_num_pairs = int(ci_num_pairs)
        except (TypeError, ValueError):
            raise ValueError("ci_num_pairs must be an integer or null")
        if ci_num_pairs < 0:
            raise ValueError("ci_num_pairs must be >= 0")
        ci_pairs = ci_pairs[:ci_num_pairs]

    try:
        ci_num_solutions_per_pair = int(ci_num_solutions_per_pair) if ci_num_solutions_per_pair is not None else 0
    except (TypeError, ValueError):
        raise ValueError("ci_num_solutions_per_pair must be an integer")
    if ci_num_solutions_per_pair < 0:
        raise ValueError("ci_num_solutions_per_pair must be >= 0")

    if clear_outputs:
        clear_output_files(judge_model, output_dir)

    logging.info(f"Solution models: {solution_models}")
    logging.info(f"Judge model: {judge_model}")
    logging.info(f"Max tokens: {max_tokens}")
    logging.info(f"Max workers: {max_workers}")
    if reasoning_effort:
        logging.info(f"Reasoning effort: {reasoning_effort}")
    logging.info(f"CI mode will run: {len(ci_pairs)} prompt pairs, {ci_num_solutions_per_pair} solutions per pair (seed={ci_seed})")

    # Lazy imports so unit tests can run without these deps installed.
    from datasets import load_dataset
    from tqdm import tqdm

    dataset = load_dataset("open-r1/codeforces", split="test")
    dataset_size = len(dataset)
    logging.info(f"Dataset size: {dataset_size}")

    # Load results per solution model
    solutions_by_model = {m: load_results_for_model(m, output_dir) for m in solution_models}

    tasks = []
    # Write BOTH the full run and CI run into the same file.
    # CI records are still distinguishable via ci_pair_id != None.
    output_file_full = test_results_filename(judge_model, output_dir, ci_mode=False)
    output_file_ci = output_file_full

    # --- Full/normal run (PROMPTS_EVALUATE) ---
    full_solution_pool: list[tuple[int, str, int]] = build_solution_pool(solutions_by_model)
    logging.info(
        f"Full-run sampling pool (solutions across {len(solution_models)} solution_models): "
        f"{len(full_solution_pool)} solutions (dataset_size={dataset_size})"
    )
    if num_tasks is None:
        # Backwards compatible: if user wants "all", evaluate all available solutions.
        sampled_full = list(full_solution_pool)
    else:
        try:
            full_k = int(num_tasks)
        except (TypeError, ValueError):
            raise ValueError("num_tasks must be an integer or null")
        if full_k < 0:
            raise ValueError("num_tasks must be >= 0 or null")
        full_k = min(full_k, len(full_solution_pool))
        if ci_seed is None:
            rng_full = random.Random()
        else:
            rng_full = random.Random(int(ci_seed) + 99991)
        sampled_full = rng_full.sample(full_solution_pool, k=full_k) if full_k > 0 else []
    logging.info(f"Full run will evaluate: {len(sampled_full)} solution instances")

    for (idx, sol_model, occ_i) in sampled_full:
        example = dataset[int(idx)]
        task_title = example.get("title", f"Task {idx}")
        task_rating = example.get("rating") or example.get("difficulty") or example.get("level")
        problem_text = construct_text_condition(example)

        solution_records = solutions_by_model.get(sol_model, {}).get(idx) or []
        if occ_i < 0 or occ_i >= len(solution_records):
            continue
        solution_record = solution_records[occ_i]
        solution_code = (solution_record or {}).get("code") if isinstance(solution_record, dict) else None
        if not solution_code:
            continue

        for prompt_text, prompt_label in prompts.items():
            tasks.append(
                (
                    idx,
                    task_title,
                    task_rating,
                    sol_model,
                    occ_i,
                    None,  # ci_pair_id
                    prompt_label,
                    prompt_text,
                    problem_text,
                    solution_code,
                    output_file_full,
                )
            )

    # --- CI run (PROMPT_PAIRS_CI) ---
    eligible_solution_refs: list[tuple[int, str, int]] = build_solution_pool(solutions_by_model)
    logging.info(
        f"CI sampling pool (solutions across {len(solution_models)} solution_models): "
        f"{len(eligible_solution_refs)} solutions (dataset_size={dataset_size})"
    )

    def _solution_refs_for_ci_pair(pair_id: int) -> list[tuple[int, str, int]]:
        pool = eligible_solution_refs
        if not pool or ci_num_solutions_per_pair <= 0:
            return []
        k = min(int(ci_num_solutions_per_pair), len(pool))
        if ci_seed is None:
            rng = random.Random()  # nondeterministic
        else:
            rng = random.Random(int(ci_seed) + int(pair_id) * 1000003)
        return rng.sample(pool, k=k)

    for pair_id, pair in enumerate(ci_pairs):
        sampled = _solution_refs_for_ci_pair(pair_id)
        if not sampled:
            continue
        for (idx, sol_model, occ_i) in sampled:
            example = dataset[int(idx)]
            task_title = example.get("title", f"Task {idx}")
            task_rating = example.get("rating") or example.get("difficulty") or example.get("level")
            problem_text = construct_text_condition(example)
            solution_records = solutions_by_model.get(sol_model, {}).get(idx) or []
            if occ_i < 0 or occ_i >= len(solution_records):
                continue
            solution_record = solution_records[occ_i]
            solution_code = (solution_record or {}).get("code") if isinstance(solution_record, dict) else None
            if not solution_code:
                continue
            # Always run both prompts for the pair
            tasks.append(
                (
                    idx,
                    task_title,
                    task_rating,
                    sol_model,
                    occ_i,
                    pair_id,
                    "good",
                    pair["good"],
                    problem_text,
                    solution_code,
                    output_file_ci,
                )
            )
            tasks.append(
                (
                    idx,
                    task_title,
                    task_rating,
                    sol_model,
                    occ_i,
                    pair_id,
                    "bad",
                    pair["bad"],
                    problem_text,
                    solution_code,
                    output_file_ci,
                )
            )

    total_calls = len(tasks)
    if total_calls == 0:
        logging.warning("No evaluation tasks found (missing solution code?).")
        return
    successful = []
    failed = []
    stats_lock = threading.Lock()
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(
                process_eval_call,
                idx,
                task_title,
                task_rating,
                solution_model,
                solution_occurrence,
                ci_pair_id,
                prompt_label,
                prompt_text,
                problem_text,
                solution_code,
                judge_model,
                max_tokens,
                reasoning_effort,
                output_file,
                stats_lock,
                successful,
                failed,
            ): idx
            for (
                idx,
                task_title,
                task_rating,
                solution_model,
                solution_occurrence,
                ci_pair_id,
                prompt_label,
                prompt_text,
                problem_text,
                solution_code,
                output_file,
            ) in tasks
        }

        with tqdm(
            total=total_calls,
            desc="Evaluating solutions",
            unit="call",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ) as pbar:
            for future in as_completed(future_to_task):
                pbar.update(1)
                with stats_lock:
                    pbar.set_postfix(
                        {
                            "success": len(successful),
                            "failed": len(failed),
                        }
                    )

    total_time = time.time() - start_time
    logging.info("Evaluation completed.")
    logging.info(f"Total calls: {total_calls}")
    logging.info(f"Successful: {len(successful)}")
    logging.info(f"Failed: {len(failed)}")
    logging.info(f"Total time: {total_time/60:.2f} minutes")


if __name__ == "__main__":
    main()