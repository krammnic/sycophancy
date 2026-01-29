#!/usr/bin/env python3
import ast
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from tqdm import tqdm

from utils.call_api import call_api, APIError


def load_prompt_evaluator(prompt_file: str = "prompt.txt") -> str:
    """
    Load PROMPT_EVALUATOR from prompt.txt.
    Returns the prompt string.
    """
    with open(prompt_file, "r", encoding="utf-8") as f:
        content = f.read()

    tree = ast.parse(content)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "PROMPT_EVALUATOR":
                    if isinstance(node.value, ast.Constant):
                        return node.value.value
                    elif isinstance(node.value, ast.Str):  # Python < 3.8
                        return node.value.s
    raise ValueError("PROMPT_EVALUATOR not found in prompt.txt")


def setup_logging(output_dir: str):
    """
    Set up logging to both console and file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"evaluate_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return log_file


def load_test_results(test_results_file: str) -> list:
    """
    Load test results from a JSONL file.
    Returns a list of records.
    """
    records = []
    if not os.path.exists(test_results_file):
        logging.warning(f"Test results file not found: {test_results_file}")
        return records

    with open(test_results_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse line in {test_results_file}: {e}")
                continue
    return records


def clear_output_files(test_result_files: list, output_dir: str):
    """
    Clear old evaluation output files.
    """
    removed = []
    for test_file in test_result_files:
        # Extract model name from test_results_{model}.jsonl or test_results_ci_{model}.jsonl
        basename = os.path.basename(test_file)
        model_name = None
        if basename.startswith("test_results_ci_") and basename.endswith(".jsonl"):
            model_name = basename[len("test_results_ci_"):-len(".jsonl")]
        elif basename.startswith("test_results_") and basename.endswith(".jsonl"):
            model_name = basename[len("test_results_"):-len(".jsonl")]
        if model_name is not None:
            eval_file = os.path.join(output_dir, f"evaluate_results_{model_name}.jsonl")
            if os.path.exists(eval_file):
                try:
                    os.remove(eval_file)
                    removed.append(eval_file)
                except OSError as e:
                    logging.warning(f"Failed to remove {eval_file}: {e}")
    if removed:
        logging.info(f"Removed old evaluation outputs: {', '.join(removed)}")


def process_evaluation_call(
    record: dict,
    evaluator_prompt: str,
    evaluator_model: str,
    max_tokens: int,
    reasoning_effort: str,
    output_file: str,
    stats_lock: threading.Lock,
    successful: list,
    failed: list,
):
    """
    Process a single evaluation call.
    """
    try:
        # Construct the full prompt: evaluator prompt + previous LLM text
        full_prompt = evaluator_prompt + record.get("response", "")
        
        call_start = time.time()
        response = call_api(
            query=full_prompt,
            model=evaluator_model,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        with open(f"all_logs/all_logs_evaluate", "a", encoding="utf-8") as f:
            f.write(f"record: {record}, prompt: {full_prompt}, model: {evaluator_model}, max_tokens: {max_tokens}, reasoning_effort: {reasoning_effort}, response: {response}\n")
        duration = time.time() - call_start

        # Extract evaluation result
        evaluation_result = ""
        if "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            if "message" in choice:
                evaluation_result = choice["message"].get("content", "").strip().lower()
            elif "text" in choice:
                evaluation_result = choice["text"].strip().lower()
            elif "content" in choice:
                evaluation_result = choice["content"].strip().lower()

        # Parse the result - should be "correct", "incorrect", or "no evaluation"
        result_lower = evaluation_result.lower()
        
        # Check for "no evaluation" first (two words)
        if "no evaluation" in result_lower or result_lower.startswith("no evaluation"):
            evaluation_result = "no evaluation"
        elif "incorrect" in result_lower:
            evaluation_result = "incorrect"
        elif "correct" in result_lower:
            evaluation_result = "correct"
        else:
            # Try to extract first word as fallback
            first_word = result_lower.split()[0] if result_lower else ""
            if first_word in ["correct", "incorrect"]:
                evaluation_result = first_word
            else:
                logging.warning(f"Unexpected evaluation result for idx={record.get('idx')}: '{evaluation_result}' (first word: '{first_word}')")
                evaluation_result = "no evaluation"  # Default fallback

        # Extract token usage if available
        usage_info = response.get("usage", {})
        prompt_tokens = usage_info.get("prompt_tokens", 0)
        completion_tokens = usage_info.get("completion_tokens", 0)
        total_tokens = usage_info.get("total_tokens", 0)

        eval_record = {
            "idx": record.get("idx"),
            "task_title": record.get("task_title"),
            "task_rating": record.get("task_rating"),
            # Propagate solution identity if present in test_results_*.jsonl
            "solution_model": record.get("solution_model"),
            "solution_occurrence": record.get("solution_occurrence"),
            # Propagate CI prompt-pair identity if present
            "ci_pair_id": record.get("ci_pair_id"),
            "prompt_label": record.get("prompt_label"),
            "judge_model": record.get("judge_model"),
            "evaluator_model": evaluator_model,
            "evaluation": evaluation_result,
            "timestamp": datetime.now().isoformat(),
            "call_duration_seconds": round(duration, 2),
            "tokens": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        }

        with stats_lock:
            with open(output_file, "a", encoding="utf-8") as outf:
                outf.write(json.dumps(eval_record, ensure_ascii=False) + "\n")
            successful.append(1)

        idx = record.get("idx", "?")
        logging.info(f"  ✓ Success for idx={idx} (took {duration:.2f}s, tokens: {total_tokens}, evaluation: {evaluation_result})")
        return True, None
    except APIError as e:
        with stats_lock:
            failed.append(1)
        
        idx = record.get("idx", "?")
        logging.error(f"  ✗ API Error for idx={idx}: {e}")
        
        # Log the error to a separate file
        evaluator_safe = evaluator_model.replace("/", "_").replace("\\", "_")
        output_dir = os.path.dirname(output_file) if output_file else "outputs"
        error_file = os.path.join(output_dir, f"errors_evaluate_{evaluator_safe}.jsonl")
        error_record = {
            "idx": record.get("idx"),
            "task_title": record.get("task_title"),
            "evaluator_model": evaluator_model,
            "error": str(e),
            "error_type": "APIError",
            "timestamp": datetime.now().isoformat()
        }
        
        with stats_lock:
            with open(error_file, "a", encoding="utf-8") as errf:
                errf.write(json.dumps(error_record, ensure_ascii=False) + "\n")
        
        return False, str(e)
    except Exception as e:
        with stats_lock:
            failed.append(1)
        
        idx = record.get("idx", "?")
        logging.error(f"  ✗ Unexpected error for idx={idx}: {e}", exc_info=True)
        
        # Log unexpected errors too
        evaluator_safe = evaluator_model.replace("/", "_").replace("\\", "_")
        output_dir = os.path.dirname(output_file) if output_file else "outputs"
        error_file = os.path.join(output_dir, f"errors_evaluate_{evaluator_safe}.jsonl")
        error_record = {
            "idx": record.get("idx"),
            "task_title": record.get("task_title"),
            "evaluator_model": evaluator_model,
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat()
        }
        
        with stats_lock:
            with open(error_file, "a", encoding="utf-8") as errf:
                errf.write(json.dumps(error_record, ensure_ascii=False) + "\n")
        
        return False, str(e)


def main():
    
    with open(f"all_logs/all_logs_evaluate", "a", encoding="utf-8") as f:
        f.write(f"NEW RUN")
    config_path = "config_evaluate.json"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    log_file = setup_logging(output_dir)
    logging.info(f"Starting evaluation run. Log file: {log_file}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    test_result_files = cfg.get("test_result_files", [])
    evaluator_model = cfg.get("evaluator_model")
    max_tokens = cfg.get("max_tokens", 2000)
    max_workers = cfg.get("max_workers", 5)
    reasoning_effort = cfg.get("reasoning_effort")
    clear_outputs = cfg.get("clear_outputs", False)

    if not test_result_files or not evaluator_model:
        raise ValueError("config_evaluate.json must include test_result_files and evaluator_model")

    evaluator_prompt = load_prompt_evaluator("prompt.txt")
    logging.info(f"Loaded PROMPT_EVALUATOR from prompt.txt")
    logging.info(f"Evaluator model: {evaluator_model}")
    logging.info(f"Max tokens: {max_tokens}")
    logging.info(f"Max workers: {max_workers}")
    if reasoning_effort:
        logging.info(f"Reasoning effort: {reasoning_effort}")

    if clear_outputs:
        clear_output_files(test_result_files, output_dir)

    # Load all test results
    all_records = []
    for test_file in test_result_files:
        test_file_path = os.path.join(output_dir, test_file) if not os.path.isabs(test_file) else test_file
        records = load_test_results(test_file_path)
        logging.info(f"Loaded {len(records)} records from {test_file}")
        all_records.extend(records)

    total_calls = len(all_records)
    logging.info(f"Total evaluations to perform: {total_calls}")

    if total_calls == 0:
        logging.warning("No records to evaluate. Exiting.")
        return

    # Group records by judge_model to determine output file names
    records_by_model = {}
    for record in all_records:
        judge_model = record.get("judge_model", "unknown")
        model_safe = judge_model.replace("/", "_").replace("\\", "_")
        if model_safe not in records_by_model:
            records_by_model[model_safe] = []
        records_by_model[model_safe].append(record)

    # Prepare tasks
    tasks = []
    for model_safe, records in records_by_model.items():
        output_file = os.path.join(output_dir, f"evaluate_results_{model_safe}.jsonl")
        for record in records:
            tasks.append((record, output_file))

    # Statistics
    stats_lock = threading.Lock()
    successful = []
    failed = []
    model_stats = {evaluator_model: {"successful": 0, "failed": 0}}

    start_time = time.time()

    # Process with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for record, output_file in tasks:
            future = executor.submit(
                process_evaluation_call,
                record,
                evaluator_prompt,
                evaluator_model,
                max_tokens,
                reasoning_effort,
                output_file,
                stats_lock,
                successful,
                failed,
            )
            futures.append(future)

        # Progress bar with postfix
        with tqdm(
            total=total_calls,
            desc="Evaluating",
            unit="call",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ) as pbar:
            for future in as_completed(futures):
                try:
                    success, error = future.result()
                    if success:
                        with stats_lock:
                            model_stats[evaluator_model]["successful"] += 1
                    else:
                        with stats_lock:
                            model_stats[evaluator_model]["failed"] += 1
                except Exception as e:
                    logging.error(f"Future exception: {e}", exc_info=True)
                    with stats_lock:
                        failed.append(1)
                        model_stats[evaluator_model]["failed"] += 1
                pbar.update(1)
                with stats_lock:
                    pbar.set_postfix(
                        {
                            "success": len(successful),
                            "failed": len(failed),
                        }
                    )

    end_time = time.time()
    total_time = (end_time - start_time) / 60

    # Final statistics
    logging.info("Evaluation completed.")
    logging.info(f"Total calls: {total_calls}")
    logging.info(f"Successful: {len(successful)}")
    logging.info(f"Failed: {len(failed)}")
    logging.info(f"Total time: {total_time:.2f} minutes")

    for model, stats in model_stats.items():
        logging.info(f"  {model}: {stats['successful']} successful, {stats['failed']} failed")


if __name__ == "__main__":
    main()
