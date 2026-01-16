#!/usr/bin/env python3
import ast
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from datasets import load_dataset
from num2words import num2words
from tqdm import tqdm

from utils.call_api import call_api, APIError


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


def construct_text_condition(task, prompt_prefix: str = ""):
    """
    Constructs a text condition from a task dictionary.
    """
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
    Load results_{model}.jsonl into a dict keyed by idx.
    """
    model_safe = model.replace("/", "_").replace("\\", "_")
    path = os.path.join(output_dir, f"results_{model_safe}.jsonl")
    if not os.path.exists(path):
        return {}

    results = {}
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
                results[idx] = record
    return results


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


def clear_output_files(solution_models, output_dir: str):
    removed = []
    for model in solution_models:
        model_safe = model.replace("/", "_").replace("\\", "_")
        out_file = os.path.join(output_dir, f"test_results_{model_safe}.jsonl")
        if os.path.exists(out_file):
            try:
                os.remove(out_file)
                removed.append(out_file)
            except OSError as e:
                logging.warning(f"Failed to remove {out_file}: {e}")
    if removed:
        logging.info(f"Removed old test outputs: {', '.join(removed)}")


def process_eval_call(
    idx: int,
    task_title: str,
    task_rating,
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
        eval_prompt = build_eval_prompt(prompt_text, problem_text, solution_code)
        call_start = time.time()
        response = call_api(
            query=eval_prompt,
            model=judge_model,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        duration = time.time() - call_start

        # Extract answer text from response
        answer_text = ""
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
    except APIError as e:
        with stats_lock:
            failed.append(1)
        logging.error(f"API error for idx={idx}: {e}")
        return False, str(e)
    except Exception as e:
        with stats_lock:
            failed.append(1)
        logging.error(f"Unexpected error for idx={idx}: {e}", exc_info=True)
        return False, str(e)


def main():
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
    max_tokens = cfg.get("max_tokens", 2000)
    max_workers = cfg.get("max_workers", 5)
    reasoning_effort = cfg.get("reasoning_effort")
    prompt_labels = cfg.get("prompt_labels")
    clear_outputs = cfg.get("clear_outputs", False)

    if not solution_models or not judge_model:
        raise ValueError("config_test.json must include solution_models and judge_model")

    prompts = load_prompts_evaluate("prompt.txt")
    if prompt_labels:
        prompts = {k: v for k, v in prompts.items() if v in set(prompt_labels)}

    if clear_outputs:
        clear_output_files(solution_models, output_dir)

    logging.info(f"Solution models: {solution_models}")
    logging.info(f"Judge model: {judge_model}")
    logging.info(f"Max tokens: {max_tokens}")
    logging.info(f"Max workers: {max_workers}")
    if reasoning_effort:
        logging.info(f"Reasoning effort: {reasoning_effort}")

    dataset = load_dataset("open-r1/codeforces", split="test")
    if num_tasks is not None:
        dataset = dataset.select(range(min(num_tasks, len(dataset))))
    total_examples = len(dataset)
    logging.info(f"Dataset size: {total_examples}")

    # Load results per solution model
    solutions_by_model = {m: load_results_for_model(m, output_dir) for m in solution_models}

    tasks = []
    for idx, example in enumerate(dataset):
        task_title = example.get("title", f"Task {idx}")
        task_rating = example.get("rating") or example.get("difficulty") or example.get("level")
        problem_text = construct_text_condition(example)

        for sol_model in solution_models:
            solution_record = solutions_by_model.get(sol_model, {}).get(idx)
            if not solution_record:
                continue
            solution_code = solution_record.get("code")
            if not solution_code:
                continue

            model_safe = sol_model.replace("/", "_").replace("\\", "_")
            output_file = os.path.join(output_dir, f"test_results_{model_safe}.jsonl")

            for prompt_text, prompt_label in prompts.items():
                tasks.append(
                    (
                        idx,
                        task_title,
                        task_rating,
                        prompt_label,
                        prompt_text,
                        problem_text,
                        solution_code,
                        output_file,
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
