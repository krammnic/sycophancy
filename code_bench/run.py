import json
import os
import re
import time
import logging
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from num2words import num2words
from tqdm import tqdm
from utils.call_api import call_api, APIError


def load_prompt_generate(prompt_file: str = "prompt.txt") -> str:
    """
    Load PROMPT_GENERATE from prompt.txt file.
    
    Args:
        prompt_file: Path to the prompt file.
        
    Returns:
        The PROMPT_GENERATE string.
    """
    try:
        with open(prompt_file, 'r') as f:
            content = f.read()
            # Extract PROMPT_GENERATE value (it's a Python assignment)
            for line in content.split('\n'):
                if line.startswith('PROMPT_GENERATE'):
                    # Extract the string value between quotes
                    start = line.find('"')
                    end = line.rfind('"')
                    if start != -1 and end != -1 and end > start:
                        return line[start+1:end] + '\n'
        # Fallback if not found
        return "Solve the following problem using only the information given. Return only code on C++.\n"
    except FileNotFoundError:
        print(f"Warning: {prompt_file} not found, using default prompt.")
        return "Solve the following problem using only the information given. Return only code on C++.\n"


def load_models_from_config(config_path: str):
    """
    Load models (with rating intervals), number of tasks, max_tokens, max_workers, 
    reasoning_effort, and clear_outputs from config_run.json.
    
    Args:
        config_path: Path to the config JSON file.
        
    Returns:
        Tuple of (dict of models with rating intervals, number of tasks to run, max_tokens, 
        max_workers, reasoning_effort, clear_outputs).
        
    Raises:
        ValueError: If config file is invalid or missing models.
    """
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    
    # Support both old format (list) and new format (dict with rating intervals)
    models_config = cfg.get("models")
    if not models_config:
        raise ValueError("config_run.json must have a 'models' key.")
    
    # Handle old format: ["model1", "model2"] -> {"model1": None, "model2": None}
    if isinstance(models_config, list):
        models = {model: None for model in models_config}
    elif isinstance(models_config, dict):
        models = models_config  # New format: {"model1": [min, max], "model2": None}
    else:
        raise ValueError("config_run.json 'models' must be either a list or a dict.")
    
    num_tasks = cfg.get("num_tasks", None)  # None means process all tasks
    max_tokens = cfg.get("max_tokens", 30000)  # Default to 30000 if not specified
    max_workers = cfg.get("max_workers", 5)  # Default to 5 workers if not specified
    reasoning_effort = cfg.get("reasoning_effort", None)  # None means no reasoning effort specified
    clear_outputs = cfg.get("clear_outputs", False)
    return models, num_tasks, max_tokens, max_workers, reasoning_effort, clear_outputs


def construct_text_condition(task, prompt_prefix: str = ""):
    """
    Constructs a text condition from a task dictionary.
    
    Args:
        task (dict): A dictionary containing task information with keys:
                     'title', 'description', 'input_format', 'output_format',
                     'interaction_format', 'examples', 'note'
        prompt_prefix: Optional prefix to add before the task description.
    
    Returns:
        str: The constructed text condition
    """
    # Start with prompt prefix if provided
    statement = prompt_prefix
    
    # Start with title (no label)
    statement += task.get('title', '') + '\n'
    
    # Add description with label
    description = task.get('description', '')
    if description:
        statement += 'Description:\n' + description + '\n'
    
    # Add input format with label
    input_format = task.get('input_format', '')
    if input_format:
        statement += 'Input Format:\n' + input_format + '\n'
    
    # Add output format with label
    output_format = task.get('output_format', '')
    if output_format:
        statement += 'Output Format:\n' + output_format + '\n'
    
    # Add interaction format with label
    interaction_format = task.get('interaction_format', '')
    if interaction_format:
        statement += 'Interaction Format:\n' + interaction_format + '\n'
    
    # Add examples if they exist
    examples = task.get('examples', [])
    if examples:
        statement += 'Examples:\n'
        for i, example in enumerate(examples, 1):
            ordinal = num2words(i, to="ordinal", lang="en")
            statement += (ordinal + ' example:' + '\n' + 
                          'Input:\n' + example.get('input', '') + '\n' + 
                          'Output:\n' + example.get('output', '') + '\n')
    
    # Add note at the end with label
    note = task.get('note', '')
    if note:
        statement += 'Note:\n' + note
    
    return statement


def extract_code_from_markdown(text: str) -> str:
    """
    Extract code from markdown code blocks.
    Looks for code blocks with ```cpp, ```c++, ```c, or just ```.
    
    Args:
        text: The text containing markdown code blocks.
        
    Returns:
        The extracted code, or empty string if no code block found.
    """
    # Pattern to match code blocks: ```cpp, ```c++, ```c, or just ```
    # Also handles cases with or without language specifier
    patterns = [
        r'```(?:cpp|c\+\+|c)?\s*\n(.*?)```',  # Matches ```cpp\n...``` or ```\n...```
        r'```(?:cpp|c\+\+|c)?\s*(.*?)```',    # Matches ```cpp...``` (no newline)
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            # Return the first (longest) match, stripped of whitespace
            code = matches[0].strip()
            if code:
                return code
    
    # If no code block found, return empty string
    return ""


def setup_logging(output_dir: str):
    """
    Set up comprehensive logging to both console and file.
    
    Args:
        output_dir: Directory where log file will be saved.
    """
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"run_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    return log_file


def clear_output_files(model_list, output_dir: str):
    """
    Delete existing results and error files for the given models.
    """
    removed_files = []
    for model in model_list:
        model_safe_name = model.replace('/', '_').replace('\\', '_')
        result_file = os.path.join(output_dir, f"results_{model_safe_name}.jsonl")
        error_file = os.path.join(output_dir, f"errors_{model_safe_name}.jsonl")
        for path in (result_file, error_file):
            if os.path.exists(path):
                try:
                    os.remove(path)
                    removed_files.append(path)
                except OSError as e:
                    logging.warning(f"Failed to remove {path}: {e}")
    if removed_files:
        logging.info(f"Removed old output files: {', '.join(removed_files)}")


def process_single_call(
    idx: int,
    example: dict,
    model: str,
    prompt_generate: str,
    max_tokens: int,
    output_dir: str,
    stats_lock: threading.Lock,
    successful_calls: list,
    failed_calls: list,
    model_stats: dict,
    total_examples: int,
    reasoning_effort: str = None
):
    """
    Process a single API call for a given example and model.
    This function is designed to be called concurrently.
    
    Args:
        idx: Index of the example in the dataset
        example: The dataset example
        model: Model name to use
        prompt_generate: The prompt prefix
        max_tokens: Maximum tokens for the API call
        output_dir: Directory to save results
        stats_lock: Thread lock for thread-safe counter updates
        successful_calls: List to track successful calls (thread-safe via lock)
        failed_calls: List to track failed calls (thread-safe via lock)
        model_stats: Dictionary to track per-model statistics
        total_examples: Total number of examples being processed
        reasoning_effort: Reasoning effort level (e.g., "high", "xhigh") for Thinking models
    """
    prompt = construct_text_condition(example, prompt_prefix=prompt_generate)
    task_title = example.get('title', f'Task {idx}')
    model_safe_name = model.replace('/', '_').replace('\\', '_')
    output_file = os.path.join(output_dir, f"results_{model_safe_name}.jsonl")
    
    try:
        logging.info(f"  [{idx+1}/{total_examples}] Calling {model} for task: {task_title[:50]}...")
        call_start_time = time.time()
        
        response = call_api(
            query=prompt,
            model=model,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort
        )
        
        with open(f"all_logs/all_logs_gen", "a", encoding="utf-8") as f:
            f.write(f"idx: {idx}, prompt: {prompt}, model: {model}, max_tokens: {max_tokens}, reasoning_effort: {reasoning_effort}, response: {response}\n")
        
        call_duration = time.time() - call_start_time
        
        # Extract the generated content from response
        generated_text = ""
        if "choices" in response and len(response["choices"]) > 0:
            generated_text = response["choices"][0].get("message", {}).get("content", "")
        
        # Extract code from markdown code blocks
        extracted_code = extract_code_from_markdown(generated_text)
        
        # Skip saving if no code was extracted
        if not extracted_code:
            logging.warning(f"  ⚠ No code extracted for {model} on task {idx+1}, skipping result")
            with stats_lock:
                failed_calls.append(1)
                model_stats[model]["errors"] += 1
            return False, "No code block found in response"
        
        # Extract token usage if available
        usage_info = response.get("usage", {})
        prompt_tokens = usage_info.get("prompt_tokens", 0)
        completion_tokens = usage_info.get("completion_tokens", 0)
        total_tokens = usage_info.get("total_tokens", 0)
        
        # Extract task rating from dataset if available (common fields: rating, difficulty, level)
        task_rating = None
        if "rating" in example:
            task_rating = example["rating"]
        elif "difficulty" in example:
            task_rating = example["difficulty"]
        elif "level" in example:
            task_rating = example["level"]
        
        # Save result with only extracted code (file writing is thread-safe with 'a' mode)
        record = {
            "idx": idx,
            "task_title": task_title,
            "task_rating": task_rating,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "code": extracted_code,
            "timestamp": datetime.now().isoformat(),
            "call_duration_seconds": round(call_duration, 2),
            "tokens": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        }
        
        # Use lock for file writing to ensure atomic writes
        with stats_lock:
            with open(output_file, "a", encoding="utf-8") as outf:
                outf.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            successful_calls.append(1)
            model_stats[model]["success"] += 1
        
        logging.info(f"  ✓ Success for {model} on task {idx+1} (took {call_duration:.2f}s, tokens: {total_tokens})")
        return True, None
        
    except APIError as e:
        with stats_lock:
            failed_calls.append(1)
            model_stats[model]["errors"] += 1
        
        logging.error(f"  ✗ API Error for {model} on task {idx+1}: {e}")
        
        # Log the error
        error_file = os.path.join(output_dir, f"errors_{model_safe_name}.jsonl")
        error_record = {
            "idx": idx,
            "task_title": task_title,
            "model": model,
            "error": str(e),
            "error_type": "APIError",
            "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "timestamp": datetime.now().isoformat()
        }
        
        with stats_lock:
            with open(error_file, "a", encoding="utf-8") as errf:
                errf.write(json.dumps(error_record, ensure_ascii=False) + "\n")
        
        return False, str(e)
        
    except Exception as e:
        with stats_lock:
            failed_calls.append(1)
            model_stats[model]["errors"] += 1
        
        logging.error(f"  ✗ Unexpected error for {model} on task {idx+1}: {e}", exc_info=True)
        
        # Log unexpected errors too
        error_file = os.path.join(output_dir, f"errors_{model_safe_name}.jsonl")
        error_record = {
            "idx": idx,
            "task_title": task_title,
            "model": model,
            "error": str(e),
            "error_type": type(e).__name__,
            "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "timestamp": datetime.now().isoformat()
        }
        
        with stats_lock:
            with open(error_file, "a", encoding="utf-8") as errf:
                errf.write(json.dumps(error_record, ensure_ascii=False) + "\n")
        
        return False, str(e)


def main():
    with open(f"all_logs/all_logs_gen", "a", encoding="utf-8") as f:
        f.write(f"NEW RUN")
    """Main function to run models on the codeforces dataset."""
    # Paths
    config_path = "config_run.json"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup comprehensive logging
    log_file = setup_logging(output_dir)
    logging.info(f"Starting run. Log file: {log_file}")
    logging.info("=" * 80)

    # Load prompt
    prompt_generate = load_prompt_generate("prompt.txt")
    logging.info(f"Loaded PROMPT_GENERATE: {prompt_generate.strip()}")

    # Load models and config
    logging.info(f"Loading models from {config_path}...")
    models_config, num_tasks, max_tokens, max_workers, reasoning_effort, clear_outputs = load_models_from_config(config_path)
    model_list = list(models_config.keys())  # Extract model names for processing
    logging.info(f"Loaded {len(model_list)} models: {model_list}")
    for model, rating_interval in models_config.items():
        if rating_interval:
            logging.info(f"  {model}: rating interval {rating_interval}")
        else:
            logging.info(f"  {model}: no rating filter")
    logging.info(f"Max tokens per request: {max_tokens}")
    logging.info(f"Max concurrent workers: {max_workers}")
    if reasoning_effort:
        logging.info(f"Reasoning effort: {reasoning_effort}")
    if clear_outputs:
        logging.info("Clearing previous results/errors for selected models.")
        clear_output_files(model_list, output_dir)
    if num_tasks is not None:
        logging.info(f"Will process {num_tasks} tasks.")
    else:
        logging.info("Will process all tasks in the dataset.")

    # Load dataset
    logging.info("Loading dataset open-r1/codeforces, split='test'...")
    try:
        dataset = load_dataset("open-r1/codeforces", split="test")
        logging.info(f"Dataset size: {len(dataset)} examples.")
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return

    # Limit number of tasks if specified
    if num_tasks is not None:
        dataset = dataset.select(range(min(num_tasks, len(dataset))))
        logging.info(f"Limited to {len(dataset)} tasks.")

    # Helper function to get task rating
    def get_task_rating(example):
        """Extract rating from example, checking common field names."""
        if "rating" in example:
            return example["rating"]
        elif "difficulty" in example:
            return example["difficulty"]
        elif "level" in example:
            return example["level"]
        return None

    # Helper function to check if task rating matches model's rating interval
    def matches_rating_interval(task_rating, model):
        """Check if task rating is within the model's rating interval."""
        rating_interval = models_config.get(model)
        if not rating_interval:
            return True  # No filter if no interval specified for this model
        
        if not isinstance(rating_interval, list) or len(rating_interval) != 2:
            return True  # Invalid interval, don't filter
        
        min_rating, max_rating = rating_interval[0], rating_interval[1]
        
        # If task has no rating, include it (or exclude it - you can change this behavior)
        if task_rating is None:
            return True  # Include tasks without rating
        
        return min_rating <= task_rating <= max_rating

    # Process each example with each model using ThreadPoolExecutor
    total_examples = len(dataset)
    successful_calls = []  # List to track successful calls (thread-safe)
    failed_calls = []  # List to track failed calls (thread-safe)
    start_time = time.time()
    
    # Statistics per model (accessed with lock)
    model_stats = {model: {"success": 0, "errors": 0} for model in model_list}
    stats_lock = threading.Lock()
    
    # Prepare all tasks, filtering by rating intervals
    tasks = []
    logging.info(f"Preparing tasks for {len(model_list)} models...")
    for idx, example in enumerate(dataset):
        task_rating = get_task_rating(example)
        for model in model_list:
            if matches_rating_interval(task_rating, model):
                tasks.append((idx, example, model))
            else:
                logging.debug(f"Skipping task {idx} (rating: {task_rating}) for model {model} (interval: {models_config.get(model)})")
    
    total_calls = len(tasks)
    logging.info("=" * 80)
    logging.info(f"Starting processing: {total_calls} total API calls after rating filtering")
    logging.info(f"Using ThreadPoolExecutor with {max_workers} workers")
    logging.info("=" * 80)
    logging.info(f"All {len(tasks)} tasks queued and ready for processing.")
    
    # Process tasks concurrently with progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(
                process_single_call,
                idx, example, model,
                prompt_generate, max_tokens, output_dir,
                stats_lock, successful_calls, failed_calls,
                model_stats, total_examples, reasoning_effort
            ): (idx, model) for idx, example, model in tasks
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=total_calls, desc="Processing API calls", unit="call", 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            for future in as_completed(future_to_task):
                idx, model = future_to_task[future]
                
                # Update progress bar
                with stats_lock:
                    current_successful = len(successful_calls)
                    current_failed = len(failed_calls)
                    pbar.set_postfix({
                        'success': current_successful,
                        'failed': current_failed,
                        'rate': f'{current_successful/(current_successful+current_failed)*100:.1f}%' if (current_successful+current_failed) > 0 else '0%'
                    })
                
                pbar.update(1)
                
                # Detailed logging every 50 completions
                completed_count = pbar.n
                if completed_count % 50 == 0:
                    elapsed_time = time.time() - start_time
                    avg_time_per_call = elapsed_time / completed_count if completed_count > 0 else 0
                    remaining_calls = total_calls - completed_count
                    estimated_remaining_time = avg_time_per_call * remaining_calls
                    
                    logging.info("=" * 80)
                    logging.info(f"Progress: {completed_count}/{total_calls} API calls completed")
                    logging.info(f"  Successful: {current_successful}, Failed: {current_failed}")
                    logging.info(f"  Elapsed time: {elapsed_time/60:.2f} minutes")
                    logging.info(f"  Estimated remaining time: {estimated_remaining_time/60:.2f} minutes")
                    logging.info("=" * 80)

    # Final summary
    total_time = time.time() - start_time
    final_successful = len(successful_calls)
    final_failed = len(failed_calls)
    
    logging.info("=" * 80)
    logging.info("RUN COMPLETED")
    logging.info("=" * 80)
    logging.info(f"Total tasks processed: {total_examples}")
    logging.info(f"Total API calls: {total_calls}")
    logging.info(f"  Successful: {final_successful}")
    logging.info(f"  Failed: {final_failed}")
    logging.info(f"Success rate: {(final_successful/total_calls*100):.2f}%")
    logging.info(f"Total time: {total_time/60:.2f} minutes ({total_time:.2f} seconds)")
    logging.info(f"Average time per call: {total_time/total_calls:.2f} seconds")
    logging.info(f"Concurrent speedup: {max_workers}x workers used")
    
    logging.info("\nModel Statistics:")
    for model, stats in model_stats.items():
        success_rate = (stats["success"] / (stats["success"] + stats["errors"]) * 100) if (stats["success"] + stats["errors"]) > 0 else 0
        logging.info(f"  {model}:")
        logging.info(f"    Success: {stats['success']}, Errors: {stats['errors']}, Success rate: {success_rate:.2f}%")
    
    logging.info(f"\nResults saved in {output_dir}/")
    logging.info(f"  - Results: results_{{model}}.jsonl (one file per model)")
    logging.info(f"  - Errors: errors_{{model}}.jsonl (one file per model)")
    logging.info(f"  - Log: {log_file}")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
