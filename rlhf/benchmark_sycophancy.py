from __future__ import annotations
import pandas as pd
from pprint import pprint
import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.exceptions import (
    ChunkedEncodingError,
    ReadTimeout,
    ConnectionError,
    HTTPError,
)
import re
import matplotlib.pyplot as plt
import json
import random
from tqdm import tqdm
import time
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark LLM models')
    parser.add_argument('--models', nargs='+', help='List of models to benchmark')
    parser.add_argument('--folder', type=str, default="benchmark_results", help='Folder to save results')
    args = parser.parse_args()

    workers = 32

    data=pd.read_csv('data.csv')

    n = data.shape[0]

    if args.models:
        models = args.models
    else:
        raise ValueError("No models specified. Use --models argument")

    with open("api.json", "r", encoding="utf-8") as f:
        api = json.load(f)

    with open("prompts.json", "r", encoding="utf-8") as f:
        prompts = json.load(f)
        val_prompts = prompts["val_prompts"]

    folder = args.folder

    os.makedirs(folder, exist_ok=True)
    
    url = ""
    from typing import Dict, Tuple
    import time
    import concurrent.futures

    from vllm import LLM, SamplingParams

    _LLM_CACHE: Dict[str, LLM] = {}

    def _get_llm(model: str, tp: int = 4) -> LLM:
        key = (model, tp)
        llm = _LLM_CACHE.get(key)
        if llm is None:
            llm = LLM(
                model=model,
                tensor_parallel_size=tp,
            )
            _LLM_CACHE[key] = llm
        return llm


    llm = _get_llm("qwen3_dpo_output/epoch_0/", tp = 4)

    sampling_params = SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1.0,
            max_tokens=8048,
    )

    def call_api(
        query: str,
        model: str,
        max_tokens: int = 100000,
        max_retries: int = 5,
        timeout: int = 6000,
    ) -> Tuple[bool, str, dict]:
        """
        vLLM inference wrapper.

        Returns:
        (True, output_text, row_dict)

        - First bool is always True (as requested).
        - output_text is the generated string.
        - row_dict is a dict version of the vLLM result "row" (best-effort).
        """

        last_exc: Exception | None = None

        def _run_generate():
            return llm.generate([query], sampling_params)

        for attempt in range(1, max_retries + 1):
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(_run_generate)
                    outputs = fut.result(timeout=timeout)

                req = outputs[0]

                comp = req.outputs[0]
                output_text = comp.text
                row = {
                    "model": model,
                    "prompt": getattr(req, "prompt", query),
                    "prompt_token_ids": getattr(req, "prompt_token_ids", None),
                    "request_id": getattr(req, "request_id", None),
                    "finished": getattr(req, "finished", None),
                    "text": comp.text,
                    "token_ids": getattr(comp, "token_ids", None),
                    "cumulative_logprob": getattr(comp, "cumulative_logprob", None),
                    "finish_reason": getattr(comp, "finish_reason", None),
                    "stop_reason": getattr(comp, "stop_reason", None),
                }

                ptoks = row["prompt_token_ids"]
                ctoks = row["token_ids"]
                if isinstance(ptoks, list) and isinstance(ctoks, list):
                    row["usage"] = {
                        "prompt_tokens": len(ptoks),
                        "completion_tokens": len(ctoks),
                        "total_tokens": len(ptoks) + len(ctoks),
                    }

                return True, output_text, row

            except concurrent.futures.TimeoutError as e:
                last_exc = e
                time.sleep(min(2.0**(attempt - 1), 8.0))
            except Exception as e:
                last_exc = e
                time.sleep(min(2.0**(attempt - 1), 8.0))

        return False, "", {
            "model": model,
            "prompt": query,
            "error": f"{type(last_exc).__name__}: {last_exc}" if last_exc else "Unknown error",
            "max_retries": max_retries,
            "timeout": timeout,
        }


    def _log_and_wait(exception, attempt, max_retries, query, model):
        delay = 2 ** attempt
        with open(f"{folder}/logs", "a", encoding="utf-8") as f:
            f.write(
                f"\n[WARN] {type(exception).__name__}: {exception}\n"
                f"query: {query[:100]}\n"
                f"model: {model}\n"
                f"Attempt {attempt + 1}/{max_retries}, repeat after {delay} seconds.\n"
            )
        time.sleep(delay)


    def _log_http_error_and_wait(e, attempt, max_retries, query, model):
        delay = 2 ** attempt
        status = getattr(e.response, 'status_code', None)
        text = getattr(e.response, 'text', '')[:300]
        with open(f"{folder}/logs", "a", encoding="utf-8") as f:
            f.write(
                f"\n[ERROR] HTTP error: {e}\n"
                f"query: {query[:100]}\n"
                f"model: {model}\n"
                f"(status={status}, text={text!r})\n"
                f"Attempt {attempt + 1}/{max_retries}, repeat after {delay} seconds."
            )
        time.sleep(delay)

    def calculate_scores(row, model):
        response_neutral = call_api(f"{val_prompts[0]}\n\nproblem:\n{row['question']}\n\nsolution:\n{row['solution_attempt']}", model)
        response_negative = call_api(f"{val_prompts[1]}\n\nproblem:\n{row['question']}\n\nsolution:\n{row['solution_attempt']}", model)

        if response_neutral[0] == False or response_negative[0] == False:
            row['score_neutral'] = None
            row['score_negative'] = None
            row['json_resp_neutral'] = None
            row['json_resp_negative'] = None
            return row

        pattern = re.compile(r'\\boxed\{\s*([-+]?\d+)')
        
        try:
            grade_neutral = int(pattern.findall(response_neutral[1])[-1])
        except (IndexError, ValueError):
            grade_neutral = None
            
        try:
            grade_negative = int(pattern.findall(response_negative[1])[-1])
        except (IndexError, ValueError):
            grade_negative = None

        row['score_neutral'] = grade_neutral
        row['score_negative'] = grade_negative
        row['json_resp_neutral'] = json.dumps(response_neutral[2], ensure_ascii=False)
        row['json_resp_negative'] = json.dumps(response_negative[2], ensure_ascii=False)

        return row

    def extract_usage(resp_json, model_name):
        ANTHROPIC_PRICES = {
            "anthropic/claude-sonnet-4-5-20250929": {
                "input": 3.0 / 1_000_000,
                "output": 15.0 / 1_000_000,
            }
        }

        usage = resp_json.get("usage")
        if not usage:
            return 0, 0.0

        if "total_tokens" in usage:
            tokens = usage.get("total_tokens", 0)
            cost = usage.get("cost", 0.0)
            return tokens, cost

        if "input_tokens" in usage and "output_tokens" in usage:
            input_tokens = usage["input_tokens"]
            output_tokens = usage["output_tokens"]
            tokens = input_tokens + output_tokens

            price = ANTHROPIC_PRICES.get(model_name)
            if not price:
                return tokens, 0.0

            cost = (
                input_tokens * price["input"] +
                output_tokens * price["output"]
            )
            return tokens, cost

        return 0, 0.0

    model_stats = {}

    summary_stats = {}

    print("=" * 80)
    print("STARTING BENCHMARK PROCESS")
    print("=" * 80)

    print("=" * 80, file=open(f"{folder}/logs", "w", encoding="utf-8"))
    print("STARTING BENCHMARK PROCESS", file=open(f"{folder}/logs", "a", encoding="utf-8"))
    print("=" * 80, file=open(f"{folder}/logs", "a", encoding="utf-8"))

    for model in models:
        print(f"\nProcessing model: {model}")
        print("-" * 50)
        
        print(f"\nProcessing model: {model}", file=open(f"{folder}/logs", "a", encoding="utf-8"))
        print("-" * 50, file=open(f"{folder}/logs", "a", encoding="utf-8"))
        
        results_list = [None] * len(data)
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {executor.submit(calculate_scores, row, model): i for i, (_, row) in enumerate(data.iterrows())}
            
            for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc=f"Processing {model}"):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results_list[index] = result
                except Exception as e:
                    print(f"Error processing row at index {index}: {e}")
                    original_row = data.iloc[index].copy()
                    original_row['score_neutral'] = None
                    original_row['score_negative'] = None
                    results_list[index] = original_row
        
        data_model = pd.DataFrame(results_list)

        data_model.to_csv(f"{folder}/{model.replace('/', '_')}.csv", index=False)
        print(f"Results saved to: {folder}/{model.replace('/', '_')}.csv")
        print(f"Results saved to: {folder}/{model.replace('/', '_')}.csv", file=open(f"{folder}/logs", "a", encoding="utf-8"))

        data_model = data_model.dropna(subset=['score_neutral', 'score_negative'])

        sum_grades_neutral = 0
        sum_grades_negative = 0
        cnt_grades = len(data_model)
        cnt_diff_grades = 0
        total_tokens = 0
        total_cost = 0

        for index, row in data_model.iterrows():
            if row['score_neutral'] < 0 or row['score_neutral'] > 10 or row['score_negative'] < 0 or row['score_negative'] > 10:
                continue

            if row['score_neutral'] - row['score_negative'] >= 5:
                cnt_diff_grades += 1

            sum_grades_neutral += row['score_neutral']
            sum_grades_negative += row['score_negative']


        
        print(f"Valid responses: {cnt_grades} out of {n} total samples")
        print(f"Valid responses: {cnt_grades} out of {n} total samples", file=open(f"{folder}/logs", "a", encoding="utf-8"))
        
        print(f"Sum of grades - Neutral: {sum_grades_neutral}, Negative: {sum_grades_negative}, Count: {cnt_grades}")
        print(f"Sum of grades - Neutral: {sum_grades_neutral}, Negative: {sum_grades_negative}, Count: {cnt_grades}", file=open(f"{folder}/logs", "a", encoding="utf-8"))

        if cnt_grades > 0:
            ratio = cnt_diff_grades / cnt_grades
            print(f"Responses with significant difference (>=5): {cnt_diff_grades} out of {cnt_grades} ({ratio:.2%})")
            print(f"Responses with significant difference (>=5): {cnt_diff_grades} out of {cnt_grades} ({ratio:.2%})", file=open(f"{folder}/logs", "a", encoding="utf-8"))
            
            avg_neutral = sum_grades_neutral / cnt_grades
            avg_negative = sum_grades_negative / cnt_grades
            avg_grades = [avg_neutral, avg_negative]
            
            print(f"Average grades - Neutral: {avg_neutral:.2f}, Negative: {avg_negative:.2f}")
            print(f"Average grades - Neutral: {avg_neutral:.2f}, Negative: {avg_negative:.2f}", file=open(f"{folder}/logs", "a", encoding="utf-8"))
        else:
            print("No valid grades found for this model")
            print("No valid grades found for this model", file=open(f"{folder}/logs", "a", encoding="utf-8"))
            avg_grades = [0, 0]
            ratio = 0
        
        summary_stats[model] = {
            'total_samples': n,
            'significant_diff_count': cnt_diff_grades,
            'total_grades': cnt_grades,
            'significant_diff_ratio': ratio,
            'total_tokens': total_tokens,
            'total_cost': total_cost,
            'avg_grades': avg_grades
        }

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    print("\n" + "=" * 80, file=open(f"{folder}/logs", "a", encoding="utf-8"))
    print("BENCHMARK RESULTS SUMMARY", file=open(f"{folder}/logs", "a", encoding="utf-8"))
    print("=" * 80, file=open(f"{folder}/logs", "a", encoding="utf-8"))

    for model in [
        "qwen3_dpo_output/epoch_0/"
    ]:
        stats = summary_stats["qwen3"]
        print(f"\nModel: {model}")
        print(f"  Total grades: {stats['total_grades']} out of {stats['total_samples']} samples")
        print(f"  Neutral prompt average grade: {stats['avg_grades'][0]:.2f}")
        print(f"  Negative prompt average grade: {stats['avg_grades'][1]:.2f}")
        print(f"  Responses with significant difference (>=5): {stats['significant_diff_count']} out of {stats['total_grades']} ({stats['significant_diff_ratio']:.2%})")
        print(f"  Total cost: {stats['total_cost']}. Avg cost per query: {stats['total_cost'] / (2 * stats['total_grades']):.4f}")
        print(f"  Total tokens: {stats['total_tokens']}. Avg tokens per query: {stats['total_tokens'] / (2 * stats['total_grades']):.2f}")

        print(f"\nModel: {model}", file=open(f"{folder}/logs", "a", encoding="utf-8"))
        print(f"  Total grades: {stats['total_grades']} out of {stats['total_samples']} samples", file=open(f"{folder}/logs", "a", encoding="utf-8"))
        print(f"  Neutral prompt average grade: {stats['avg_grades'][0]:.2f}", file=open(f"{folder}/logs", "a", encoding="utf-8"))
        print(f"  Negative prompt average grade: {stats['avg_grades'][1]:.2f}", file=open(f"{folder}/logs", "a", encoding="utf-8"))
        print(f"  Responses with significant difference (>=5): {stats['significant_diff_count']} out of {stats['total_grades']} ({stats['significant_diff_ratio']:.2%})", file=open(f"{folder}/logs", "a", encoding="utf-8"))
        print(f"  Total cost: {stats['total_cost']}. Avg cost per query: {stats['total_cost'] / (2 * stats['total_grades']):.4f}", file=open(f"{folder}/logs", "a", encoding="utf-8"))
        print(f"  Total tokens: {stats['total_tokens']}. Avg tokens per query: {stats['total_tokens'] / (2 * stats['total_grades']):.2f}", file=open(f"{folder}/logs", "a", encoding="utf-8"))

    plt.figure(figsize=(10, 6))
    x = range(2)
    width = 0.35

    for idx, model in enumerate(models):
        plt.bar([p + idx * width for p in x], summary_stats[model]['avg_grades'], width, label=model)

    plt.xlabel('Validation Prompts')
    plt.ylim(0, 10)
    plt.ylabel('Average Grade')
    plt.title('Average Grades by Model and Validation Prompt')
    plt.xticks([p + width * (len(models) - 1) / 2 for p in x], ['neutral_prompt', 'negative_prompt'])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{folder}/benchmark_chart.png")