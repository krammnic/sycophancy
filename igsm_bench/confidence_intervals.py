import json
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from validate import calculate_scores
import os
import time

parser = argparse.ArgumentParser(description='Benchmark LLM models')
parser.add_argument('--models', nargs='+', default="openrouter/qwen/qwen3-8b", help='List of models to benchmark')
parser.add_argument('--judge_model', type=str, default="tgpt/qwen3-235b-a22b", help='Model to judge answers')
parser.add_argument('--url', type=str, default="https://llm-proxy.t-tech.team/v1/chat/completions", help='Base URL for API requests')
parser.add_argument('--folder', type=str, default="benchmark_results", help='Folder to save results')
parser.add_argument('--l_prompt_id', type=int, default=0, help='Left bound for prompt_id interval')
parser.add_argument('--r_prompt_id', type=int, default=10, help='Right bound for prompt_id interval')
parser.add_argument('--l_op', type=int, default=7, help='Left bound for max_op interval')
parser.add_argument('--r_op', type=int, default=21, help='Right bound for max_op interval')
parser.add_argument('--l_edge', type=int, default=15, help='Left bound for max_edge interval')
parser.add_argument('--r_edge', type=int, default=26, help='Right bound for max_edge interval')
parser.add_argument('--reset_data', type=bool, default=True, help='Reset data')

args = parser.parse_args()
if args.models:
    models = args.models
else:
    raise ValueError("No models specified. Use --models argument")
model_judge = args.judge_model
workers = 55

n_per_diff = 5
seed = 23

def sample_tasks_for_prompt(df_all, prompt_set_id):
    return (df_all
            .groupby(["max_op", "max_edge"], group_keys=False)
            .sample(n=n_per_diff, replace=False, random_state=seed + prompt_set_id)
            .reset_index(drop=True))

with open("prompts/prompt_sets.json", "r", encoding="utf-8") as f:
    prompt_sets = json.load(f)

st_op, f_op, st_edge, f_edge = 7, 21, 15, 26
l_op, r_op = args.l_op, args.r_op
l_edge, r_edge = args.l_edge, args.r_edge
l_prompt_id, r_prompt_id = args.l_prompt_id, args.r_prompt_id

def process_edge(max_op, max_edge, df_sampled, model, cfg, prompt_set_id, start, log_dir):
    df = df_sampled[(df_sampled["max_op"] == max_op) & (df_sampled["max_edge"] == max_edge)].reset_index(drop=True)

    correct = incompetence_contr = incompetence_solve = sycophancy = error = failure = total = 0
    results_list = [None] * len(df)
    tasks_result_local = []

    for i, (_, row) in enumerate(df.iterrows()):
        result = calculate_scores(row, model, max_op, max_edge, i, cfg)
        total += 1
        results_list[i] = result["status"]

        result["prompt_id"] = prompt_set_id
        tasks_result_local.append(result)

        status = result["status"]
        if status == "correct":
            correct += 1
        elif status == "sycophancy":
            sycophancy += 1
        elif status == "incompetence/unable to find contradiction":
            incompetence_contr += 1
        elif status == "incompetence/unable to solve":
            incompetence_solve += 1
        elif status == "error":
            error += 1
        elif status == "failure":
            failure += 1
        else:
            print("Unknown result:", status)

    cur_result = {
        "prompt_id": prompt_set_id,
        "max_op": max_op,
        "max_edge": max_edge,
        "correct": correct,
        "incompetence_to_find_contradiction": incompetence_contr,
        "incompetence_to_solve": incompetence_solve,
        "sycophancy": sycophancy,
        "error": error,
        "total": total,
        "failure": failure,
        "sycophancy_percentage": sycophancy / total * 100 if total else 0,
        "incompetence_percentage": (incompetence_contr + incompetence_solve) / total * 100 if total else 0,
        "sycophancy_precision": sycophancy / (correct + sycophancy) * 100 if (correct + sycophancy) else 0,
    }

    execution_time = time.time() - start

    with open(f"{log_dir}/difficulty_{prompt_set_id}_{max_op}_{max_edge}.txt", "w", encoding="utf-8") as f:
        f.write(
            f"dif_{prompt_set_id}_{max_op}_{max_edge}: {execution_time:.2f}\n"
            f"Status: {results_list}\n"
            f"Correct: {correct}\n"
            f"Incompetence/unable to solve: {incompetence_solve}\n"
            f"Incompetence/unable to find contradiction: {incompetence_contr}\n"
            f"Sycophancy: {sycophancy}\n"
            f"Error: {error}\n"
            f"Failure: {failure}\n"
            f"Total: {total}\n"
        )

    return cur_result, tasks_result_local

def main():
    df_all = pd.read_csv("data/iGSM_bench_data.csv")
    for model in models:
        os.makedirs(f"logs/confidence_intervals/{model.replace('/', '_')}/tasks", exist_ok=True)
        os.makedirs(f"logs/confidence_intervals/{model.replace('/', '_')}/stats", exist_ok=True)
        os.makedirs(f"confidence_intervals_results/{model.replace('/', '_')}", exist_ok=True)
        start = time.time()
        try:
            df_model = pd.read_csv(f"confidence_intervals_results/{model.replace('/', '_')}/model_result.csv")
            model_result = df_model.to_dict(orient='records')
        except:
            model_result = []
        try:
            df_tasks = pd.read_csv(f"confidence_intervals_results/{model.replace('/', '_')}/tasks_result.csv")
            tasks_result = df_tasks.to_dict(orient='records')
        except:
            tasks_result = []
        for prompt_set_id, cfg in enumerate(prompt_sets):
            if prompt_set_id < l_prompt_id or prompt_set_id > r_prompt_id:
                continue
            df_sampled = sample_tasks_for_prompt(df_all, prompt_set_id)
            for max_op in range(st_op, f_op):
                if (l_prompt_id == prompt_set_id and max_op < l_op) or (r_prompt_id == prompt_set_id and max_op > r_op):
                    continue

                edges = []
                for max_edge in range(st_edge, f_edge):
                    if prompt_set_id == l_prompt_id and max_op == l_op and max_edge < l_edge:
                        continue
                    if prompt_set_id == r_prompt_id and max_op == r_op and max_edge > r_edge:
                        continue
                    edges.append(max_edge)
                
                with ThreadPoolExecutor(max_workers=min(workers, len(edges))) as executor:
                    futures = {
                        executor.submit(process_edge, max_op, max_edge, df_sampled, model, cfg, prompt_set_id, start,
                                f"logs/confidence_intervals/{model.replace('/', '_')}/stats"): max_edge
                        for max_edge in edges
                    }

                    for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing edges {model}"):
                        cur_result, tasks_local = future.result()
                        model_result.append(cur_result)
                        tasks_result.extend(tasks_local)
                
                model_result_df = pd.DataFrame(model_result).set_index(['prompt_id', 'max_op', 'max_edge']).sort_index()
                model_result_df.to_csv(f"confidence_intervals_results/{model.replace('/', '_')}/model_result.csv")

                tasks_result_df = pd.DataFrame(tasks_result).set_index(['prompt_id', 'max_op', 'max_edge', 'task_num']).sort_index()
                tasks_result_df.to_csv(f"confidence_intervals_results/{model.replace('/', '_')}/tasks_result.csv")

if __name__ == "__main__":
    main()
