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
from datasets import load_dataset
import re
import matplotlib.pyplot as plt 
import json
import random
from tqdm import tqdm
import time
from pandarallel import pandarallel

tqdm.pandas()
pandarallel.initialize(progress_bar=True)

ds = pd.read_parquet('humanity_last_exam.parquet')

ds = ds[ds['category'] == 'Math']

ds = ds[(ds['image'].isnull() | (ds['image'] == '')) & (ds['rationale_image'].isnull() | (ds['rationale_image'] == ''))]

ds = ds.reset_index().rename(columns={'index' : 'hle_index'})

n = ds.shape[0]

with open("prompts.json", "r", encoding="utf-8") as f:
    prompts = json.load(f)
    solve_prompt = prompts["solve_prompt"]
    val_prompts = prompts["val_prompts"]
    compare_prompt = prompts["compare_prompt"]

with open("solution_gen_models.json", "r", encoding="utf-8") as f:
    solve_models = json.load(f)["solution_gen_models"]

workers = 64

data = pd.DataFrame(columns=['hle_index', 'id', 'question', 'solution_attempt', 'solution_attempt_source', 'answer_attempt', 'answer',
       'answer_type', 'is_correct', 'rationale', 'raw_subject', 'category', 'canary'])

with open("api.json", "r", encoding="utf-8") as f:
    api = json.load(f)

URL = ""

def call_api(query: str, model: str, max_tokens=30000, max_retries=5, timeout=1200):
    data = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "user", "content": query}
        ],
        "temperature": 0
    }

    api_key = api["token1"] if model in api["models1"] else api["token2"]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if model.startswith("anthropic/"):
        headers["anthropic-version"] = "2023-06-01"

    url = api["links"].get(model, URL)

    start_time = time.time()
    remaining_timeout = timeout
    last_exc = None

    for attempt in range(max_retries):
        if remaining_timeout <= 0:
            return False, "Timeout exceeded before making the request."

        try:
            resp = requests.post(url, headers=headers, json=data, timeout=min(remaining_timeout, timeout // 2))
            resp.raise_for_status()
            json_resp = resp.json()

            if not json_resp:
                raise RuntimeError("Empty response from API")
            elif 'content' in json_resp:
                if isinstance(json_resp['content'], list) and len(json_resp['content']) > 0:
                    content = json_resp['content'][0].get('text', '')
                else:
                    raise RuntimeError("Empty response from API")
            else:
                if 'choices' in json_resp and len(json_resp['choices']) > 0 and 'message' in json_resp['choices'][0]:
                    content = json_resp["choices"][0]["message"].get("content")
                else:
                    raise RuntimeError("Empty response from API")
            if not content:
                raise RuntimeError("Empty content in response")

            return True, content

        except ReadTimeout as e:
            last_exc = e
            break  # Not retrying on timeout

        except json.JSONDecodeError as e:
            last_exc = e
            print(json_resp, file=open("logs", "a", encoding="utf-8"))
            elapsed = time.time() - start_time
            remaining_timeout = timeout - elapsed
            _log_and_wait(e, attempt, max_retries, query, model)

        except (ChunkedEncodingError, ConnectionError, RuntimeError) as e:
            last_exc = e
            elapsed = time.time() - start_time
            remaining_timeout = timeout - elapsed
            _log_and_wait(e, attempt, max_retries, query, model)

        except HTTPError as e:
            last_exc = e
            elapsed = time.time() - start_time
            remaining_timeout = timeout - elapsed
            _log_http_error_and_wait(e, attempt, max_retries, query, model)

    info = (
        f"query: {query[:100]} ...\n"
        f"model: {model}\n"
        f"Error! {last_exc}\n"
    )
    with open("logs", "a", encoding="utf-8") as f:
        f.write(info)

    return False, info


def _log_and_wait(exception, attempt, max_retries, query, model):
    delay = 2 ** attempt
    with open("logs", "a", encoding="utf-8") as f:
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
    with open("logs", "a", encoding="utf-8") as f:
        f.write(
            f"\n[ERROR] HTTP error: {e}\n"
            f"query: {query[:100]}\n"
            f"model: {model}\n"
            f"(status={status}, text={text!r})\n"
            f"Attempt {attempt + 1}/{max_retries}, repeat after {delay} seconds."
        )
    time.sleep(delay)


solve_jobs = []

#sampling models uniformly
base_count = n // len(solve_models)
extra = n % len(solve_models)

model_selection = []

for model in solve_models:
    model_selection += [model] * base_count

model_selection += random.sample(solve_models, extra)

random.shuffle(model_selection)

for i in range(n):
    model = model_selection[i]
    solve_jobs.append((i, f"{solve_prompt}\n{ds.loc[i, "question"]}", model))

with open("logs", "w", encoding="utf-8") as f:
    f.write(f"Solutions generation started. Total jobs: {len(solve_jobs)}")

results_ok = 0
results_fail = 0

common_columns = list(set(data.columns) & set(ds.columns))

with ThreadPoolExecutor(max_workers=workers) as executor:
    future_to_job = {
        executor.submit(call_api, query, model): (ind, query, model)
        for (ind, query, model) in solve_jobs
    }
    
    with tqdm(
        total=len(future_to_job),
        desc="Jobs",
        unit="job",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    ) as pbar:
        for fut in as_completed(future_to_job):
            ind, query, model = future_to_job[fut]
            ok, solution_attempt = fut.result()
            if ok:
                results_ok += 1

                new_row = ds.loc[ind, common_columns]
                new_row["solution_attempt"] = solution_attempt
                new_row["solution_attempt_source"] = model
                
                data.loc[len(data)] = new_row
            else:
                results_fail += 1
            
            pbar.set_postfix({
                'OK': results_ok,
                'Failed': results_fail,
                'Success Rate': f'{results_ok/(results_ok + results_fail)*100:.1f}%' if (results_ok + results_fail) > 0 else '0%'
            })
            pbar.update(1)
            


with open("logs", "a", encoding="utf-8") as f:
    f.write(
        f"Solutions generation finished. Successful jobs: {results_ok}\n"
        f"Errors: {results_fail}.\n"
    )

data.to_csv("solutions_03-12-25_02-53", index=False)

def process_row(row):
    try:
        if row['solution_attempt_source'] == 'rationale':
            row['answer_attempt'] = 'rationale answer'
            row['is_correct'] = True
            return row
        else:
            matches = re.findall(r"\\boxed\{([^}]*)\}", row['solution_attempt'])

            if not matches:
                row['answer_attempt'] = 'no answer'
                row['is_correct'] = False
                return row
            
            answer_attempt = matches[-1]

            row['answer_attempt'] = answer_attempt

            query = compare_prompt.replace("{{CORRECT_ANSWER}}", row['answer']) \
                     .replace("{{ATTEMPTED_ANSWER}}", answer_attempt)

            response = call_api(query, 'openrouter/deepseek/deepseek-r1-0528', max_tokens=5000, timeout=120)

            row['is_correct'] = response[0] and response[1] == "Yes"
            return row
    except:
        row['answer_attempt'] = None
        row['is_correct'] = None
        return row

data[['answer_attempt', 'is_correct']] = data.parallel_apply(
    lambda row: pd.Series(process_row(row)[['answer_attempt', 'is_correct']]), 
    axis=1
)

data.to_csv('final_benchmark.csv', index=False)
