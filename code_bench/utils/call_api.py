import os
import time
import requests
import json
from typing import Dict, Any
from requests.exceptions import (
    ChunkedEncodingError,
    ReadTimeout,
    ConnectionError,
    HTTPError,
)

class APIError(Exception):
    """Custom exception for API errors."""
    pass

folder = "logs"

def call_api(query: str, model: str, max_tokens=30000, max_retries=5, timeout=1200, reasoning_effort="high"):
    with open("api.json", "r", encoding="utf-8") as f:
        api_config = json.load(f)
    if "gpt-oss" in model:
        query = f"Reasoning:High\n{query}"

    data = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "user", "content": query}
        ],
        "temperature": 0
    }

    if "openai/gpt-" in model:
        data["reasoning"] = {"effort": reasoning_effort}

    api_key = api_config["token1"] if model in api_config["models1"] else api_config["token2"]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if model.startswith("anthropic/"):
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        data["thinking"] = { "type": "enabled", "budget_tokens": int(0.75 * max_tokens) }
        data["temperature"] = 1

    url = api_config["links"].get(model, api_config.get("URL", "https://llm-proxy.t-tech.team/v1/chat/completions"))

    start_time = time.time()
    remaining_timeout = timeout
    last_exc = None
    json_resp = None

    for attempt in range(max_retries):
        if remaining_timeout <= 0:
            raise APIError("error, too many retries")

        try:
            resp = requests.post(url, headers=headers, json=data, timeout=remaining_timeout)

            try:
                json_resp = resp.json()
            except ValueError:
                json_resp = None

            if not resp.ok:
                msg = None
                if isinstance(json_resp, dict):
                    if isinstance(json_resp.get("error"), dict):
                        msg = json_resp.get("error", {}).get("message")
                    else:
                        msg = json_resp.get("error")
                raise APIError(f"HTTP {resp.status_code}: {msg or resp.text}")

            if not json_resp:
                raise APIError("Empty response from API")

            content = None

            if isinstance(json_resp, dict) and "content" in json_resp:
                blocks = json_resp.get("content")
                if isinstance(blocks, list) and blocks:
                    parts = []
                    for b in blocks:
                        if isinstance(b, dict) and b.get("type") == "text":
                            parts.append(b.get("text", ""))
                    content = "".join(parts).strip()
                else:
                    raise APIError("Empty response from API")

            elif isinstance(json_resp, dict) and "choices" in json_resp:
                try:
                    content = (json_resp["choices"][0]["message"].get("content") or "").strip()
                except Exception:
                    raise APIError("Empty response from API")

            else:
                raise APIError("Empty response from API")

            if not content:
                raise APIError("Empty content in response")
            return json_resp
        
        except ReadTimeout as e:
            last_exc = e
            break  # Not retrying on timeout

        except json.JSONDecodeError as e:
            last_exc = e
            if json_resp:
                print(json_resp, file=open(f"{folder}/logs", "a", encoding="utf-8"))
            elapsed = time.time() - start_time
            remaining_timeout = timeout - elapsed
            _log_and_wait(e, attempt, max_retries, query, model, folder)

        except (ChunkedEncodingError, ConnectionError, APIError) as e:
            last_exc = e
            elapsed = time.time() - start_time
            remaining_timeout = timeout - elapsed
            _log_and_wait(e, attempt, max_retries, query, model, folder)

        except HTTPError as e:
            last_exc = e
            elapsed = time.time() - start_time
            remaining_timeout = timeout - elapsed
            _log_http_error_and_wait(e, attempt, max_retries, query, model, folder)

    info = (
        f"query: {query[:100]} ...\n"
        f"model: {model}\n"
        f"Error! {last_exc}\n"
    )
    with open(f"{folder}/logs", "a", encoding="utf-8") as f:
        f.write(info)
        if json_resp is not None:
            f.write("\n--- json_resp ---\n")
            try:
                f.write(json.dumps(json_resp, ensure_ascii=False, indent=4))
            except Exception:
                f.write(str(json_resp))

        f.write("\n\n")

    raise APIError("error")


def _log_and_wait(exception, attempt, max_retries, query, model, folder):
    delay = 2 ** attempt
    with open(f"{folder}/logs", "a", encoding="utf-8") as f:
        f.write(
            f"\n[WARN] {type(exception).__name__}: {exception}\n"
            f"query: {query[:100]}\n"
            f"model: {model}\n"
            f"Attempt {attempt + 1}/{max_retries}, repeat after {delay} seconds.\n"
        )
    time.sleep(delay)


def _log_http_error_and_wait(e, attempt, max_retries, query, model, folder):
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