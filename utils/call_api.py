import os
import time
import requests
from typing import Dict, Any


class APIError(Exception):
    """Custom exception for API errors."""
    pass


def call_api(
    query: str,
    model: str,
    max_tokens: int = 30000,
    max_retries: int = 5,
    timeout: int = 1200,
    reasoning_effort: str = None,
) -> Dict[str, Any]:
    """
    Call the OpenRouter API (chat completions) with retries, error handling, and timeout.

    Args:
        query: The user prompt/query string.
        model: The model to use (e.g., 'openai/gpt-4', 'openai/gpt-5.2-thinking').
        max_tokens: Maximum tokens for the response.
        max_retries: Maximum number of retry attempts on transient failures.
        timeout: Request timeout in seconds.
        reasoning_effort: Reasoning effort level for Thinking models (e.g., 'high', 'xhigh').
                         Options: 'none', 'low', 'medium', 'high', 'xhigh'. Default: None.

    Returns:
        Parsed JSON response from the API.

    Raises:
        APIError: On non-retryable errors or when retries are exhausted.
    """
    # Read API key from apikey.txt file
    api_key_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "apikey.txt")
    try:
        with open(api_key_path, "r") as f:
            api_key = f.read().strip()
    except FileNotFoundError:
        raise APIError(f"API key file not found at {api_key_path}")
    except Exception as e:
        raise APIError(f"Failed to read API key: {e}")

    if not api_key:
        raise APIError("API key is empty")

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": query}
        ],
        "max_tokens": max_tokens,
    }
    
    # Add reasoning effort parameter if provided (OpenRouter expects a nested object)
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}

    attempt = 0
    backoff = 1  # initial backoff in seconds

    while attempt < max_retries:
        attempt += 1
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.Timeout:
            if attempt == max_retries:
                raise APIError(f"Request timed out after {max_retries} attempts")
            time.sleep(backoff)
            backoff *= 2
            continue
        except requests.RequestException as e:
            # Network error, DNS, etc.
            if attempt == max_retries:
                raise APIError(f"Network error: {e}")
            time.sleep(backoff)
            backoff *= 2
            continue

        # Got a response
        if resp.status_code == 200:
            try:
                return resp.json()
            except ValueError:
                raise APIError("Failed to parse JSON response")
        elif resp.status_code in (429, 502, 503, 504):
            # Rate limit or temporary server errors => retry
            if attempt == max_retries:
                raise APIError(f"Error {resp.status_code}: {resp.text}")
            time.sleep(backoff)
            backoff *= 2
            continue
        else:
            # 400s that aren't rate limit, etc. are likely non-retryable
            raise APIError(f"Error {resp.status_code}: {resp.text}")

    # If we reach here, retries have been exhausted
    raise APIError("Retries exhausted without successful response")
