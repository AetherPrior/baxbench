#!/usr/bin/env python3
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any
from openai import OpenAI
# import httpclient
from requests import Session
import jinja2


# Set the server URL
server_url = os.getenv("REASONING_REWRITE_SERVER_URL", "http://localhost:8000/v1/completions")

# Configure the client to be requests client
client = Session()

# load template gpt-oss.jinja
template_loader = jinja2.FileSystemLoader(searchpath="./")
template_env = jinja2.Environment(loader=template_loader)
template = template_env.get_template("gpt-oss.jinja2")


def send_request(idx: int, model: str = "gpt-oss-120b") -> Dict[str, Any]:
    messages = [
        {"role": "user", "content": f"Request #{idx}: What's 1+1?"}
    ]

    # format the prompt using the template "message"
    rendered_prompt = template.render(messages=messages, reasoning_effort="high", add_generation_prompt=False)
    print(f"Debug: rendered prompt for request #{idx}:\n{rendered_prompt}\n{'-'*40}")
    payload = {
        "model": model,
        "prompt": [rendered_prompt],
        "temperature": 1.0,
        "max_tokens": 4096,
    }

    resp = client.post(server_url, json=payload)
    resp = resp.json()
    # print(f"Debug: full response for request #{idx}: {resp}")
    # Be tolerant of different server shapes
    text = getattr(resp, "output_text", None)
    if text is None:
        # best-effort fallback
        try:
            choice = (getattr(resp, "output", None) or getattr(resp, "choices", []))[0]
            # choice can have "text" or "message" with "content"
            content = choice.get("text", None) or choice.get("message", {}).get("content", None)
            if isinstance(content, list):
                text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            elif isinstance(content, str):
                text = content
        except Exception:
            print(f"Warning: unable to parse response for request #{idx}: {resp}")
            text = ""
    return {"idx": idx, "text": (text or "").strip()}

def run_concurrent(n: int = 16, model: str = "gpt-oss-120b", max_workers: int = 8):
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(send_request, i, model): i for i in range(1, n + 1)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                res = fut.result()
                snippet = res["output"]
                print(f"Req #{idx} → {snippet}")
            except Exception as e:
                print(f"Req #{idx} → ERROR: {e}")

if __name__ == "__main__":
    run_concurrent(n=1, model="gpt-oss-120b", max_workers=8)
