# completions_runner.py
import os
import requests
from typing import Any, Dict, List, Optional, TypedDict

from patch_helper import (
    select_manual_formatter,
    default_stop_tokens,
    split_reasoning_and_answer,
    load_reasoning_trace_for_instance
)

from thinking_budget_enforcer import run_vllm_enforced_completions

from env.base import REASONING_EFFORT_MODELS, REASONING_TOKENS_MODELS, THINK_TOKEN

class CompletionsItem(TypedDict):
    prompt: str                 # full prompt sent to vLLM (for logging/debug)
    raw: str                    # raw text returned by vLLM
    reasoning: str              # merged: injected trace + generated continuation (until </think> OR before <|channel|>final)
    answer: str                 # text after </think> (Qwen/DeepSeek) or after <|channel|>final<|message|> (GPT-OSS)
    model_family: str           # 'gpt-oss' | 'qwen' | 'deepseek-qwen'
    stop: Optional[List[str]]   # stop strings sent to vLLM

CompletionsResult = List[CompletionsItem]

def run_vllm_completions(
    *,
    base_url: str,
    model: str,
    batch_size: int,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: Optional[int],
    csv_path: str,  # path to reasoning traces CSV
    reasoning_effort: Optional[str] = None,
    max_thinking_tokens: int = 0,
    env: Optional[str] = None,
    scenario: Optional[str] = None,
    spec_type: Optional[str] = None,
    safety_prompt: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    leave_think_open: bool = False,   
    timeout: int = 1200,
    intervention_str: str = "Wait,"
) -> CompletionsResult:
    """
    vLLM /v1/completions runner with manual chat formatting and robust reasoning/answer split.

    - Loads trace from collated CSV (env/scenario/spec_type/safety_prompt -> full_fix),
      falls back to src/gpt_prompt.txt if CSV not found, else injects nothing.
    - Manually formats prompts for GPT-OSS (Harmony channels), Qwen, and DeepSeek-distilled-Qwen.
    - Posts to /v1/completions.
    - Parses reasoning vs answer:
        * GPT-OSS: splits at '<|channel|>final<|message|>' (answer after it, before optional '<|end|>').
        * Qwen/DeepSeek: splits at '</think>'; if none, all generation is reasoning.
    """

    if max_thinking_tokens is not None and max_thinking_tokens > 0:
        return run_vllm_enforced_completions(
            base_url=base_url,
            model=model,
            messages=messages,
            batch_size=batch_size,
            temperature=temperature,
            max_tokens=max_tokens,
            csv_path=csv_path,
            reasoning_effort=reasoning_effort,
            max_thinking_tokens=max_thinking_tokens,
            env=env,
            scenario=scenario,
            spec_type=spec_type,
            safety_prompt=safety_prompt,
            headers=headers,
            leave_think_open=leave_think_open,
            timeout=timeout,
            intervention_str=intervention_str
        )

    # 1) Trace lookup (CSV -> file -> None)
    analysis_trace: Optional[str] = None
    try:
        analysis_trace = load_reasoning_trace_for_instance(
            model=model,
            max_thinking_tokens=max_thinking_tokens,
            filters={
                "env": env,
                "scenario": scenario,
                "spec_type": spec_type,
                "safety_prompt": safety_prompt,
            },
            first_fix=leave_think_open,
            csv_path=csv_path
        )
    except Exception:
        analysis_trace = None


    # 2) Manual formatting per model family
    family, formatter = select_manual_formatter(model)
    if family == "gpt-oss":
        formatted_prompt = formatter(
            messages=messages,
            analysis_trace=analysis_trace,
            reasoning_effort=reasoning_effort,
            add_generation_prompt= not leave_think_open,
            leave_think_open=leave_think_open,
        )
    else:
        formatted_prompt = formatter(
            messages=messages,
            analysis_trace=analysis_trace,
            add_generation_prompt= False, # not leave_think_open,
            leave_think_open=leave_think_open,
        )

    stop_list = default_stop_tokens(family)

    # 3) POST /v1/completions
    if headers is None:
        headers = {"Content-Type": "application/json"}

    url = f"{base_url.rstrip('/')}/v1/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": formatted_prompt,
        "temperature": float(temperature),
        "max_tokens": max_tokens,
        "stream": False,
        "stop": stop_list,
        "n": batch_size,
    }
    # prune Nones
    payload = {k: v for k, v in payload.items() if v is not None}
    # breakpoint()
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    # 4) Extract raw completion text
    raw = []
    try:
        for data in data["choices"]:
            raw.append(data.get("text", "") or "")
    except Exception:
        raw = []

    # 5) Split into reasoning vs answer (handles GPT-OSS channels and Qwen/DeepSeek </think>)
    reasoning = []
    answer = []
    for raw_text in raw:
        r, a = split_reasoning_and_answer(analysis_trace, raw_text)
        reasoning.append(r)
        answer.append(a)
    # breakpoint()
    # return {
    #     "prompt": formatted_prompt,
    #     "raw": raw,
    #     "reasoning": reasoning,
    #     "answer": answer,
    #     "model_family": family,
    #     "stop": stop_list,
    # }
    return [{
        "prompt": formatted_prompt,
        "raw": raw[i],
        "reasoning": reasoning[i],
        "answer": answer[i],
        "model_family": family,
        "stop": stop_list,
    } for i in range(len(raw))]

def run_ollama_completions(
    *,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: Optional[int],
    reasoning_effort: Optional[str] = None,
    max_thinking_tokens: Optional[int] = None,
    env: Optional[str] = None,
    scenario: Optional[str] = None,
    spec_type: Optional[str] = None,
    safety_prompt: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    leave_think_open: bool = False,   # applies to Qwen/DeepSeek (<think> path). GPT-OSS ignores this.
    timeout: int = 1200,
) -> CompletionsResult:
    """
    vLLM /v1/completions runner with manual chat formatting and robust reasoning/answer split.

    - Loads trace from collated CSV (env/scenario/spec_type/safety_prompt -> full_fix),
      falls back to src/gpt_prompt.txt if CSV not found, else injects nothing.
    - Manually formats prompts for GPT-OSS (Harmony channels), Qwen, and DeepSeek-distilled-Qwen.
    - Posts to /v1/completions.
    - Parses reasoning vs answer:
        * GPT-OSS: splits at '<|channel|>final<|message|>' (answer after it, before optional '<|end|>').
        * Qwen/DeepSeek: splits at '</think>'; if none, all generation is reasoning.
    """
    # 1) Trace lookup (CSV -> file -> None)
    analysis_trace: Optional[str] = None
    try:
        if model.lower() in REASONING_EFFORT_MODELS:
            if reasoning_effort is None:
                reasoning_effort = "medium"
            thinking_suffix = reasoning_effort
        elif model.lower() in REASONING_TOKENS_MODELS:
            thinking_suffix = max_thinking_tokens
        analysis_trace = load_reasoning_trace_for_instance(
            model=model,
            max_thinking_tokens=thinking_suffix,
            filters={
                "env": env,
                "scenario": scenario,
                "spec_type": spec_type,
                "safety_prompt": safety_prompt,
            },
            first_fix=leave_think_open
        )
    except Exception:
        analysis_trace = None


    # Optional: budget injected trace length (simple character budget;
    # replace with token-based logic if you prefer)
    if analysis_trace and max_thinking_tokens and max_thinking_tokens > 0:
        analysis_trace = analysis_trace[:max_thinking_tokens]

    # 2) Manual formatting per model family
    family, formatter = select_manual_formatter(model)
    if family == "gpt-oss":
        # GPT-OSS ignores leave_think_open (uses analysis/final channels)
        formatted_prompt = formatter(
            messages=messages,
            analysis_trace=analysis_trace,
            reasoning_effort=reasoning_effort,
            add_generation_prompt= not leave_think_open,
            leave_think_open=leave_think_open,
        )
    else:
        formatted_prompt = formatter(
            messages=messages,
            analysis_trace=analysis_trace,
            add_generation_prompt= not leave_think_open,
            leave_think_open=leave_think_open,
        )
    stop_list = default_stop_tokens(family)

    # 3) POST /api/generate
    if headers is None:
        headers = {"Content-Type": "application/json"}

    url = f"{base_url.rstrip('/')}/api/generate"

    payload: Dict[str, Any] =  {
                    "model": model,
                    "prompt": formatted_prompt,
                    "stream": False,
                    "options": {
                        "temperature": float(temperature),
                        "num_predict": max_tokens,
                        # "stream": False,
                        "stop": stop_list,
                    },
                }
    # prune Nones
    payload = {k: v for k, v in payload.items() if v is not None}

    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    # 4) Extract raw completion text
    raw = ""
    try:
        raw = data.get("response","") or ""
    except Exception:
        raw = ""

    # 5) Split into reasoning vs answer (handles GPT-OSS channels and Qwen/DeepSeek </think>)
    reasoning, answer = split_reasoning_and_answer(analysis_trace, raw)

    return {
        "prompt": formatted_prompt,
        "raw": raw,
        "reasoning": reasoning,
        "answer": answer,
        "model_family": family,
        "stop": stop_list,
    }
