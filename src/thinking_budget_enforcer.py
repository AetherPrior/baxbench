from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from transformers import AutoTokenizer
import requests
import math

from patch_helper import (
    select_manual_formatter,
    default_stop_tokens,
    split_reasoning_and_answer,
    load_reasoning_trace_for_instance
)

@dataclass
class CompletionsItem:
    prompt: str
    raw: str
    reasoning: str
    answer: str
    model_family: str
    stop: List[str]

CompletionsResult = List[CompletionsItem]

# --- Helpers -------------------------------------------------------------

_tokenizer_cache: Dict[str, Any] = {}

def get_tokenizer(model_name: str):
    tok = _tokenizer_cache.get(model_name)
    if tok is None:
        tok = AutoTokenizer.from_pretrained(model_name)
        _tokenizer_cache[model_name] = tok
    return tok

def token_len(text: str, model_name: str) -> int:
    if not text:
        return 0
    tok = get_tokenizer(model_name)
    return len(tok.encode(text))

def trim_to_budget(text: str, budget: int, model_name: str) -> Tuple[str, bool]:
    if budget <= 0 or not text:
        return text
    tok = get_tokenizer(model_name)
    ids = tok.encode(text)
    if len(ids) <= budget:
        return text, False
    ids = ids[:budget]
    return tok.decode(ids, skip_special_tokens=True), True

def _format_prompt_for_family(
    *,
    family: str,
    formatter,
    messages: List[Dict[str, str]],
    analysis_trace: Optional[str],
    reasoning_effort: Optional[str],
    leave_think_open: bool
) -> str:
    if family == "gpt-oss":
        return formatter(
            messages=messages,
            analysis_trace=analysis_trace,
            reasoning_effort=reasoning_effort,
            add_generation_prompt=not leave_think_open,
            leave_think_open=leave_think_open,
        )
    else:
        return formatter(
            messages=messages,
            analysis_trace=analysis_trace,
            add_generation_prompt=False,
            leave_think_open=leave_think_open,
        )

def _post_completions(
    *,
    base_url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: int
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/completions"
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

# --- Main function -------------------------------------------------------

def run_vllm_enforced_completions(
    *,
    base_url: str,
    model: str,
    batch_size: int,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: Optional[int],
    csv_path: Optional[str] = None,  # path to reasoning traces CSV
    reasoning_effort: Optional[str] = None,
    max_thinking_tokens: int = 0,
    env: Optional[str] = None,
    scenario: Optional[str] = None,
    spec_type: Optional[str] = None,
    safety_prompt: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    leave_think_open: bool = False,
    timeout: int = 1200,
    thinking_budget_max_retries: int = 6,   # NEW: per-generation max retries
    intervention_str = "Wait,"
) -> CompletionsResult:
    """
    vLLM /v1/completions runner with token-budgeted reasoning control.

    Budget logic (per generation):
      - If generated reasoning tokens < max_thinking_tokens:
          * Remove closing think tag (leave_think_open=True) and append " wait"
            to the injected analysis_trace, then re-query.
      - If generated reasoning tokens > max_thinking_tokens:
          * Slice reasoning to budget, forcibly close think tag (leave_think_open=False),
            then re-query.
      - Up to `thinking_budget_max_retries` times per generation.

    Notes:
      - Different generations may receive different interventions.
      - Works with GPT-OSS (channels) and Qwen/DeepSeek (</think>) via your manual formatters.
    """
    # breakpoint()
    # 1) Trace lookup (CSV -> file -> None)
    analysis_trace: Optional[str] = None
    if csv_path:
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
            breakpoint()

    if analysis_trace is None:
        breakpoint()
    # Optional: budget injected trace length upfront (simple hard cap)
    if analysis_trace and max_thinking_tokens and max_thinking_tokens > 0:
        analysis_trace, trimmed = trim_to_budget(analysis_trace, max_thinking_tokens, model)

    # breakpoint()
    if trimmed:
        leave_think_open = False  # forcibly close if trimmed
    
    # 2) Manual formatting per model family
    family, formatter = select_manual_formatter(model)
    formatted_prompt = _format_prompt_for_family(
        family=family,
        formatter=formatter,
        messages=messages,
        analysis_trace=analysis_trace,
        reasoning_effort=reasoning_effort,
        leave_think_open=leave_think_open,
    )

    stop_list = default_stop_tokens(family)

    # 3) POST initial /v1/completions (n=batch_size)
    if headers is None:
        headers = {"Content-Type": "application/json"}

    base_payload: Dict[str, Any] = {
        "model": model,
        "prompt": formatted_prompt,
        "temperature": float(temperature),
        "max_tokens": max_tokens,
        "stream": False,
        "stop": stop_list,
        "n": batch_size,
    }
    base_payload = {k: v for k, v in base_payload.items() if v is not None}
    data = _post_completions(base_url=base_url, headers=headers, payload=base_payload, timeout=timeout)

    # 4) Extract raw completion text
    raw_list: List[str] = []
    try:
        for choice in data.get("choices", []):
            raw_list.append(choice.get("text", "") or "")
    except Exception:
        raw_list = []

    # 5) Split into reasoning vs answer
    reasoning_list: List[str] = []
    answer_list: List[str] = []
    for raw_text in raw_list:
        r, a = split_reasoning_and_answer(analysis_trace, raw_text)
        reasoning_list.append(r)
        answer_list.append(a)

    # 6) Per-generation token budget enforcement with adaptive retries
    results: List[CompletionsItem] = []
    for i in range(len(raw_list)):
        current_reasoning = reasoning_list[i] or ""
        current_answer = answer_list[i] or ""
        current_raw = raw_list[i] or ""

        # Skip if no budget or already trimmed
        if not max_thinking_tokens or max_thinking_tokens <= 0 or trimmed:
            results.append(CompletionsItem(
                prompt=formatted_prompt,
                raw=current_raw,
                reasoning=current_reasoning,
                answer=current_answer,
                model_family=family,
                stop=stop_list,
            ))
            continue

        retries = 0
        r_tokens = token_len(current_reasoning, model)
        # We adapt per generation; we’ll construct individualized prompts for re-queries (n=1).
        while retries < thinking_budget_max_retries:
            if r_tokens == max_thinking_tokens:
                break  # exactly on budget

            # UNDER budget: remove closing think tag & append " wait"
            elif r_tokens < max_thinking_tokens:
                # New injected analysis trace is the *current model reasoning* plus a nudge
                # (without closing tag); formatter will keep it open.
                next_injected_trace = (current_reasoning.rstrip() + f"\n\n{intervention_str}").strip()
                next_formatted_prompt = _format_prompt_for_family(
                    family=family,
                    formatter=formatter,
                    messages=messages,
                    analysis_trace=next_injected_trace,
                    reasoning_effort=reasoning_effort,
                    leave_think_open=True,   # keep thinking open
                )
                payload_i = {
                    "model": model,
                    "prompt": next_formatted_prompt,
                    "temperature": float(temperature),
                    "max_tokens": max_thinking_tokens - r_tokens + 50,  # allow some buffer
                    "stream": False,
                    "stop": stop_list,
                    "n": 1,
                }
                payload_i = {k: v for k, v in payload_i.items() if v is not None}
                data_i = _post_completions(base_url=base_url, headers=headers, payload=payload_i, timeout=timeout)
                new_raw = (data_i.get("choices", [{}])[0].get("text") or "")
                new_reasoning, new_answer = split_reasoning_and_answer(next_injected_trace, new_raw)

                current_raw = new_raw
                current_reasoning = new_reasoning or ""
                current_answer = new_answer or ""
                retries += 1
                r_tokens = token_len(current_reasoning, model)
                continue

            else:
                break

        # OVER budget: trim to budget & close
        if r_tokens > max_thinking_tokens:
            analysis_trace, trimmed = trim_to_budget(current_reasoning, max_thinking_tokens, model)
        else:
            analysis_trace = current_reasoning
        # Close the think channel/tag by forcing leave_think_open=False
        next_formatted_prompt = _format_prompt_for_family(
            family=family,
            formatter=formatter,
            messages=messages,
            analysis_trace=analysis_trace,
            reasoning_effort=reasoning_effort,
            leave_think_open=False,
        )
        payload_i = {
            "model": model,
            "prompt": next_formatted_prompt,
            "temperature": float(temperature),
            "max_tokens": max_tokens,
            "stream": False,
            "stop": stop_list,
            "n": 1,
        }
        payload_i = {k: v for k, v in payload_i.items() if v is not None}
        data_i = _post_completions(base_url=base_url, headers=headers, payload=payload_i, timeout=timeout)
        new_raw = (data_i.get("choices", [{}])[0].get("text") or "")
        new_reasoning, new_answer = split_reasoning_and_answer(analysis_trace, new_raw)

        current_raw = new_raw
        current_reasoning = new_reasoning or ""
        current_answer = new_answer or ""

        # Done with per-gen enforcement
        results.append(CompletionsItem(
            prompt=formatted_prompt,
            raw=current_raw,
            reasoning=current_reasoning,
            answer=current_answer,
            model_family=family,
            stop=stop_list,
        ))

    # 7) Return normalized result list
    return [
        {
            "prompt": item.prompt,
            "raw": item.raw,
            "reasoning": item.reasoning,
            "answer": item.answer,
            "model_family": item.model_family,
            "stop": item.stop,
        }
        for item in results
    ]
