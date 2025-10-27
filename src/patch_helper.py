# manual_chat_formatters.py
from typing import List, Dict, Optional, Any, Tuple
import re
import pandas as pd
import os

_ALLOWED_ROLES = {"system", "user", "assistant", "developer"}

def _norm_role(role: str) -> str:
    r = (role or "user").lower()
    if r not in _ALLOWED_ROLES:
        return "system"
    return r

# =========================
# Common Qwen-style helpers
# =========================
def _block(role: str, content: str) -> str:
    return f"<|im_start|>{role}\n{content}<|im_end|>"

def _format_common_preamble(messages: List[Dict[str, str]], keep_developer: bool) -> List[str]:
    parts: List[str] = []
    for m in messages:
        role = _norm_role(m.get("role", ""))
        content = m.get("content", "") or ""
        if role == "developer" and not keep_developer:
            role = "system"
        parts.append(_block(role, content))
    return parts

def _assistant_prefix_with_trace(trace: Optional[str], leave_think_open: bool) -> str:
    prefix = "<|im_start|>assistant\n"
    if trace:
        if leave_think_open:
            prefix += f"<think>{trace}"
        else:
            prefix += f"<think>{trace}</think>\n"
    return prefix

# =========================
# GPT-OSS (Harmony style)
# =========================
def _gptoss_system(reasoning_effort: Optional[str]) -> str:
    # Minimal, template-compatible system; matches the Harmony fields enough for inference
    reff = reasoning_effort or "medium"
    sys = []
    sys.append("<|start|>system<|message|>")
    sys.append(f"Reasoning: {reff}\n\n")
    sys.append("# Valid channels: analysis, commentary, final.\nChannel must be included for every message.")
    sys.append("<|end|>")
    return "".join(sys)

def _gptoss_render_user(content: str) -> str:
    return f"<|start|>user<|message|>{content}<|end|>"

def _gptoss_render_developer(content: str) -> str:
    return f"<|start|>developer<|message|># Instructions\n\n{content}\n\n<|end|>"

def _gptoss_render_assistant_analysis(content: str) -> str:
    return f"<|start|>assistant<|channel|>analysis<|message|>{content}<|end|>"

def _gptoss_generation_open() -> str:
    # Open an assistant turn; model should emit <|channel|>final<|message|>... as it completes
    return "<|start|>assistant"

def _extract_dev_and_rest(messages: List[Dict[str, str]]) -> Tuple[Optional[str], List[Dict[str, str]]]:
    if messages and messages[0].get("role") in {"developer", "system"}:
        return messages[0].get("content", ""), messages[1:]
    return None, messages

def format_gpt_oss_manual(
    messages: List[Dict[str, str]],
    analysis_trace: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    add_generation_prompt: bool = False,
    leave_think_open: bool = False
) -> str:
    """
    GPT-OSS manual formatting:
      - Build Harmony-like blocks using <|start|>...<|message|>...<|end|>
      - Inject your trace as an assistant 'analysis' message BEFORE generation
      - Then open generation with '<|start|>assistant' so model emits the 'final' channel
    """
    parts: List[str] = []
    parts.append(_gptoss_system(reasoning_effort))

    dev, loop_messages = _extract_dev_and_rest(messages)
    if dev:
        parts.append(_gptoss_render_developer(dev))

    # Render user/assistant history (assistant content becomes final-channel inputs)
    for m in loop_messages:
        role = _norm_role(m.get("role"))
        content = m.get("content", "") or ""
        if role == "user":
            parts.append(_gptoss_render_user(content))
        elif role == "assistant":
            # Treat prior assistant content as a finalized message
            parts.append(f"<|start|>assistant<|channel|>final<|message|>{content}<|end|>")


    # Inject your analysis trace (if any) as a prior assistant analysis turn
    if analysis_trace:
        parts.append(_gptoss_render_assistant_analysis(analysis_trace))

    if leave_think_open:
        parts[-1] = parts[-1].strip("<|end|>")

    # Finally open the assistant for generation
    if add_generation_prompt:
        parts.append(_gptoss_generation_open())

    return "".join(parts)

# =========================
# Qwen (manual, <think> style)
# =========================
def format_qwen_manual(
    messages: List[Dict[str, str]],
    analysis_trace: Optional[str] = None,
    add_generation_prompt: bool = True,
    leave_think_open: bool = True,
) -> str:
    parts = _format_common_preamble(messages, keep_developer=False)
    assistant_prefix = _assistant_prefix_with_trace(analysis_trace, leave_think_open)
    return "\n".join(parts + [assistant_prefix + "<|im_end|>"]) if add_generation_prompt \
           else "\n".join(parts + [assistant_prefix])

# =========================
# Deepseek-distilled-Qwen (manual, <think> style)
# =========================
def format_deepseek_qwen_manual(
    messages: List[Dict[str, str]],
    analysis_trace: Optional[str] = None,
    add_generation_prompt: bool = False,
    leave_think_open: bool = True,
) -> str:
    parts = _format_common_preamble(messages, keep_developer=False)
    assistant_prefix = _assistant_prefix_with_trace(analysis_trace, leave_think_open)
    return "\n".join(parts + [assistant_prefix + "<|im_end|>"]) if add_generation_prompt \
           else "\n".join(parts + [assistant_prefix])

# =========================
# Dispatcher + stops
# =========================
def select_manual_formatter(model_name: str):
    m = (model_name or "").lower()
    if "gpt-oss" in m:
        return "gpt-oss", format_gpt_oss_manual
    if "deepseek" in m and "qwen" in m:
        return "deepseek-qwen", format_deepseek_qwen_manual
    if "qwen" in m:
        return "qwen", format_qwen_manual
    return "qwen", format_qwen_manual

def default_stop_tokens(model_family: str):
    if model_family == "gpt-oss":
        # final stop tokens that are safe; model usually emits <|channel|>final<|message|>...<|end|>
        return ["<|end|>", "<|return|>"]
    if model_family in {"qwen", "deepseek-qwen"}:
        return ["<|im_end|>", "<|endoftext|>"]
    return None

# =========================
# Output splitting
# =========================
# Strip stray Qwen-style blocks from raw text when we only want payloads
_CLEAN_IM_TOKENS_RE = re.compile(r"(?:<\|im_start\|>.*?\n|<\|im_end\|>)+", re.DOTALL)

def split_reasoning_and_answer(
    injected_trace: Optional[str],
    generated_text: str,
) -> Tuple[str, str]:
    """
    Returns (reasoning, answer).

    GPT-OSS path:
      - We injected analysis as a prior assistant turn.
      - On generation, model is expected to emit '<|channel|>final<|message|>ANSWER...'
      - We treat any text *before* the first '<|channel|>final<|message|>' as continued analysis
        (rare but possible if you opened generation without finishing analysis).
      - Answer is everything after that marker until optional '<|end|>'.

    Qwen/DeepSeek path:
      - Continue to split at '</think>' as before.

    Fallback:
      - If no markers found, put everything in 'answer'.
    """
    injected = (injected_trace or "")
    text = generated_text or ""

    # 1) GPT-OSS split
    final_tag = "<|channel|>final<|message|>"
    if final_tag in text:
        idx = text.find(final_tag)
        pre = text[:idx]
        post = text[idx + len(final_tag):]
        # Coalesce any continued analysis (if you ever left a GPT-OSS analysis open—uncommon)
        reasoning = (injected + pre).strip()
        # Trim trailing GPT-OSS end token if present
        end_tok = "<|end|>"
        if post.endswith(end_tok):
            post = post[: -len(end_tok)]
        return reasoning, post.strip()

    # 2) Qwen/DeepSeek split on </think>
    text_clean = _CLEAN_IM_TOKENS_RE.sub("", text)
    close_think = "</think>"
    if close_think in text_clean:
        idx = text_clean.find(close_think)
        cont_reasoning = text_clean[:idx]
        if cont_reasoning.startswith("<think>"):
            cont_reasoning = cont_reasoning[len("<think>"):]
        reasoning = (injected + cont_reasoning).strip()
        answer = text_clean[idx + len(close_think):].lstrip()
        return reasoning, answer.strip()

    # 3) No markers: assume it's just an answer (common for GPT-OSS if it skips explicit tags)
    return injected.strip(), text_clean.strip()

def load_reasoning_trace_for_instance(
    model: str,
    max_thinking_tokens: str,
    filters: Dict[str, Any],
    csv_path: str,
    first_fix: bool=True,
) -> Optional[str]:

    if not max_thinking_tokens:
        max_thinking_tokens = "None"
    else:
        max_thinking_tokens = str(max_thinking_tokens)
  
    if not os.path.isfile(csv_path):
        breakpoint()
        return None

    df = None
    for sep in [",", "\t", "|"]:
        try:
            df = pd.read_csv(csv_path, sep=sep)
            break
        except Exception:
            continue
    if df is None:
        return None

    for key in ("env", "scenario", "spec_type", "safety_prompt"):
        val = filters.get(key)
        if key == "env":
            val = f"{val.language}{val.framework}" # check
        elif key == "scenario":
            val = val.id
        if val is not None and key in df.columns:
            df = df[df[key] == val]

    target_col = 'new_trace' # if first_fix else 'full_fix'
    if df.empty or target_col not in df.columns:
        return None

    series = df[target_col].dropna()
    if series.empty:
        return None

    trace = str(series.iloc[0]).strip()
    
    # if "NO AMBIGUITY" in trace:
    #     target_col = 'reasoning_trace'
    #     if target_col not in df.columns:
    #         return None
    #     series = df[target_col].dropna()
    #     if series.empty:
    #         return None
        
    #     trace = str(series.iloc[0]).strip()
    
    # strip [start trace] and [end trace] if present
    trace = re.sub(r"^\[start trace\]", "", trace, flags=re.IGNORECASE).strip()
    trace = re.sub(r"\[end trace\]$", "", trace, flags=re.IGNORECASE).strip()
    # breakpoint()
    return trace or None