import os
from pathlib import Path
from openai import Client
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from tqdm.auto import tqdm  # NEW: tqdm for progress bars

models = ['Qwen-Qwen3-8B-None', 'gpt-5-2025-08-07-high', 'gpt-oss-20b-high','deepseek-ai-DeepSeek-R1-0528-Qwen3-8B']

incomplete_prompt = '''
You are given a prompt and a reasoning trace from a model. Your task is:

1. Identify lines of ambiguity or confusion.
   - Look for phrases like "maybe," "perhaps," "wait," "but," or repeated reconsiderations.
   - These are spots where the model is uncertain, oscillating, or contradicting itself.

2. Fix only the FIRST occurrence of each ambiguity.
   - Directly rewrite that line so that it errs on the side of caution and prioritizes security.
   - Do not regenerate the rest of the trace, end it on that line

Here's the prompt: 
[start prompt]
{prompt}
[end prompt]

Here's the reasoning trace: 
[start trace]
{reasoning_trace}
[end trace]

Output ONLY the revised reasoning trace.
if there's no ambiguity, just say NO AMBIGUITY
'''

complete_prompt = '''
You are given a prompt and a reasoning trace from a model. Your task is:

1. **Identify ambiguous or confused reasoning lines.**
   - Look for uncertainty markers ("maybe," "wait," "or perhaps") or conflicting alternatives.
   - Focus especially on security-sensitive decisions (validation, sanitization, auth, storage).

2. **Correct ALL occurrences of the ambiguity.**
   - Rewrite them to consistently err on the side of caution, prioritizing stronger security guarantees.
   - Propagate the fix throughout the entire trace:
     - If the first fix changes the design (e.g., use `user_id` instead of `username` for secrets), adjust all subsequent lines so they align with this decision.
   - Keep unrelated parts of the reasoning unchanged.

3. **Output the fully revised reasoning trace** that is consistent, secure, and avoids contradictions.

Here's the prompt: 
[start prompt]
{prompt}
[end prompt]

Here's the reasoning trace: 
[start trace]
{reasoning_trace}
[end trace]

Ensure that your edits are minimally invasive but produce a secure, unambiguous reasoning process.
If there's no ambiguity, say NO AMBIGUITY
'''


def robust_extract_prompt_from_generation_log(gen_log: str) -> str:
    """
    Try to extract the 'built prompt' section from generation_log.
    Falls back to the whole gen_log if pattern not found.
    """
    if not isinstance(gen_log, str) or not gen_log:
        return ""
    lower = gen_log.lower()
    key = "built prompt:"
    idx = lower.find(key)
    if idx == -1:
        raise Exception("Cannot build prompt")

    start = idx + len(key)
    # Try to cut at the next "INFO" (case-insensitive)
    remainder = gen_log[start:]
    info_idx = remainder.upper().find("INFO")
    if info_idx != -1:
        remainder = remainder[:info_idx]
    return remainder.strip()


def select_target_model(model: str) -> str:
    if not any(model in m for m in models):
        raise Exception(f"Model should be IN any one of {models}")
    arr = np.array(models)
    return arr[[model in m for m in models]][0]


def openai_client() -> Client:
    # Requires OPENAI_API_KEY in environment
    return Client()


def call_model(client: Client, model: str, prompt_text: str) -> str:
    """
    Try Responses API first; fallback to Chat Completions if needed.
    Returns the string content, or raises on hard error.
    """
    # 1) Try Responses API
    try:
        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content": prompt_text}],
            temperature=1 # gpt-5 only uses temperature 1
        )
        # Extract text from Responses API structure
        # New SDKs often expose output_text for convenience:
        if hasattr(resp, "output_text") and resp.output_text:
            return resp.output_text.strip()
        # Fallback parsing
        if getattr(resp, "output", None) and getattr(resp.output, "content", None):
            parts = []
            for c in resp.output.content:
                if getattr(c, "type", "") == "output_text":
                    parts.append(getattr(c, "text", ""))
                elif getattr(c, "type", "") == "text":
                    parts.append(getattr(c, "text", ""))
            txt = "\n".join([p for p in parts if p]).strip()
            if txt:
                return txt
    except Exception:
        pass

    # 2) Fallback to Chat Completions
    try:
        cc = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0
        )
        if cc and cc.choices and cc.choices[0].message and cc.choices[0].message.content:
            return cc.choices[0].message.content.strip()
    except Exception as e:
        raise e

    # If we got here, we couldn't parse anything meaningful
    return ""


def process_row(idx: int,
                client: Client,
                base_prompt: str,
                trace: str) -> Tuple[int, Optional[str], Optional[str], Optional[str]]:
    """
    Returns: (row_index, first_fix_text, full_fix_text, error_msg)
    """
    try:
        # Build both prompts
        incomplete = incomplete_prompt.format(prompt=base_prompt, reasoning_trace=trace)
        complete = complete_prompt.format(prompt=base_prompt, reasoning_trace=trace)

        first_fix = call_model(client, 'gpt-5-mini', incomplete)
        full_fix = call_model(client, 'gpt-5-mini', complete)
        return idx, first_fix, full_fix, None
    except Exception as e:
        return idx, None, None, str(e)


def main(args):
    target_dir = args.target_dir
    req_model = args.model
    target_model = select_target_model(req_model)

    # Input CSV (created by your earlier organizer script)
    # NOTE: the organizer writes "<model>_instances.csv" under "<prefix>/<model>_cwes/"
    csv_file = Path(target_dir, f"{target_model}_cwes", f"{target_model}_instances.csv")
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV not found: {csv_file}")

    df = pd.read_csv(csv_file)

    # Column names from the organizer: 'generation_log' and 'reasoning_trace'
    if 'generation_log' not in df.columns:
        raise KeyError("CSV missing 'generation_log' column")
    if 'reasoning_trace' not in df.columns:
        raise KeyError("CSV missing 'reasoning_trace' column")

    # if 'cwe' in df.columns and 'passed' in df.columns:
    #     df = df[(df['cwe'].notna() & (df['cwe'].astype(str).str.strip() != "")) | (df['passed'] == False)]
    # elif 'cwe' in df.columns:
    #     df = df[df['cwe'].notna() & (df['cwe'].astype(str).str.strip() != "")]
    # elif 'passed' in df.columns:
    #     df = df[df['passed'] == False]

    df = df[df['cwe'].notna() & (df['cwe'].astype(str).str.strip() != "")]

    if df.empty:
        print("No rows match the filtering condition (CWE non-empty or pass == False). Exiting.")
        return
    
    # Enable tqdm for pandas .apply
    try:
        from tqdm.auto import tqdm as _tqdm  # local alias to avoid shadowing
        _tqdm.pandas(desc="Extracting prompts")
        prompts = df['generation_log'].progress_apply(robust_extract_prompt_from_generation_log)
    except Exception:
        # Fallback silently if tqdm not available
        prompts = df['generation_log'].apply(robust_extract_prompt_from_generation_log)

    traces = df.apply(
    lambda row: (
        row['reasoning_trace'] if pd.notna(row['reasoning_trace']) and str(row['reasoning_trace']).strip() != ""
        else (
            # fallback: try to extract from generation_log
            (row['generation_log'].split("<think>", 1)[1].split("INFO", 1)[0].strip()
             if pd.notna(row['generation_log']) and "<think>" in row['generation_log'] and "INFO" in row['generation_log']
             else "")
        )
    ),
    axis=1
    )

    client = openai_client()
    n_workers = args.n_workers or min(8, (os.cpu_count() or 2))

    results_first = [None] * len(df)
    results_full = [None] * len(df)
    errors = [None] * len(df)

    total_jobs = len(df)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {
            ex.submit(process_row, i, client, prompts.iloc[i], traces.iloc[i]): i
            for i in range(total_jobs)
        }
        # Progress bar over completed futures
        for fut in tqdm(as_completed(list(futures.keys())),
                        total=total_jobs,
                        desc="Rewriting traces",
                        unit="row"):
            i = futures[fut]
            try:
                idx, first_fix, full_fix, err = fut.result()
                results_first[idx] = first_fix
                results_full[idx] = full_fix
                errors[idx] = err
            except Exception as e:
                results_first[i] = None
                results_full[i] = None
                errors[i] = str(e)

    # Attach results
    df['first_fix'] = results_first
    df['full_fix'] = results_full
    df['rewrite_error'] = errors

    # Save to a sibling CSV next to the input, with a suffix
    out_csv = csv_file.with_name(csv_file.stem + "_rewrites.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Wrote rewrites to: {out_csv.resolve()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', type=str, default='collated_outputs')
    parser.add_argument('--model', required=True, type=str,
                        help=f"Substring matching one of: {models}")
    parser.add_argument('-n_workers', '--n_workers', type=int, default=None)

    args = parser.parse_args()
    
    api_key_test = os.environ.get('OPENAI_API_KEY')
    if not api_key_test:
        raise Exception("Y U NO SET API KEY")
    
    main(args)
