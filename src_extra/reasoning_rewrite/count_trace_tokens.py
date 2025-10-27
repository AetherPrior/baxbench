from typing import Any, Dict, List, Optional, Tuple
import os
import pandas as pd
from tqdm import tqdm
import json
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer

task_cfl = (
    "Minimally rewrite the entire trace so it explicitly shows the order of safety-"
    "relevant checks (validation, authentication, authorization, and other defenses) "
    "relative to the core logic. Summarize whether each sensitive operation is only "
    "reachable after the required checks on all code paths; if uncertain, label it "
    "as 'unclear'. Preserve the original language, phrasing, and structure as much "
    "as possible, only edit/add/remove lines strictly necessary to make the safety "
    "flow explicit. Edit only the provided trace text."
)

task_recovery = (
    "Minimally rewrite the entire trace so it explicitly shows the order of error-handling and recovery steps (detection, containment, fallback, and restoration) relative to the core logic. Preserve the original language, phrasing, and structure as much as possible, only edit/add/remove lines strictly necessary to make the error-recovery flow explicit. Edit only the provided trace text."
)

#  "The model explicitly traces sensitive variables from data from source to sink; notes where it's validated, transformed or stored."

task_dfl = (
    "Minimally rewrite the entire trace so it explicitly shows the flow of sensitive data (e.g., personal data, credentials, secrets) from sources (e.g., user input, files, network) to sinks (e.g., databases, external services, logs). Preserve the original language, phrasing, and structure as much as possible, only edit/add/remove lines strictly necessary to make the sensitive data flow explicit. Edit only the provided trace text."
)

# "The model generates partial or skeletal code—such as pseudocode, templates, or placeholders—that outlines the structure or major steps of the solution before producing a complete implementation."

task_scaffold = (
    "Minimally rewrite the entire trace so it explicitly shows the scaffolding structure of the code (e.g., function definitions, class structures, comments) that outlines the major steps of the solution before producing a complete implementation. If scaffolding is already present, simply copy the existing scaffold without modification. Preserve the original language, phrasing, and structure as much as possible, only edit/add/remove lines strictly necessary to make the scaffolding structure explicit. Edit only the provided trace text."
)

task_mapping = {
    "cfl": task_cfl,
    "recovery": task_recovery,
    "dfl": task_dfl,
    "scaffold": task_scaffold
}

diff_prompt = """ You will be given a prompt and a model's internal thought. 

{task}

Output format (strict JSON only): produce a JSON array of edit objects. Each edit object must have exactly two string fields:

"find" — the exact line (or block of consecutive lines) to locate in the original trace.
For additions, put the line that precedes where the new line(s) will be inserted.
For removals, put the exact consecutive block of lines to be removed (copy the lines exactly).
For replacements, put the exact line(s) to be replaced.

"replace" — the exact new text that should replace the "find" text.
For additions, the "replace" value should contain the original "find" text followed immediately by the new lines (i.e., the effect is inserting the new lines after the "find").
For removals, set "replace" to the empty string ("").
For replacements, set "replace" to the new line(s) that should take the place of the "find" block.

Return valid JSON only (an array of objects). 
Matches are exact: whitespace and punctuation must match the original lines for "find" to succeed.
If multiple consecutive lines must be matched/edited, include them in "find" as a single multi-line string (preserve exact newlines).
If you must add multiple new lines, include them in "replace" using \\n and appropriate indentation so the trace style is preserved.

Prompt: 

{prompt}

Reasoning trace:

{reasoning_trace}
"""

gpt_prompt = """ 
## PROMPT:
<PROMPT>
{prompt}
</PROMPT>

## trace:
<TRACE>
{reasoning_trace}
</TRACE>
"""

# load tokenizer for model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

# helper to get collated csv path
def get_collated_csv(collate_path: str, model: str, collated_dir: str) -> pd.DataFrame:
    # check if csv already exists
    csv_path = os.path.join(collated_dir, f'{model}_reasoning_traces.csv')
    # breakpoint()
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df
    # get all gen_{scenario}_{env}_{temp}_{prompt_type}_{safety_prompt}.log files
    files = [f for f in os.listdir(collate_path) if f.startswith('gen_') and f.endswith('.log')]

    # filter those whose env starts with 'Python'
    files = [f for f in files if f.split('_')[2].startswith('Python')]

    # extract trace from file: (between) <think> and </think>
    def extract_trace(file_path):
        with open(file_path, 'r') as f:
            data = f.read()
            try:
                trace = data.split('<think>')[1].split('</think>')[0].strip()
                # tokenize trace and truncate to 4000 tokens
                # tokens = tokenizer.encode(trace)
                # if len(tokens) > 4000:
                #     tokens = tokens[:4000]
                #     trace = tokenizer.decode(tokens)
                # breakpoint()
                return trace
            except IndexError:
                return None
            
    # extract prompt from file: (between) built prompt: and INFO 20
    def extract_prompt(file_path):
        with open(file_path, 'r') as f:
            data = f.read()
            try:
                prompt = data.split('built prompt:')[1].split('INFO 20')[0].strip()
                return prompt
            except IndexError:
                return None

    output = []
    for file in tqdm(files):
        file_path = os.path.join(collate_path, file)
        trace = extract_trace(file_path)
        prompt_text = extract_prompt(file_path)
        if trace and prompt_text:
            # create a dict with keys: prompt, trace, scenario, env, temperature, spec_type, safety_prompt
            output.append({
                "prompt": prompt_text,
                "trace": trace,
                "scenario": file.split('_')[1],
                "env": file.split('_')[2],
                "temperature": file.split('_')[3],
                "spec_type": file.split('_')[4],
                "safety_prompt": file.split('_')[5].replace('.log', ''),
                "trace_token_count": len(tokenizer.encode(trace))
            })

    # save to csv
    df = pd.DataFrame(output)
    save_path = os.path.join(collated_dir, f'{model}_reasoning_traces.csv')
    df.to_csv(save_path, index=False)
    return df

def prompt_rewrite(reasoning_trace: str, prompt: str, task: str, gpt_prompt: str, model='gpt-5-mini') -> str:
    openai = OpenAI()
    full_prompt = gpt_prompt.format(reasoning_trace=reasoning_trace, prompt=prompt, task=task)
    messages = [
        {'role': 'developer', 'content': "You are a trace rewriter. You will be given a trace from another LLM.\n"
        f"## TASK: {task}\n"
        "Provide only the whole, MINIMALLY REWRITTEN, trace for the LLM within <TRACE>...</TRACE> tags.\n" },
        {'role': 'user', 'content': full_prompt}
    ]
    # breakpoint()
    resp = openai.responses.create(
        model=model,
        input=messages,   # plain text prompt (no messages array needed)
        temperature=1.0,
        stream=False,
        max_output_tokens=16384,
        # reasoning={'effort': 'high', 'summary': 'detailed'},
    )
    return resp.output_text

def ig_match(lines: List[str], find_lines: List[str]) -> bool:
    if len(lines) != len(find_lines):
        return False
    
    find = False
    for l1, l2 in zip(lines, find_lines):
        # remove non-ascii characters
        l1 = ''.join(c for c in l1 if ord(c) < 128)
        l2 = ''.join(c for c in l2 if ord(c) < 128)

        if l1.strip().lower() == l2.strip().lower():
            find = True
        else:
            return False     

collated_dir = './all_output_collate'
model =   'deepseek-ai-DeepSeek-R1-0528-Qwen3-8B-None' # 'Qwen-Qwen3-8B-None'

collate_path = os.path.join(collated_dir, f"{model}_collated")


# load or get collated csv
df = get_collated_csv(collate_path, model, collated_dir)

rewritten_df = pd.read_csv(os.path.join(collated_dir, f'{model}_reasoning_rewrites_cfl.csv')) 
# # get average token count
# avg_token_count = df['trace_token_count'].mean()
# print(f"Average token count: {avg_token_count}")
# # get max token count
# max_token_count = df['trace_token_count'].max()
# print(f"Max token count: {max_token_count}")
# # get min token count
# min_token_count = df['trace_token_count'].min()
# print(f"Min token count: {min_token_count}")

# get average token count of rewritten traces using 'new_trace' column and tokenizer
if 'new_trace' in rewritten_df.columns:
    rewritten_token_count = rewritten_df['new_trace'].apply(lambda x: len(tokenizer.encode(x)))
    avg_rewritten_token_count = rewritten_token_count.mean()
    print(f"Average rewritten token count: {avg_rewritten_token_count}")
    max_rewritten_token_count = rewritten_token_count.max()
    # get the argmax row and the trace
    max_rewritten_row = rewritten_df.loc[rewritten_df['new_trace'].apply(lambda x: len(tokenizer.encode(x))) == max_rewritten_token_count]
    # get trace
    max_rewritten_trace = max_rewritten_row['new_trace'].values[0]
    # # truncate
    # if len(tokenizer.encode(max_rewritten_trace)) > 8192:
    #     tokens = tokenizer.encode(max_rewritten_trace)[:8192]
    #     max_rewritten_trace = tokenizer.decode(tokens)  
    
    # # send back to df
    # rewritten_df.loc[max_rewritten_row.index, 'new_trace'] = max_rewritten_trace
    # # save back to csv
    # rewritten_df.to_csv(os.path.join(collated_dir, f'{model}_reasoning_rewrites_cfl.csv'), index=False)

    # get token count again
    rewritten_token_count = rewritten_df['new_trace'].apply(lambda x: len(tokenizer.encode(x)))
    max_rewritten_token_count = rewritten_token_count.max()

    print(f"Max rewritten token count: {max_rewritten_token_count}")
    min_rewritten_token_count = rewritten_token_count.min()
    print(f"Min rewritten token count: {min_rewritten_token_count}")

if 'original_trace' in rewritten_df.columns:
    original_token_count = rewritten_df['original_trace'].apply(lambda x: len(tokenizer.encode(x)))
    avg_original_token_count = original_token_count.mean()
    print(f"Average original token count: {avg_original_token_count}")
    max_original_token_count = original_token_count.max()
    # get the argmax row and the trace
    max_original_row = rewritten_df.loc[rewritten_df['original_trace'].apply(lambda x: len(tokenizer.encode(x))) == max_original_token_count]

    # get the token count again
    original_token_count = rewritten_df['original_trace'].apply(lambda x: len(tokenizer.encode(x)))
    max_original_token_count = original_token_count.max()
    
    print(f"Max original token count: {max_original_token_count}")
    min_original_token_count = original_token_count.min()
    print(f"Min original token count: {min_original_token_count}")

import matplotlib.pyplot as plt
import os
import pandas as pd

datasets = ['cfl', 'dfl', 'scaffold', 'recovery']

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for idx, dataset in enumerate(datasets):
    # Load the rewritten dataframe for this dataset
    rewritten_df = pd.read_csv(os.path.join(collated_dir, f'{model}_reasoning_rewrites_{dataset}.csv'))
    
    # Calculate token counts
    if 'original_trace' in rewritten_df.columns and 'new_trace' in rewritten_df.columns:
        original_token_count = rewritten_df['original_trace'].apply(lambda x: len(tokenizer.encode(x)))
        new_token_count = rewritten_df['new_trace'].apply(lambda x: len(tokenizer.encode(x)))
        
        # Create subplot for this dataset
        ax = axes[idx]
        
        # Plot histograms
        ax.hist(original_token_count, bins=30, alpha=0.6, label='Original', color='blue', edgecolor='black')
        ax.hist(new_token_count, bins=30, alpha=0.6, label='New', color='orange', edgecolor='black')
        
        # Add statistics
        ax.axvline(original_token_count.mean(), color='blue', linestyle='--', linewidth=2, 
                   label=f'Original Mean: {original_token_count.mean():.0f}')
        ax.axvline(new_token_count.mean(), color='orange', linestyle='--', linewidth=2,
                   label=f'New Mean: {new_token_count.mean():.0f}')
        
        # Labels and title
        ax.set_xlabel('Token Count', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{dataset.upper()} - Trace Length Distribution', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(collated_dir, f'{model}_trace_length_histograms.png'), dpi=300, bbox_inches='tight')
plt.show()

print(f"Saved histogram plots to {os.path.join(collated_dir, f'{model}_trace_length_histograms.png')}")