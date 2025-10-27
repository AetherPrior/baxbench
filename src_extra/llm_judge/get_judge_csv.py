import pandas as pd
import json

df = pd.read_csv("./intervention_test_none/deepseek-ai-DeepSeek-R1-0528-Qwen3-8B-None_all_analysis.csv", sep='\t', encoding='utf-8')
with open("./cache_judge/v2_gpt-4o_full_def_judge_deepseek.jsonl", "r") as f:
    cache = [json.loads(line) for line in f.readlines()]

df_cache = pd.DataFrame(cache)
df_cache = df_cache.rename(columns={"Id": "index"})

# merge on id (inner join)
df_cache['index'] = df_cache['index'].astype(int)
merged_df = pd.merge(df, df_cache, on='index', how='inner')
# make evidence a tuple
merged_df['evidence_lines'] = merged_df['evidence_lines'].apply(lambda x: tuple(x) if isinstance(x, list) else x)

merged_df = merged_df.sort_values(by=['action_key'])
merged_df.to_csv("./intervention_test_none/deepseek_gpt4o_judge_merged.csv", sep='\t', encoding='utf-8', index=False)