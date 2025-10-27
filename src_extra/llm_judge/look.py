# %%
import pandas as pd 

# %%
import os
os.getcwd()

# %%
os.listdir("../../intervention_test_none/deepseek_human_annotations_final.csv")

# %%
human_df = pd.read_csv("../../intervention_test_none/deepseek_human_annotations_final.csv", sep='\t', encoding='utf-8')
qwen_df = pd.read_csv("../../intervention_test_none/deepseek_qwen_judge_merged.csv", sep='\t', encoding='utf-8')
gpt4o_df = pd.read_csv("../../intervention_test_none/deepseek_gpt4o_judge_merged.csv", sep='\t', encoding='utf-8')

# %%
# join human_annotation of human_df with qwen on index column
human_df['human_annotation']
print(human_df.columns)
# join human_annotation of human_df with qwen on index column
merged_qwen = pd.merge(qwen_df, human_df[['index', 'human_annotation']], on='index', how='inner', suffixes=('', '_human'))

# %%
merged_qwen

# %%



