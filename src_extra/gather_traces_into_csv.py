import os
import pandas as pd

models = ['Qwen-Qwen3-8B-None', 'gpt-oss-20b-high', 'gpt-5-2025-08-07-high', 'deepseek-ai-DeepSeek-R1-0528-Qwen3-8B-None']


def extract_reasoning(gen_test_data):
    gen_test_data = gen_test_data.split('<think>')[1]
    return gen_test_data

def get_reasoning(text):
    return [x for x in ['low','medium','high','None'] if x in text]

def gather_traces(models, target_suffix='_cwes', save_prefix='cwe'):
    file_data = {}
    for dir in os.listdir('./'):

        if os.path.isdir(dir) and dir in [name+target_suffix for name in models]:
            for file in os.listdir(os.path.join('./',dir)):
                if file.endswith('.log'):
                    cwe, gen_test, scenario, env, temp, prompt_type, granularity = ([''] + file.split('_'))[-7:]

                    model = dir.split('_')[0]
                    reasoning = get_reasoning(dir)

                    file_key = "_".join([cwe, scenario, env, temp, prompt_type, granularity])
                    if file_key not in file_data:
                        file_data[file_key] = {
                            'cwe': cwe,
                            'model': model,
                            'scenario': scenario,
                            'env': env,
                            'temp': temp,
                            'prompt_type': prompt_type,
                            'granularity': granularity,
                            'reasoning': 'thinking' if reasoning == 'None' else reasoning
                        }

                    
                    with open(os.path.join(dir, file), 'r') as f:
                        gen_test_data = f.read()

                    if gen_test == 'gen':
                        file_data[file_key]['gen_log'] = extract_reasoning(gen_test_data)
                    
                    elif gen_test == 'test':
                        file_data[file_key]['test_log'] = gen_test_data

            
    file_list = []
    for k, v in file_data.items():
        v['id'] = k
        file_list.append(v)

    df = pd.DataFrame(file_list)

    df.to_csv(f'{save_prefix}_analysis.csv', sep='\t', encoding='utf-8')

            
gather_traces(models=models, target_suffix='_cwes', save_prefix='cwe')
gather_traces(models=models, target_suffix='_sec', save_prefix='sec')
gather_traces(models=models, target_suffix='_failed', save_prefix='failed')

            
            


