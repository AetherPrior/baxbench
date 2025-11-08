import os
import pandas as pd
import json
import argparse

parser = argparse.ArgumentParser(description="Gather traces into CSV")
parser.add_argument('--models', nargs='+', help='List of model names', required=True)
parser.add_argument('--target_dir', type=str, help='Target directory containing logs', required=True)
parser.add_argument('--json_file', type=str, help='JSON file with overall results', required=True)
args = parser.parse_args()

# models = ['openai-gpt-oss-120b_high'] #, 'deepseek-ai-DeepSeek-R1-0528-Qwen3-8B-None'] #  'Qwen-Qwen3-8B-None'] #['Qwen-Qwen3-8B-None'] #, 'gpt-oss-20b-high', 'gpt-5-2025-08-07-high', 'deepseek-ai-DeepSeek-R1-0528-Qwen3-8B-None']

# target_dir = 'new_interventions/none/generic' # './intervention_test_none/specific/'

# json_file = 'new_interventions/none/overall_results_openai-gpt-oss-120b_1.0_openapi_generic_high.json' # './intervention_test_none/overall_results_openai-gpt-oss-120b_1.0_openapi_specific_high.json'

models = args.models
target_dir = args.target_dir
json_file = args.json_file

def get_results(gen_test_data, model, scenario, env, json_file_data):
    try:
        results = json.loads(gen_test_data)
        json_file_data = json.loads(json_file_data)
        pass_at_1 = None
        sec_pass_at_1 = None
        # find matching results in json_file_data
        for item in json_file_data['detailed_results']['data']:
            model_scenario = item['model_scenario']
            scenario_name = model_scenario.split('_')[1]
            env_name = item['environment'].split('_')[0]
            if (scenario_name == scenario) and (env_name == env):
                # pull out results for the model
                metrics = item['metrics']
                pass_at_1 = metrics['pass_at_k']['1']['value']
                sec_pass_at_1 = metrics['sec_pass_at_k']['1']['value']
                break

        if pass_at_1 is None:
            # capability: pass@1 estimation
            pass_at_1 = (results.get('num_passed_ft', 0) /
                                results.get('num_total_ft', 1))
        if sec_pass_at_1 is None:
            # safety: sec-pass@1 estimation
            sec_pass_at_1 = (results.get('num_passed_st', 0) /
                             results.get('num_total_st', 1))
            # check for cwes
            cwes = results.get('cwes', [])
            
            if len(cwes) == 0:
                sec_pass_at_1 = 1.0 if pass_at_1 == 1.0 else 0
            else:
                sec_pass_at_1 = 1 - (len(cwes) / results.get('num_total_st', 1))
            
        return {
            # 'num_passed_ft': results.get('num_passed_ft', 0),
            # 'num_total_ft': results.get('num_total_ft', 0),
            # 'num_total_st': results.get('num_total_st', 0),
            # 'num_st_exceptions': results.get('num_st_exceptions', 0),
            'capability_score': pass_at_1,
            'safety_score': sec_pass_at_1,
            'cwes': cwes
        }
    except Exception as e:
        breakpoint()
        raise Exception("Error parsing results") from e

def extract_prompt(gen_test_data):
    gen_test_data = "built prompt:".join(gen_test_data.split("built prompt:")[1:]).split('INFO 20')[0].strip()
    return gen_test_data

def extract_reasoning(gen_test_data):
    try:
        gen_test_data = gen_test_data.split('<think>')[1]
        if '</think>' in gen_test_data:
            gen_test_data = gen_test_data.split('</think>')[0].strip()
        else:
            gen_test_data = gen_test_data.split('INFO 20')[0].strip()
    except IndexError:
        gen_test_data = gen_test_data.split('got model responses:')[1]
        gen_test_data = gen_test_data.split('</think>')[0].strip()
    
    return gen_test_data

def get_reasoning(text):
    return [x for x in ['low','medium','high','None'] if x in text]

def gather_traces(models, json_file_data, target_suffix='_cwes', save_prefix='cwe'):
    file_data = {}
    for name in models:
        for dir in os.listdir(target_dir):
            if os.path.isdir(os.path.join(target_dir, dir)) and dir == name+target_suffix:
                for file in os.listdir(os.path.join(target_dir, dir)):
                    if file.endswith('.log') or file.endswith('.json'):
                        # get file format
                        file_format = 'log' if file.endswith('.log') else 'json'
                        # strip down file format
                        file = file.replace('.log','').replace('.json','')
                        gen_test, scenario, env, temp, prompt_type, safety_prompt = (file.split('_'))[-6:]
                        # breakpoint()
                        model = dir.split('_')[0]
                        reasoning = get_reasoning(dir)

                        file_key = "_".join([scenario, env, temp, prompt_type, safety_prompt])
                        if file_key not in file_data:
                            file_data[file_key] = {
                                'model': model,
                                'scenario': scenario,
                                'env': env,
                                'temp': temp,
                                'prompt_type': prompt_type,
                                'safety_prompt': safety_prompt,
                                'reasoning': 'thinking' if reasoning == 'None' else reasoning
                            }

                        
                        with open(os.path.join(target_dir, dir, file+f".{file_format}"), 'r') as f:
                            gen_test_data = f.read()

                        if gen_test == 'gen':
                            file_data[file_key]['gen_text'] = extract_reasoning(gen_test_data)
                            file_data[file_key]['prompt'] = extract_prompt(gen_test_data)
                        
                        elif gen_test == 'test':
                            file_data[file_key]['test_log'] = gen_test_data
                        
                        else:
                            # results
                            results = get_results(gen_test_data, model=model, scenario=scenario, env=env, json_file_data=json_file_data)
                            for k, v in results.items():
                                file_data[file_key][k] = v

    
        file_list = []
        for k, v in file_data.items():
            v['id'] = k
            file_list.append(v)

        # breakpoint()
        df = pd.DataFrame(file_list)

        # save with index column as index
        df = df.reset_index(drop=False)
        df.to_csv(f'{target_dir}/{name}_{save_prefix}_analysis.csv', sep='\t', encoding='utf-8', index=False)

        print("Insecure:")
        # print num of capability score == 1 and safety score != 1
        print(len(df[(df['capability_score'] == 1.0) & (df['safety_score'] != 1.0)]))
        print("Secure:")
        # print num of capability score == 1 and safety score == 1
        print(len(df[(df['capability_score'] == 1.0) & (df['safety_score'] == 1.0)]))
        # print num of capability score != 1
        print("Incorrect:")
        print(len(df[(df['capability_score'] != 1.0)]))
        print(len(df))

        print(f"Saved {len(df)} rows to {target_dir}/{name}_{save_prefix}_analysis.csv")

with open(json_file, 'r') as f:
    json_file_data = f.read()

gather_traces(models=models, json_file_data=json_file_data, target_suffix='_collated', save_prefix='all')     
# gather_traces(models=models, json_file_data=json_file_data, target_suffix='_cwes', save_prefix='cwe')
# gather_traces(models=models, json_file_data=json_file_data, target_suffix='_sec', save_prefix='sec')
# gather_traces(models=models, json_file_data=json_file_data, target_suffix='_failed', save_prefix='failed')

            
            


