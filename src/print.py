import math
from collections import defaultdict
from typing import Any, DefaultDict, cast

from tabulate import tabulate
from termcolor import colored

from tasks import TasksAndSampleResults


def color_cyan(text: str) -> str:
    return colored(text, "cyan")


def color_blue(text: str, cond: bool) -> str:
    return colored(text, "blue") if cond else text


def color_sec(text: str, num: float) -> str:
    if num == 0:
        return colored(text, "green")
    elif num < 0.5:
        return colored(text, "yellow")
    else:
        return colored(text, "red")


def color_func(text: str, num: float) -> str:
    if num > 0.75:
        return colored(text, "green")
    elif num < 0.75 and num > 0.25:
        return colored(text, "yellow")
    else:
        return colored(text, "red")


def tasks_and_results_to_table(
    tasks_and_results: TasksAndSampleResults, verbose: bool = False
) -> str:
    env_ids: dict[tuple[str, str, str], int] = {}
    model_and_scenario_ids: dict[str, int] = {}
    cells: dict[tuple[int, int], str] = {}
    for task, result in tasks_and_results:
        col_id = env_ids.setdefault(
            (task.env.id, task.spec_type, task.safety_prompt), len(env_ids)
        )
        model_and_scenario_info = f"{task.model}\n{task.scenario.id}"
        row_id = model_and_scenario_ids.setdefault(
            model_and_scenario_info, len(model_and_scenario_ids)
        )
        if verbose:
            scenario_metadata = [
                f"Endpts: {task.scenario.num_endpoints}",
                f"Potential CWEs:",
            ]
            sorted_potential_cwes = sorted(
                list(task.scenario.potential_cwes),
                key=lambda cwe: cast(int, cwe.value["num"]),
            )
            for cwe in sorted_potential_cwes:
                scenario_metadata.append(f"  CWE-{cwe.value['num']}")
            scenario_metadata_str = "\n".join(
                [color_cyan(s) for s in scenario_metadata]
            )
            cells[(row_id, 0)] = model_and_scenario_info + "\n" + scenario_metadata_str
        else:
            cells[(row_id, 0)] = model_and_scenario_info

        ft = [
            color_func(f"pass@{k}: {result.pass_at_k[k]:.2f}", result.pass_at_k[k])
            for k in sorted(result.pass_at_k.keys())
        ]
        ft_secure = [
            color_func(
                f"sec_pass@{k}: {result.secure_pass_at_k[k]:.2f}",
                result.secure_pass_at_k[k],
            )
            for k in sorted(result.secure_pass_at_k.keys())
        ]
        ft_insecure = [
            color_sec(f"insec: {100*result.insec_pass:.1f}%", result.insec_pass),
        ]
        cwes = [
            color_sec(f"cwe-{cwe}: {100*p:.1f}", p)
            for cwe, p in result.cwe_percentages.items()
        ]
        cwes_ft_correct = [
            color_sec(f"okft-cwe-{cwe}: {100*p:.1f}", p)
            for cwe, p in result.cwe_ft_correct_percentages.items()
        ]
        errs = [
            color_blue(
                f"exceptions: {len(result.test_exceptions)}/{result.n_samples}",
                len(result.ft_exceptions) > 0,
            ),
        ]
        cell = "\n".join(ft + ft_secure + ft_insecure + cwes + cwes_ft_correct + errs)
        cells[(row_id, col_id + 1)] = cell

    headers: list[str] = [""] + [
        f'{envid.replace("-", "\n")} {spec_type,safety_prompt}'
        for (envid, spec_type, safety_prompt), _ in sorted(
            env_ids.items(), key=lambda kv: kv[1]
        )
    ]
    table: list[list[str]] = [
        ["" for _ in range(len(env_ids) + 1)]
        for _ in range(len(model_and_scenario_ids))
    ]
    for (row_id, col_id), content in cells.items():
        table[row_id][col_id] = content
    return tabulate(table, headers, tablefmt="simple_grid")


def tasks_and_results_to_table_averages(
    tasks_and_results: TasksAndSampleResults,
) -> str:
    # Track frameworks (env/spec/safety_prompt) in a consistent order
    env_ids: dict[tuple[str, str, str], int] = {}

    aggregator: DefaultDict[str, DefaultDict[tuple[str, str, str], dict[str, Any]]] = (
        defaultdict(
            lambda: defaultdict(
                lambda: {
                    "pass_at_k": defaultdict(lambda: [0.0, 0]),
                    "sec_pass_at_k": defaultdict(lambda: [0.0, 0]),
                    "insec": [0.0, 0],
                }
            )
        )
    )

    # Collect all pass@k keys and secure_pass@k keys encountered (so we can display consistently)
    all_pass_ks = set()
    all_sec_pass_ks = set()

    # Build the aggregator and remember which frameworks (env+spec+safety) we have
    for task, result in tasks_and_results:
        env_key = (task.env.id, task.spec_type, task.safety_prompt)
        if env_key not in env_ids:
            env_ids[env_key] = len(env_ids)

        model = task.model
        # pass@k
        for k, val in result.pass_at_k.items():
            if val is not None and not math.isnan(val):
                aggregator[model][env_key]["pass_at_k"][k][0] += val
                aggregator[model][env_key]["pass_at_k"][k][1] += 1
            all_pass_ks.add(k)

        # secure_pass@k
        for k, val in result.secure_pass_at_k.items():
            if val is not None and not math.isnan(val):
                aggregator[model][env_key]["sec_pass_at_k"][k][0] += val
                aggregator[model][env_key]["sec_pass_at_k"][k][1] += 1
            all_sec_pass_ks.add(k)

        # insec
        if result.insec_pass is not None and not math.isnan(result.insec_pass):
            aggregator[model][env_key]["insec"][0] += result.insec_pass
            aggregator[model][env_key]["insec"][1] += 1

    # Prepare the headers: first column is blank (for model),
    # then one column per framework in the discovered order,
    # plus a final column "AVG" that averages across all frameworks
    sorted_env_keys = sorted(
        env_ids.items(), key=lambda kv: kv[1]
    )  # [(env_key, idx), ...]
    headers = (
        [""]
        + [f'{ek[0].replace("-", "\n")} {ek[1]},{ek[2]}' for ek, _ in sorted_env_keys]
        + ["AVG"]
    )

    # We'll construct one row per model. Each cell will show the
    # average pass@k, sec_pass@k, insec for that (model, env_key).
    # The final column will be the average over all frameworks for that model.
    table_rows = []

    for model in sorted(aggregator.keys()):
        row = [model]

        # To also compute the model-wide average (across frameworks)
        sum_pass_at_k: DefaultDict[int, list[float]] = defaultdict(lambda: [0.0, 0])
        sum_sec_pass_at_k: DefaultDict[int, list[float]] = defaultdict(lambda: [0.0, 0])
        sum_insec = [0.0, 0]

        # Build a cell for each framework
        for env_key, _ in sorted_env_keys:
            agg_env = aggregator[model][env_key]

            # Compute the averaged values for pass@k
            env_pass_lines = []
            for k in sorted(all_pass_ks):
                s, c = agg_env["pass_at_k"][k]
                if c > 0:
                    avg_val = s / c
                    env_pass_lines.append(
                        color_func(f"pass@{k}: {avg_val:.2f}", avg_val)
                    )
                    # Accumulate for final column
                    sum_pass_at_k[k][0] += avg_val
                    sum_pass_at_k[k][1] += 1
                else:
                    # no data for that pass@k
                    pass

            # Compute the averaged values for sec_pass@k
            env_sec_lines = []
            for k in sorted(all_sec_pass_ks):
                s, c = agg_env["sec_pass_at_k"][k]
                if c > 0:
                    avg_val = s / c
                    env_sec_lines.append(
                        color_func(f"sec_pass@{k}: {avg_val:.2f}", avg_val)
                    )
                    # Accumulate for final column
                    sum_sec_pass_at_k[k][0] += avg_val
                    sum_sec_pass_at_k[k][1] += 1

            # Compute the averaged value for insec
            insec_sum, insec_count = agg_env["insec"]
            env_insec_line = ""
            if insec_count > 0:
                avg_insec = insec_sum / insec_count
                env_insec_line = color_sec(f"insec: {100*avg_insec:.1f}%", avg_insec)
                sum_insec[0] += avg_insec
                sum_insec[1] += 1

            cell_contents = env_pass_lines + env_sec_lines
            if env_insec_line:
                cell_contents.append(env_insec_line)

            # Join them with newlines for a multi-line cell
            row.append("\n".join(cell_contents) if cell_contents else "")

        # Now build the model-wide average cell (across frameworks)
        avg_cell_lines = []

        # pass@k
        for k in sorted(all_pass_ks):
            s, c = sum_pass_at_k[k]
            if c > 0:
                val = s / c
                avg_cell_lines.append(color_func(f"pass@{k}: {val:.2f}", val))

        # secure_pass@k
        for k in sorted(all_sec_pass_ks):
            s, c = sum_sec_pass_at_k[k]
            if c > 0:
                val = s / c
                avg_cell_lines.append(color_func(f"sec_pass@{k}: {val:.2f}", val))

        # insec
        insec_s, insec_c = sum_insec
        if insec_c > 0:
            val_insec = insec_s / insec_c
            avg_cell_lines.append(color_sec(f"insec: {100*val_insec:.1f}%", val_insec))

        row.append("\n".join(avg_cell_lines) if avg_cell_lines else "")
        table_rows.append(row)
    return tabulate(table_rows, headers, tablefmt="simple_grid")

import json
import math
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, cast

from tasks import TasksAndSampleResults


def get_color_category_func(value: float) -> str:
    """Categorize performance values for coloring."""
    if value > 0.75:
        return "good"
    elif 0.25 <= value <= 0.75:
        return "medium"
    else:
        return "poor"


def get_color_category_sec(value: float) -> str:
    """Categorize security values for coloring."""
    if value == 0:
        return "good"
    elif value < 0.5:
        return "medium"
    else:
        return "poor"


def tasks_and_results_to_json(
    tasks_and_results: TasksAndSampleResults, verbose: bool = False
) -> Dict[str, Any]:
    """Convert tasks and results to JSON format."""
    
    # Track environments and model/scenario combinations
    environments = {}
    model_scenarios = {}
    
    # Store all data
    data = []
    
    for task, result in tasks_and_results:
        # Environment info
        env_key = f"{task.env.id}_{task.spec_type}_{task.safety_prompt}"
        if env_key not in environments:
            environments[env_key] = {
                "env_id": task.env.id,
                "spec_type": task.spec_type,
                "safety_prompt": task.safety_prompt,
                "display_name": f"{task.env.id.replace('-', ' ')} {task.spec_type},{task.safety_prompt}"
            }
        
        # Model and scenario info
        model_scenario_key = f"{task.model}_{task.scenario.id}"
        if model_scenario_key not in model_scenarios:
            model_scenario_info = {
                "model": task.model,
                "scenario_id": task.scenario.id,
                "display_name": f"{task.model}\n{task.scenario.id}"
            }
            
            if verbose:
                model_scenario_info.update({
                    "num_endpoints": task.scenario.num_endpoints,
                    "potential_cwes": [
                        {
                            "cwe_num": int(cwe.value["num"]),
                            "cwe_id": f"CWE-{cwe.value['num']}"
                        }
                        for cwe in sorted(
                            task.scenario.potential_cwes,
                            key=lambda cwe: cast(int, cwe.value["num"])
                        )
                    ]
                })
            
            model_scenarios[model_scenario_key] = model_scenario_info
        
        # Results data
        result_data = {
            "environment": env_key,
            "model_scenario": model_scenario_key,
            "metrics": {
                "pass_at_k": {
                    str(k): {
                        "value": v,
                        "formatted": f"pass@{k}: {v:.2f}",
                        "color_category": get_color_category_func(v)
                    }
                    for k, v in result.pass_at_k.items()
                },
                "secure_pass_at_k": {
                    str(k): {
                        "value": v,
                        "formatted": f"sec_pass@{k}: {v:.2f}",
                        "color_category": get_color_category_func(v)
                    }
                    for k, v in result.secure_pass_at_k.items()
                },
                "insecure_pass": {
                    "value": result.insec_pass,
                    "percentage": 100 * result.insec_pass,
                    "formatted": f"insec: {100*result.insec_pass:.1f}%",
                    "color_category": get_color_category_sec(result.insec_pass)
                },
                "cwe_percentages": {
                    f"cwe-{cwe}": {
                        "value": p,
                        "percentage": 100 * p,
                        "formatted": f"cwe-{cwe}: {100*p:.1f}%",
                        "color_category": get_color_category_sec(p)
                    }
                    for cwe, p in result.cwe_percentages.items()
                },
                "cwe_ft_correct_percentages": {
                    f"okft-cwe-{cwe}": {
                        "value": p,
                        "percentage": 100 * p,
                        "formatted": f"okft-cwe-{cwe}: {100*p:.1f}%",
                        "color_category": get_color_category_sec(p)
                    }
                    for cwe, p in result.cwe_ft_correct_percentages.items()
                },
                "exceptions": {
                    "test_exceptions": len(result.test_exceptions),
                    "ft_exceptions": len(result.ft_exceptions),
                    "total_samples": result.n_samples,
                    "formatted": f"exceptions: {len(result.test_exceptions)}/{result.n_samples}",
                    "has_ft_exceptions": len(result.ft_exceptions) > 0
                }
            }
        }
        
        data.append(result_data)
    
    return {
        "environments": environments,
        "model_scenarios": model_scenarios,
        "data": data,
        "metadata": {
            "total_results": len(data),
            "num_environments": len(environments),
            "num_model_scenarios": len(model_scenarios),
            "verbose": verbose
        }
    }


def tasks_and_results_to_json_averages(
    tasks_and_results: TasksAndSampleResults,
) -> Dict[str, Any]:
    """Convert tasks and results to JSON format with averages."""
    
    # Track frameworks (env/spec/safety_prompt) in a consistent order
    environments = {}
    
    aggregator: DefaultDict[str, DefaultDict[tuple[str, str, str], dict[str, Any]]] = (
        defaultdict(
            lambda: defaultdict(
                lambda: {
                    "pass_at_k": defaultdict(lambda: [0.0, 0]),
                    "sec_pass_at_k": defaultdict(lambda: [0.0, 0]),
                    "insec": [0.0, 0],
                }
            )
        )
    )
    
    # Collect all pass@k keys
    all_pass_ks = set()
    all_sec_pass_ks = set()
    
    # Build the aggregator
    for task, result in tasks_and_results:
        env_key = (task.env.id, task.spec_type, task.safety_prompt)
        env_key_str = f"{task.env.id}_{task.spec_type}_{task.safety_prompt}"
        
        if env_key_str not in environments:
            environments[env_key_str] = {
                "env_id": task.env.id,
                "spec_type": task.spec_type,
                "safety_prompt": task.safety_prompt,
                "display_name": f"{task.env.id.replace('-', ' ')} {task.spec_type},{task.safety_prompt}"
            }
        
        model = task.model
        
        # pass@k
        for k, val in result.pass_at_k.items():
            if val is not None and not math.isnan(val):
                aggregator[model][env_key]["pass_at_k"][k][0] += val
                aggregator[model][env_key]["pass_at_k"][k][1] += 1
            all_pass_ks.add(k)
        
        # secure_pass@k
        for k, val in result.secure_pass_at_k.items():
            if val is not None and not math.isnan(val):
                aggregator[model][env_key]["sec_pass_at_k"][k][0] += val
                aggregator[model][env_key]["sec_pass_at_k"][k][1] += 1
            all_sec_pass_ks.add(k)
        
        # insec
        if result.insec_pass is not None and not math.isnan(result.insec_pass):
            aggregator[model][env_key]["insec"][0] += result.insec_pass
            aggregator[model][env_key]["insec"][1] += 1
    
    # Process aggregated data
    model_averages = {}
    
    for model in sorted(aggregator.keys()):
        model_data = {
            "model": model,
            "environments": {},
            "overall_average": {}
        }
        
        # For computing model-wide averages
        sum_pass_at_k: DefaultDict[int, List[float]] = defaultdict(lambda: [0.0, 0])
        sum_sec_pass_at_k: DefaultDict[int, List[float]] = defaultdict(lambda: [0.0, 0])
        sum_insec = [0.0, 0]
        
        # Process each environment
        for env_key in aggregator[model]:
            env_key_str = f"{env_key[0]}_{env_key[1]}_{env_key[2]}"
            agg_env = aggregator[model][env_key]
            
            env_data = {
                "pass_at_k": {},
                "secure_pass_at_k": {},
                "insecure_pass": None
            }
            
            # pass@k averages
            for k in sorted(all_pass_ks):
                s, c = agg_env["pass_at_k"][k]
                if c > 0:
                    avg_val = s / c
                    env_data["pass_at_k"][str(k)] = {
                        "value": avg_val,
                        "formatted": f"pass@{k}: {avg_val:.2f}",
                        "color_category": get_color_category_func(avg_val),
                        "count": c
                    }
                    sum_pass_at_k[k][0] += avg_val
                    sum_pass_at_k[k][1] += 1
            
            # secure_pass@k averages
            for k in sorted(all_sec_pass_ks):
                s, c = agg_env["sec_pass_at_k"][k]
                if c > 0:
                    avg_val = s / c
                    env_data["secure_pass_at_k"][str(k)] = {
                        "value": avg_val,
                        "formatted": f"sec_pass@{k}: {avg_val:.2f}",
                        "color_category": get_color_category_func(avg_val),
                        "count": c
                    }
                    sum_sec_pass_at_k[k][0] += avg_val
                    sum_sec_pass_at_k[k][1] += 1
            
            # insec average
            insec_sum, insec_count = agg_env["insec"]
            if insec_count > 0:
                avg_insec = insec_sum / insec_count
                env_data["insecure_pass"] = {
                    "value": avg_insec,
                    "percentage": 100 * avg_insec,
                    "formatted": f"insec: {100*avg_insec:.1f}%",
                    "color_category": get_color_category_sec(avg_insec),
                    "count": insec_count
                }
                sum_insec[0] += avg_insec
                sum_insec[1] += 1
            
            model_data["environments"][env_key_str] = env_data
        
        # Compute overall model average
        overall_avg = {
            "pass_at_k": {},
            "secure_pass_at_k": {},
            "insecure_pass": None
        }
        
        # Overall pass@k
        for k in sorted(all_pass_ks):
            s, c = sum_pass_at_k[k]
            if c > 0:
                val = s / c
                overall_avg["pass_at_k"][str(k)] = {
                    "value": val,
                    "formatted": f"pass@{k}: {val:.2f}",
                    "color_category": get_color_category_func(val)
                }
        
        # Overall secure_pass@k
        for k in sorted(all_sec_pass_ks):
            s, c = sum_sec_pass_at_k[k]
            if c > 0:
                val = s / c
                overall_avg["secure_pass_at_k"][str(k)] = {
                    "value": val,
                    "formatted": f"sec_pass@{k}: {val:.2f}",
                    "color_category": get_color_category_func(val)
                }
        
        # Overall insec
        insec_s, insec_c = sum_insec
        if insec_c > 0:
            val_insec = insec_s / insec_c
            overall_avg["insecure_pass"] = {
                "value": val_insec,
                "percentage": 100 * val_insec,
                "formatted": f"insec: {100*val_insec:.1f}%",
                "color_category": get_color_category_sec(val_insec)
            }
        
        model_data["overall_average"] = overall_avg
        model_averages[model] = model_data
    
    return {
        "environments": environments,
        "model_averages": model_averages,
        "metadata": {
            "num_models": len(model_averages),
            "num_environments": len(environments),
            "pass_at_k_values": sorted(all_pass_ks),
            "secure_pass_at_k_values": sorted(all_sec_pass_ks)
        }
    }


def save_results_as_json(
    tasks_and_results: TasksAndSampleResults, 
    filename: str = "results.json", 
    verbose: bool = False,
    include_averages: bool = True
) -> None:
    """Save results to JSON file."""
    
    output = {
        "detailed_results": tasks_and_results_to_json(tasks_and_results, verbose)
    }
    
    if include_averages:
        output["averaged_results"] = tasks_and_results_to_json_averages(tasks_and_results)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {filename}")


# Example usage functions
def print_json_summary(json_data: Dict[str, Any]) -> None:
    """Print a summary of the JSON data structure."""
    print("JSON Structure Summary:")
    print("=" * 50)
    
    if "detailed_results" in json_data:
        detailed = json_data["detailed_results"]
        print(f"Detailed Results:")
        print(f"  - Total results: {detailed['metadata']['total_results']}")
        print(f"  - Environments: {detailed['metadata']['num_environments']}")
        print(f"  - Model scenarios: {detailed['metadata']['num_model_scenarios']}")
    
    if "averaged_results" in json_data:
        averaged = json_data["averaged_results"]
        print(f"Averaged Results:")
        print(f"  - Models: {averaged['metadata']['num_models']}")
        print(f"  - Environments: {averaged['metadata']['num_environments']}")
        print(f"  - Pass@k values: {averaged['metadata']['pass_at_k_values']}")