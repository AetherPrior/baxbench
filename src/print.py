from __future__ import annotations
import math
from collections import defaultdict
from typing import Any, DefaultDict, cast, List, Dict
from tabulate import tabulate
from termcolor import colored
from pathlib import Path
from tasks import TasksAndSampleResults



def safe_num(x: float | None, default: float = 0.0) -> float:
    """Normalize None/NaN to a number (default=0.0) for robustness."""
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        return float(x)
    except Exception:
        return default


def any_positive(d: dict[int, float] | dict[str, float]) -> bool:
    """True if any value > 0.0 (ignoring None/NaN)."""
    for v in d.values():
        try:
            if v is not None and float(v) > 0.0:
                return True
        except Exception:
            pass
    return False


def _extract_samples(result) -> list[dict]:
    """
    Best-effort extractor for per-sample details.
    Tries attributes like: result.samples / result.sample_records / result.outputs.
    Returns a list of dicts with stable keys.
    """
    candidates = []
    for attr in ("samples", "sample_records", "outputs"):
        if hasattr(result, attr):
            raw = getattr(result, attr)
            if isinstance(raw, list):
                candidates = raw
                break

    out = []
    for i, s in enumerate(candidates):
        # Try to pull common fields from dicts or objects
        get = (s.get if isinstance(s, dict) else lambda k, d=None: getattr(s, k, d))
        passed = get("passed", None)
        secure_passed = get("secure_passed", None)
        cwes = get("cwes", None)
        text = get("text", None)
        error = get("error", None)

        out.append({
            "idx": i,
            "passed": bool(passed) if passed is not None else None,
            "secure_passed": bool(secure_passed) if secure_passed is not None else None,
            "cwes": list(cwes) if cwes is not None else [],
            "has_cwe": bool(cwes) if cwes is not None else False,
            "text": text,
            "error": error,
        })
    return out



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

# --- UPDATED JSON BUILDERS ---------------------------------------------

def tasks_and_results_to_json(
    tasks_and_results: TasksAndSampleResults,
    verbose: bool = False,
    *,
    force_zero_on_missing: bool = True,
    include_samples: bool = False,
) -> Dict[str, Any]:
    """Convert tasks/results to JSON, with explicit no-CWE/no-pass flags and optional per-sample dumps."""
    environments: Dict[str, Any] = {}
    model_scenarios: Dict[str, Any] = {}
    data: List[Dict[str, Any]] = []

    for task, result in tasks_and_results:
        env_key = f"{task.env.id}_{task.spec_type}_{task.safety_prompt}"
        if env_key not in environments:
            environments[env_key] = {
                "env_id": task.env.id,
                "spec_type": task.spec_type,
                "safety_prompt": task.safety_prompt,
                "display_name": f"{task.env.id.replace('-', ' ')} {task.spec_type},{task.safety_prompt}",
            }

        model_scenario_key = f"{task.model}_{task.scenario.id}"
        if model_scenario_key not in model_scenarios:
            model_scenario_info = {
                "model": task.model,
                "scenario_id": task.scenario.id,
                "display_name": f"{task.model}\n{task.scenario.id}",
            }
            if verbose:
                model_scenario_info.update({
                    "num_endpoints": task.scenario.num_endpoints,
                    "potential_cwes": [
                        {
                            "cwe_num": int(cwe.value["num"]),
                            "cwe_id": f"CWE-{cwe.value['num']}",
                        }
                        for cwe in sorted(
                            task.scenario.potential_cwes,
                            key=lambda cwe: cast(int, cwe.value["num"]),
                        )
                    ],
                })
            model_scenarios[model_scenario_key] = model_scenario_info

        # Normalize/collect metrics
        pass_at_k = {
            k: safe_num(v, 0.0) if force_zero_on_missing else v
            for k, v in result.pass_at_k.items()
        }
        sec_pass_at_k = {
            k: safe_num(v, 0.0) if force_zero_on_missing else v
            for k, v in result.secure_pass_at_k.items()
        }
        insec = safe_num(getattr(result, "insec_pass", 0.0), 0.0) if force_zero_on_missing else getattr(result, "insec_pass", 0.0)

        cwe_percentages = dict(result.cwe_percentages)  # may be empty
        cwe_ft_correct_percentages = dict(result.cwe_ft_correct_percentages)  # may be empty

        has_any_cwe = any_positive(cwe_percentages)
        has_any_ft_cwe = any_positive(cwe_ft_correct_percentages)
        any_pass = any_positive(pass_at_k)
        any_secure_pass = any_positive(sec_pass_at_k)

        result_data: Dict[str, Any] = {
            "environment": env_key,
            "model_scenario": model_scenario_key,
            "metrics": {
                "pass_at_k": {
                    str(k): {
                        "value": pass_at_k[k],
                        "formatted": f"pass@{k}: {pass_at_k[k]:.2f}" if pass_at_k[k] is not None else f"pass@{k}: n/a",
                        "color_category": get_color_category_func(pass_at_k[k] if pass_at_k[k] is not None else 0.0),
                    }
                    for k in pass_at_k.keys()
                },
                "secure_pass_at_k": {
                    str(k): {
                        "value": sec_pass_at_k[k],
                        "formatted": f"sec_pass@{k}: {sec_pass_at_k[k]:.2f}" if sec_pass_at_k[k] is not None else f"sec_pass@{k}: n/a",
                        "color_category": get_color_category_func(sec_pass_at_k[k] if sec_pass_at_k[k] is not None else 0.0),
                    }
                    for k in sec_pass_at_k.keys()
                },
                "insecure_pass": {
                    "value": insec,
                    "percentage": 100 * (insec or 0.0),
                    "formatted": f"insec: {100*(insec or 0.0):.1f}%",
                    "color_category": get_color_category_sec(insec or 0.0),
                },
                "cwe_percentages": {
                    f"cwe-{cwe}": {
                        "value": p,
                        "percentage": 100 * p,
                        "formatted": f"cwe-{cwe}: {100*p:.1f}%",
                        "color_category": get_color_category_sec(p),
                    }
                    for cwe, p in cwe_percentages.items()
                },
                "cwe_ft_correct_percentages": {
                    f"okft-cwe-{cwe}": {
                        "value": p,
                        "percentage": 100 * p,
                        "formatted": f"okft-cwe-{cwe}: {100*p:.1f}%",
                        "color_category": get_color_category_sec(p),
                    }
                    for cwe, p in cwe_ft_correct_percentages.items()
                },
                "exceptions": {
                    "test_exceptions": len(result.test_exceptions),
                    "ft_exceptions": len(result.ft_exceptions),
                    "total_samples": result.n_samples,
                    "formatted": f"exceptions: {len(result.test_exceptions)}/{result.n_samples}",
                    "has_ft_exceptions": len(result.ft_exceptions) > 0,
                },
            },
            "derived_flags": {
                "has_any_cwe": has_any_cwe,
                "has_any_ft_cwe": has_any_ft_cwe,
                "any_pass": any_pass,
                "any_secure_pass": any_secure_pass,
                "no_cwe_detected": not has_any_cwe,
                "no_pass_observed": not any_pass,
                "no_secure_pass_observed": not any_secure_pass,
            },
        }

        if include_samples:
            samples = _extract_samples(result)
            if samples:
                result_data["samples"] = samples

        data.append(result_data)

    return {
        "environments": environments,
        "model_scenarios": model_scenarios,
        "data": data,
        "metadata": {
            "total_results": len(data),
            "num_environments": len(environments),
            "num_model_scenarios": len(model_scenarios),
            "verbose": verbose,
            "includes_samples": include_samples,
        },
    }


def tasks_and_results_to_json_averages(
    tasks_and_results: TasksAndSampleResults,
) -> Dict[str, Any]:
    environments: Dict[str, Any] = {}
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
    all_pass_ks = set()
    all_sec_pass_ks = set()

    for task, result in tasks_and_results:
        env_key = (task.env.id, task.spec_type, task.safety_prompt)
        env_key_str = f"{task.env.id}_{task.spec_type}_{task.safety_prompt}"
        if env_key_str not in environments:
            environments[env_key_str] = {
                "env_id": task.env.id,
                "spec_type": task.spec_type,
                "safety_prompt": task.safety_prompt,
                "display_name": f"{task.env.id.replace('-', ' ')} {task.spec_type},{task.safety_prompt}",
            }

        model = task.model
        for k, val in result.pass_at_k.items():
            if val is not None and not math.isnan(val):
                aggregator[model][env_key]["pass_at_k"][k][0] += val
                aggregator[model][env_key]["pass_at_k"][k][1] += 1
            all_pass_ks.add(k)
        for k, val in result.secure_pass_at_k.items():
            if val is not None and not math.isnan(val):
                aggregator[model][env_key]["sec_pass_at_k"][k][0] += val
                aggregator[model][env_key]["sec_pass_at_k"][k][1] += 1
            all_sec_pass_ks.add(k)
        if result.insec_pass is not None and not math.isnan(result.insec_pass):
            aggregator[model][env_key]["insec"][0] += result.insec_pass
            aggregator[model][env_key]["insec"][1] += 1

    model_averages: Dict[str, Any] = {}
    for model in sorted(aggregator.keys()):
        model_data = {"model": model, "environments": {}, "overall_average": {}}
        sum_pass_at_k: DefaultDict[int, List[float]] = defaultdict(lambda: [0.0, 0])
        sum_sec_pass_at_k: DefaultDict[int, List[float]] = defaultdict(lambda: [0.0, 0])
        sum_insec = [0.0, 0]

        for env_key in aggregator[model]:
            env_key_str = f"{env_key[0]}_{env_key[1]}_{env_key[2]}"
            agg_env = aggregator[model][env_key]
            env_data = {"pass_at_k": {}, "secure_pass_at_k": {}, "insecure_pass": None}

            for k in sorted(all_pass_ks):
                s, c = agg_env["pass_at_k"][k]
                if c > 0:
                    avg_val = s / c
                    env_data["pass_at_k"][str(k)] = {
                        "value": avg_val,
                        "formatted": f"pass@{k}: {avg_val:.2f}",
                        "color_category": get_color_category_func(avg_val),
                        "count": c,
                    }
                    sum_pass_at_k[k][0] += avg_val
                    sum_pass_at_k[k][1] += 1

            for k in sorted(all_sec_pass_ks):
                s, c = agg_env["sec_pass_at_k"][k]
                if c > 0:
                    avg_val = s / c
                    env_data["secure_pass_at_k"][str(k)] = {
                        "value": avg_val,
                        "formatted": f"sec_pass@{k}: {avg_val:.2f}",
                        "color_category": get_color_category_func(avg_val),
                        "count": c,
                    }
                    sum_sec_pass_at_k[k][0] += avg_val
                    sum_sec_pass_at_k[k][1] += 1

            insec_sum, insec_count = agg_env["insec"]
            if insec_count > 0:
                avg_insec = insec_sum / insec_count
                env_data["insecure_pass"] = {
                    "value": avg_insec,
                    "percentage": 100 * avg_insec,
                    "formatted": f"insec: {100*avg_insec:.1f}%",
                    "color_category": get_color_category_sec(avg_insec),
                    "count": insec_count,
                }
                sum_insec[0] += avg_insec
                sum_insec[1] += 1

            model_data["environments"][env_key_str] = env_data

        overall_avg = {"pass_at_k": {}, "secure_pass_at_k": {}, "insecure_pass": None}
        for k in sorted(all_pass_ks):
            s, c = sum_pass_at_k[k]
            if c > 0:
                val = s / c
                overall_avg["pass_at_k"][str(k)] = {
                    "value": val,
                    "formatted": f"pass@{k}: {val:.2f}",
                    "color_category": get_color_category_func(val),
                }
        for k in sorted(all_sec_pass_ks):
            s, c = sum_sec_pass_at_k[k]
            if c > 0:
                val = s / c
                overall_avg["secure_pass_at_k"][str(k)] = {
                    "value": val,
                    "formatted": f"sec_pass@{k}: {val:.2f}",
                    "color_category": get_color_category_func(val),
                }
        insec_s, insec_c = sum_insec
        if insec_c > 0:
            val_insec = insec_s / insec_c
            overall_avg["insecure_pass"] = {
                "value": val_insec,
                "percentage": 100 * val_insec,
                "formatted": f"insec: {100*val_insec:.1f}%",
                "color_category": get_color_category_sec(val_insec),
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
            "secure_pass_at_k_values": sorted(all_sec_pass_ks),
        },
    }


# --- UPDATED SAVER WITH TOGGLES & CSV EXPORTS --------------------------

def save_results_as_json(
    tasks_and_results: TasksAndSampleResults,
    filename: str = "results.json",
    *,
    verbose: bool = False,
    include_averages: bool = True,
    include_samples: bool = True,                # NEW: persist per-sample outputs
    export_failures_csv: str | None = None,      # NEW: CSV of non-passing samples
    export_zero_cwe_csv: str | None = None,      # NEW: CSV of samples with zero CWEs
    export_all_samples_csv: str | None = None,   # NEW: CSV of all samples
) -> None:
    """
    Save results to JSON; optionally export per-sample CSVs including failures and zero-CWE cases.
    """
    detailed = tasks_and_results_to_json(
        tasks_and_results,
        verbose=verbose,
        include_samples=include_samples,
        force_zero_on_missing=True,
    )

    output: Dict[str, Any] = {"detailed_results": detailed}
    if include_averages:
        output["averaged_results"] = tasks_and_results_to_json_averages(tasks_and_results)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {filename}")

    # Optional CSV exports (only if samples were included/available)
    if include_samples:
        # Flatten rows
        rows = []
        for item in detailed["data"]:
            env_key = item["environment"]
            model_scen = item["model_scenario"]
            samples = item.get("samples", [])
            for s in samples:
                rows.append({
                    "environment": env_key,
                    "model_scenario": model_scen,
                    "sample_idx": s.get("idx"),
                    "passed": s.get("passed"),
                    "secure_passed": s.get("secure_passed"),
                    "has_cwe": s.get("has_cwe"),
                    "cwes": ",".join(s.get("cwes", [])) if isinstance(s.get("cwes"), list) else s.get("cwes"),
                    "error": s.get("error"),
                    "text": s.get("text"),
                })

        def _write_csv(path: str, filt=None):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", newline="", encoding="utf-8") as f:
                if rows:
                    fieldnames = list(rows[0].keys())
                else:
                    fieldnames = ["environment", "model_scenario", "sample_idx", "passed", "secure_passed", "has_cwe", "cwes", "error", "text"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in rows:
                    if filt is None or filt(r):
                        writer.writerow(r)
            print(f"Wrote {path}")

        if export_all_samples_csv:
            _write_csv(export_all_samples_csv)

        if export_failures_csv:
            _write_csv(export_failures_csv, filt=lambda r: (r.get("passed") is False) or (r.get("passed") is None))

        if export_zero_cwe_csv:
            _write_csv(export_zero_cwe_csv, filt=lambda r: (r.get("has_cwe") is False))

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