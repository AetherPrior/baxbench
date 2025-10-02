#!/usr/bin/env python3
"""
CWE Log File Organizer with Plotting + CSV Export

This script processes model evaluation results and:
- organizes log files by CWE / security buckets
- generates visualization plots (optional)
- exports a CSV with one row per scenario instance capturing logs and reasoning trace

Usage:
    python cwe_organizer.py <json_file> <temperature> [options]

Examples:
    python cwe_organizer.py overall_results_gpt-oss:20b.json 0.001 --plot -s ./results
    python cwe_organizer.py overall_results_gpt-oss:20b.json 0.001 --csv
    python cwe_organizer.py overall_results_gpt-oss:20b.json 0.001 --csv --csv-path ./out/my.csv
"""

import argparse
import csv  # === CSV EXPORT ADDITIONS ===
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re


def extract_model_name(json_filename: str) -> str:
    """Extract model name from JSON filename."""
    if json_filename.startswith("overall_results_"):
        return json_filename[len("overall_results_"):-5]  # Remove prefix and .json
    return json_filename[:-5]  # Just remove .json


def parse_environment_config(env_key: str) -> Dict[str, str]:
    """Parse environment key to extract configuration details."""
    # Expected format: "Python-Django_openapi_specific"
    parts = env_key.split('_')
    if 'Go-net' in parts[0]:
        # make sure slashes are escaped
        parts[0] = parts[0].replace('/','-')
    return {
        'env_id': parts[0],  # e.g., "Python-Django"
        'spec_type': parts[1] if len(parts) >= 2 else 'openapi',  # e.g., "openapi"
        'safety_prompt': parts[2] if len(parts) >= 3 else 'specific'  # e.g., "specific"
    }


def find_cwe_scenarios(data: Dict, model: str) -> List[Dict]:
    """Classify scenarios into CWE-hit, secure (no CWE + some pass), and failed (no pass)."""
    cwe_scenarios = []
    non_cwe_scenarios = []
    failed_scenarios = []

    for result in data['detailed_results']['data']:
        env_config = parse_environment_config(result['environment'])
        scenario_parts = result['model_scenario'].split('_')
        # strip difficulty token if present
        if len(scenario_parts) >= 3 and scenario_parts[1] in {'low', 'medium', 'med', 'high'}:
            scenario_parts = [scenario_parts[0]] + scenario_parts[2:]
        scenario = '_'.join(scenario_parts[1:])

        metrics = result.get('metrics', {})
        flags = result.get('derived_flags', {})

        # Prefer explicit flags if present
        has_any_cwe = bool(flags.get('has_any_cwe', False))
        any_pass = bool(flags.get('any_pass', False))
        any_secure_pass = bool(flags.get('any_secure_pass', False))

        # Fallback if flags are absent
        if not flags:
            cwe_percentages = metrics.get('cwe_percentages', {})
            pass_at_k = metrics.get('pass_at_k', {})
            sec_pass_at_k = metrics.get('secure_pass_at_k', {})

            def _any_val_gt0(d):
                return any((v or {}).get('value', 0) > 0 for v in d.values())

            has_any_cwe = any((v or {}).get('value', 0) > 0 for v in cwe_percentages.values())
            any_pass = _any_val_gt0(pass_at_k)
            any_secure_pass = _any_val_gt0(sec_pass_at_k)

        base = {
            'model': model,
            'scenario': scenario,
            'env_id': env_config['env_id'],
            'spec_type': env_config['spec_type'],
            'safety_prompt': env_config['safety_prompt'],
        }

        if any_pass and has_any_cwe:
            # collect all CWEs present (not just one)
            cwes = [c for c, v in metrics.get('cwe_percentages', {}).items() if (v or {}).get('value', 0) > 0]
            entry = dict(base)
            entry['cwes'] = cwes
            cwe_scenarios.append(entry)
        elif any_pass and not has_any_cwe:
            non_cwe_scenarios.append(base)
        else:
            failed_scenarios.append(base)

    return cwe_scenarios, non_cwe_scenarios, failed_scenarios

def construct_source_paths(base_dir: str, scenario_info: Dict, temperature: str, model: str) -> Dict[str, Path]:
    """Construct source file paths based on the folder structure."""
    scenario = scenario_info['scenario']
    env_id = scenario_info['env_id']
    spec_type = scenario_info['spec_type']
    safety_prompt = scenario_info['safety_prompt']
    
    # Create temperature string (keep at least 1 decimal place)
    temp_str = temperature # modified to prevent decimal shenanigans
    
    # Construct the path components
    folder_name = f"temp{temp_str}-{spec_type}-{safety_prompt}"
    
    scenario_dir = Path(base_dir) / model / scenario / env_id / folder_name
    
    return {
        'gen_log': scenario_dir / "gen.log",
        'test_log': scenario_dir / "sample0" / "test.log"
    }


def construct_target_filename(scenario_info: Dict, log_type: str, temperature: str) -> str:
    """Construct target filename for the organized logs."""

    scenario = scenario_info['scenario']
    env_id = scenario_info['env_id'].replace('-', '')  # e.g., "Python-Flask" -> "PythonFlask"
    temp_str = temperature
    spec_type = scenario_info['spec_type']
    safety_prompt = scenario_info['safety_prompt']
    try:
        cwe = scenario_info['cwe']    
        return f"{cwe}_{log_type}_{scenario}_{env_id}_{temp_str}_{spec_type}_{safety_prompt}.log"
    except:
        return f"{log_type}_{scenario}_{env_id}_{temp_str}_{spec_type}_{safety_prompt}.log"
    
            

def esc_model_name(text: str): 
    return re.sub(r'_|/|:', '-', text)

def plot_model_results(data: Dict, model_name: str, output_dir: Path):
    """Create comprehensive plots for model evaluation results."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Extract averaged results
    averaged_results = data.get('averaged_results', {})
    model_averages = averaged_results.get('model_averages', {})
    
    # Get the model data (assuming single model)
    model_data = None
    for model_key, model_info in model_averages.items():
        if esc_model_name(model_key) in esc_model_name(model_name):
            model_data = model_info
            break
    
    if not model_data:
        print("Warning: Could not find model data for plotting")
        return
    
    environments = model_data.get('environments', {})
    overall_average = model_data.get('overall_average', {})
    
    # Prepare data for plotting
    env_names = []
    pass_at_k_values = []
    sec_pass_at_k_values = []
    insecure_pass_values = []
    
    for env_key, env_data in environments.items():
        # Clean up environment name for display
        env_display = env_key.replace('_openapi_specific', '').replace('-', ' ')
        env_names.append(env_display)
        pass_at_k_values.append(env_data.get('pass_at_k', {}).get('1', {}).get('value', 0) * 100)
        sec_pass_at_k_values.append(env_data.get('secure_pass_at_k', {}).get('1', {}).get('value', 0) * 100)
        insecure_pass = env_data.get('insecure_pass', {})
        insecure_pass_values.append(0 if insecure_pass is None else insecure_pass.get('percentage',0))
    
    # Create the main plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Model Evaluation Results: {esc_model_name(model_name)}', fontsize=16, fontweight='bold')
    
    # Plot 1: Pass@1 vs Secure Pass@1
    x = np.arange(len(env_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, pass_at_k_values, width, label='Pass@1', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x + width/2, sec_pass_at_k_values, width, label='Secure Pass@1', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('Environment')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Pass@1 vs Secure Pass@1 by Environment')
    ax1.set_xticks(x)
    ax1.set_xticklabels(env_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Insecure Pass Percentage
    bars3 = ax2.bar(env_names, insecure_pass_values, alpha=0.8, color='orange')
    ax2.set_xlabel('Environment')
    ax2.set_ylabel('Insecure Pass (%)')
    ax2.set_title('Insecure Pass Percentage by Environment')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(insecure_pass_values + [1]) * 1.1)
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        if height > 0:
            ax2.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Overall Averages
    overall_metrics = ['Pass@1', 'Secure Pass@1', 'Insecure Pass percent']
    overall_values = [
        overall_average.get('pass_at_k', {}).get('1', {}).get('value', 0) * 100,
        overall_average.get('secure_pass_at_k', {}).get('1', {}).get('value', 0) * 100,
        overall_average.get('insecure_pass', {}).get('percentage', 0)
    ]
    
    colors = ['skyblue', 'lightcoral', 'orange']
    bars4 = ax3.bar(overall_metrics, overall_values, alpha=0.8, color=colors)
    ax3.set_ylabel('Percentage (%)')
    ax3.set_title('Overall Average Performance')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, max(overall_values + [1]) * 1.1)
    
    # Add value labels
    for bar in bars4:
        height = bar.get_height()
        ax3.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 4: Security Gap Analysis
    security_gaps = [p - s for p, s in zip(pass_at_k_values, sec_pass_at_k_values)]
    bars5 = ax4.bar(env_names, security_gaps, alpha=0.8, color='red')
    ax4.set_xlabel('Environment')
    ax4.set_ylabel('Security Gap (%)')
    ax4.set_title('Security Gap (Pass@1 - Secure Pass@1)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels
    for bar in bars5:
        height = bar.get_height()
        if abs(height) > 0.1:
            ax4.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -15),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = output_dir / f"{esc_model_name(model_name)}_evaluation_results.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {plot_path}")
    
    # Create a CWE-specific plot if CWE data exists
    plot_cwe_vulnerabilities(data, model_name, output_dir)
    
    plt.show()


def plot_cwe_vulnerabilities(data: Dict, model_name: str, output_dir: Path):
    """Create a plot showing CWE vulnerability distribution."""
    
    # Extract CWE data from detailed results
    cwe_counts = {}
    cwe_by_env = {}
    for result in data['detailed_results']['data']:
        env_config = parse_environment_config(result['environment'])
        env_display = env_config['env_id'].replace('-', ' ')
        
        cwe_percentages = result['metrics'].get('cwe_percentages', {})
        func_correct = result['metrics'].get('pass_at_k', {})
        for (k, corr), (cwe, cwe_data) in zip( func_correct.items(), cwe_percentages.items()):
            if cwe_data['value'] > 0 and corr['value'] > 0:
                # Count overall CWE occurrences
                if cwe not in cwe_counts:
                    cwe_counts[cwe] = 0
                cwe_counts[cwe] += 1
                
                # Track by environment
                if env_display not in cwe_by_env:
                    cwe_by_env[env_display] = {}
                if cwe not in cwe_by_env[env_display]:
                    cwe_by_env[env_display][cwe] = 0
                cwe_by_env[env_display][cwe] += 1
                
    
    if not cwe_counts:
        print("No CWE vulnerabilities found for plotting")
        return
    
    # Create CWE visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'CWE Vulnerability Analysis: {esc_model_name(model_name)}', fontsize=14, fontweight='bold')
    
    # Plot 1: Overall CWE distribution
    cwes = list(cwe_counts.keys())
    counts = list(cwe_counts.values())
    
    bars = ax1.bar(cwes, counts, alpha=0.8, color='red')
    ax1.set_xlabel('CWE Type')
    ax1.set_ylabel('Number of Occurrences')
    ax1.set_title('CWE Vulnerability Distribution')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    # Plot 2: CWE by environment heatmap
    if cwe_by_env:
        # Prepare data for heatmap
        all_cwes = sorted(set(cwe for env_cwes in cwe_by_env.values() for cwe in env_cwes.keys()))
        all_envs = sorted(cwe_by_env.keys())
        
        heatmap_data = np.zeros((len(all_envs), len(all_cwes)))
        
        for i, env in enumerate(all_envs):
            for j, cwe in enumerate(all_cwes):
                heatmap_data[i, j] = cwe_by_env[env].get(cwe, 0)
        
        im = ax2.imshow(heatmap_data, cmap='Reds', aspect='auto')
        ax2.set_xticks(range(len(all_cwes)))
        ax2.set_yticks(range(len(all_envs)))
        ax2.set_xticklabels(all_cwes, rotation=45, ha='right')
        ax2.set_yticklabels(all_envs)
        ax2.set_title('CWE Distribution by Environment')
        
        # Add text annotations
        for i in range(len(all_envs)):
            for j in range(len(all_cwes)):
                value = int(heatmap_data[i, j])
                if value > 0:
                    ax2.text(j, i, str(value), ha='center', va='center', 
                            color='white' if value > heatmap_data.max()/2 else 'black',
                            fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Number of Vulnerabilities')
    
    plt.tight_layout()
    
    # Save the CWE plot
    cwe_plot_path = output_dir / f"{esc_model_name(model_name)}_cwe_analysis.png"
    plt.savefig(cwe_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved CWE plot to: {cwe_plot_path}")
    
    plt.show()


# === CSV EXPORT ADDITIONS ===
RE_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)

def extract_reasoning_trace_from_text(gen_text: str) -> str:
    """Return first <think>...</think> block contents, stripped. Empty string if none."""
    if not gen_text:
        return ""
    m = RE_THINK_BLOCK.search(gen_text)
    if not m:
        return ""
    return m.group(1).strip()

def read_file_text(path: Path) -> str:
    """Read file as text if exists; return empty string if missing."""
    try:
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        pass
    return ""

def build_csv_rows_for_all_instances(
    data: Dict,
    model_name: str,
    temperature: str,
    source_dir: str
) -> List[Dict[str, str]]:
    """Build rows (one per scenario instance) with logs, reasoning trace, CWE list, and pass flag."""
    rows: List[Dict[str, str]] = []

    def _any_val_gt0(d):
        # d is expected to be like {'1': {'value': float}, ...} but be defensive
        return any((v or {}).get('value', 0) > 0 for v in (d or {}).values())

    for index, result in enumerate(data['detailed_results']['data']):
        env_config = parse_environment_config(result['environment'])
        scenario_parts = result['model_scenario'].split('_')
        if len(scenario_parts) >= 3 and scenario_parts[1] in {'low', 'medium', 'med', 'high'}:
            scenario_parts = [scenario_parts[0]] + scenario_parts[2:]
        scenario = '_'.join(scenario_parts[1:])

        scenario_info = {
            'model': model_name,
            'scenario': scenario,
            'env_id': env_config['env_id'],
            'spec_type': env_config['spec_type'],
            'safety_prompt': env_config['safety_prompt'],
        }

        paths = construct_source_paths(source_dir, scenario_info, temperature, model_name)
        gen_text = read_file_text(paths['gen_log'])
        test_text = read_file_text(paths['test_log'])
        reasoning = extract_reasoning_trace_from_text(gen_text)

        # --- NEW: derive pass flag and CWEs (re-using logic consistent with find_cwe_scenarios) ---
        metrics = result.get('metrics', {}) or {}
        flags = result.get('derived_flags', {}) or {}

        # Prefer explicit flags if present
        any_pass = bool(flags.get('any_pass', False))

        # Fallback if flags are absent
        if not flags:
            any_pass = _any_val_gt0(metrics.get('pass_at_k', {}))

        # Collect all CWEs with value > 0 (if present)
        cwe_percentages = metrics.get('cwe_percentages', {}) or {}
        cwes_present = [cwe for cwe, v in cwe_percentages.items() if (v or {}).get('value', 0) > 0]
        cwe_str = ';'.join(cwes_present) if cwes_present else ""

        rows.append({
            'index': index,
            'model': model_name,
            'env': scenario_info['env_id'],
            'scenario': scenario_info['scenario'],
            'spec_type': scenario_info['spec_type'],
            'granularity': scenario_info['safety_prompt'],  # maps to requested granularity
            'temperature': str(temperature),
            'generation_log': gen_text,
            'test_log': test_text,
            'reasoning_trace': reasoning,
            'cwe': cwe_str,           # --- NEW ---
            'passed': any_pass,       # --- NEW ---
        })
    return rows


def write_csv(rows: List[Dict[str, str]], out_path: Path) -> None:
    """Write rows to CSV with utf-8 encoding and robust quoting."""
    if not rows:
        print("Warning: No rows to write for CSV export.")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'index', 'model', 'env', 'scenario', 'spec_type', 'granularity',
        'temperature', 'generation_log', 'test_log', 'reasoning_trace',
        'cwe', 'passed'  # --- NEW ---
    ]
    with out_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"CSV saved to: {out_path.resolve()}")

def parse_arguments():
    """Parse command line arguments using argparse."""
    parser = argparse.ArgumentParser(
        description="Organize CWE log files from model evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        'json_file',
        help='Path to the JSON results file (e.g., overall_results_gpt-oss:20b.json)'
    )
    parser.add_argument(
        'temperature',
        type=str,
        help='Temperature value used in the experiments (e.g., 0.0, 0.5, 1.0)'
    )
    parser.add_argument(
        '-s', '--source-dir',
        default='.',
        help='Root directory containing the model results (default: current directory)'
    )
    parser.add_argument(
        '--dry-run', action='store_true', help='Show what would be done without copying files'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='Enable verbose output'
    )
    parser.add_argument(
        '--plot', action='store_true', help='Generate plots for model evaluation results'
    )
    parser.add_argument(
        '--plot-only', action='store_true', help='Only generate plots without organizing CWE logs'
    )
    parser.add_argument(
        '--csv', action='store_true', help='Export a CSV with logs and reasoning trace'
    )
    parser.add_argument(
        '--csv-path', default=None, help='Explicit CSV output path (default: <target_dir>/<model>_instances.csv)'
    )
    # === NEW: prefix for outputs ===
    parser.add_argument(
        '--target-prefix',
        default="collated_outputs",
        help='Prefix directory for organized outputs (default: "collated_outputs")'
    )
    
    return parser.parse_args()


def organize_cwe_logs_main(args):
    """Main function wrapper that uses parsed arguments."""
    
    if not os.path.exists(args.json_file):
        print(f"Error: JSON file not found: {args.json_file}")
        sys.exit(1)
    if not args.plot_only and not os.path.exists(args.source_dir):
        print(f"Error: Source directory not found: {args.source_dir}")
        sys.exit(1)
    
    try:
        with open(args.json_file, 'r') as f:
            data = json.load(f)
        
        model_name = extract_model_name(os.path.basename(args.json_file))
        if args.verbose:
            print(f"Processing model: {esc_model_name(model_name)}")
        
        # === CHANGED: prefix applied here ===
        base_out = Path(args.target_prefix)
        target_dir = base_out / f"{esc_model_name(model_name)}_cwes"
        if not args.dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
        
        if args.plot or args.plot_only:
            print("Generating plots...")
            try:
                plot_model_results(data, model_name, target_dir)
            except Exception as e:
                print(f"Plotting error: {e}")
        if args.plot_only:
            return
        
        # --- classification + processing ---
        cwe_scenarios, non_cwe_scenarios, failed_scenarios = find_cwe_scenarios(data, model_name)
        print(f"Found {len(cwe_scenarios)} CWE scenarios")
        
        def process_scenarios(scenarios, target_suffix):
            copied_count = 0
            missing_files = []
            target_dir_local = base_out / f"{esc_model_name(model_name)}_{target_suffix}"
            if not args.dry_run:
                target_dir_local.mkdir(parents=True, exist_ok=True)
            
            for scenario_info in scenarios:
                if args.verbose or args.dry_run:
                    print(f"Processing {scenario_info.get('scenario','<unknown>')} ({scenario_info.get('env_id','?')})")
                
                source_paths = construct_source_paths(args.source_dir, scenario_info, args.temperature, model_name)
                gen_source = source_paths['gen_log']
                gen_target = target_dir_local / construct_target_filename(scenario_info, 'gen', args.temperature)
                test_source = source_paths['test_log']
                test_target = target_dir_local / construct_target_filename(scenario_info, 'test', args.temperature)
                
                for src, tgt, name in [(gen_source, gen_target, "gen.log"), (test_source, test_target, "test.log")]:
                    if src.exists():
                        if args.dry_run:
                            print(f"Would copy {name} -> {tgt.name}")
                        else:
                            shutil.copy2(src, tgt)
                            if args.verbose:
                                print(f"✓ Copied {name} -> {tgt.name}")
                        copied_count += 1
                    else:
                        missing_files.append(f"{name}: {src}")
                        if args.verbose or args.dry_run:
                            print(f"✗ Missing: {src}")
            
            print(f"Summary ({target_suffix}): {copied_count} copied, {len(missing_files)} missing")
            if not args.dry_run:
                print(f"Logs in: {target_dir_local.absolute()}")
        
        process_scenarios(cwe_scenarios, 'cwes')
        process_scenarios(non_cwe_scenarios, 'sec')
        process_scenarios(failed_scenarios, 'failed')
        
        if args.csv:
            print("Building CSV...")
            rows = build_csv_rows_for_all_instances(data, model_name, args.temperature, args.source_dir)
            default_csv_path = Path(args.csv_path) if args.csv_path else (target_dir / f"{esc_model_name(model_name)}_instances.csv")
            write_csv(rows, default_csv_path)
    
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    args = parse_arguments()
    organize_cwe_logs_main(args)