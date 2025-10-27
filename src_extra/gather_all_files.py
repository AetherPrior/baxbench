#!/usr/bin/env python3
"""
CWE Log File Organizer with Plotting (Collated Mode)

This version copies gen.log, test.log, and test_results.json for *every* scenario,
ignoring CWE/non-CWE/failed buckets. Outputs go to:
  {--target-dir}/{escaped_model_name}_collated

Plotting behavior is unchanged.

Usage:
    python cwe_organizer.py <json_file> <temperature> [options]

Example:
    python cwe_organizer.py overall_results_gpt-oss:20b.json 0.001 --plot -s ./results
"""

import argparse
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
    parts = env_key.split('_')
    if 'Go-net' in parts[0]:
        parts[0] = parts[0].replace('/', '-')
    return {
        'env_id': parts[0],                         # e.g., "Python-Django"
        'spec_type': parts[1] if len(parts) >= 2 else 'openapi',
        'safety_prompt': parts[2] if len(parts) >= 3 else 'specific'
    }


def collect_all_scenarios(data: Dict, model: str) -> List[Dict]:
    """Collect scenario info for *every* detailed result (no CWE bucketing)."""
    scenarios = []
    for result in data['detailed_results']['data']:
        env_config = parse_environment_config(result['environment'])
        scenario_parts = result['model_scenario'].split('_')
        # strip difficulty token if present
        if len(scenario_parts) >= 3 and scenario_parts[1] in {'low', 'medium', 'med', 'high'}:
            scenario_parts = [scenario_parts[0]] + scenario_parts[2:]
        scenario = '_'.join(scenario_parts[1:])

        scenarios.append({
            'model': model,
            'scenario': scenario,
            'env_id': env_config['env_id'],
            'spec_type': env_config['spec_type'],
            'safety_prompt': env_config['safety_prompt'],
        })
    return scenarios


def construct_source_paths(base_dir: str, scenario_info: Dict, temperature: str, model: str) -> Dict[str, Path]:
    """Construct source file paths based on the folder structure."""
    scenario = scenario_info['scenario']
    env_id = scenario_info['env_id']
    spec_type = scenario_info['spec_type']
    safety_prompt = scenario_info['safety_prompt']
    temp_str = temperature  # keep exact string

    folder_name = f"temp{temp_str}-{spec_type}-{safety_prompt}"
    scenario_dir = Path(base_dir) / model / scenario / env_id / folder_name

    return {
        'gen_log': scenario_dir / "gen.log",
        'test_log': scenario_dir / "sample0" / "test.log",
        'results_json': scenario_dir / "sample0" / "test_results.json"
    }


def construct_target_filename(scenario_info: Dict, log_type: str, temperature: str) -> str:
    """Construct target filename for the organized logs (no CWE prefixes)."""
    scenario = scenario_info['scenario']
    env_id = scenario_info['env_id'].replace('-', '')  # e.g., "Python-Flask" -> "PythonFlask"
    temp_str = temperature
    spec_type = scenario_info['spec_type']
    safety_prompt = scenario_info['safety_prompt']

    if log_type == 'results':
        return f"{log_type}_{scenario}_{env_id}_{temp_str}_{spec_type}_{safety_prompt}.json"
    return f"{log_type}_{scenario}_{env_id}_{temp_str}_{spec_type}_{safety_prompt}.log"


def esc_model_name(text: str):
    return re.sub(r'_|/|:', '-', text)


# ----------------- Plotting (unchanged) -----------------
def plot_model_results(data: Dict, model_name: str, output_dir: Path):
    """Create comprehensive plots for model evaluation results."""
    plt.style.use('default')
    sns.set_palette("husl")

    averaged_results = data.get('averaged_results', {})
    model_averages = averaged_results.get('model_averages', {})

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

    env_names, pass_at_k_values, sec_pass_at_k_values, insecure_pass_values = [], [], [], []
    for env_key, env_data in environments.items():
        env_display = env_key.replace('_openapi_specific', '').replace('-', ' ')
        env_names.append(env_display)
        pass_at_k_values.append(env_data.get('pass_at_k', {}).get('1', {}).get('value', 0) * 100)
        sec_pass_at_k_values.append(env_data.get('secure_pass_at_k', {}).get('1', {}).get('value', 0) * 100)
        insecure_pass = env_data.get('insecure_pass', {})
        insecure_pass_values.append(0 if insecure_pass is None else insecure_pass.get('percentage', 0))

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Model Evaluation Results: {model_name}', fontsize=16, fontweight='bold')

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

    for bar in bars1 + bars2:
        h = bar.get_height()
        ax1.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8)

    bars3 = ax2.bar(env_names, insecure_pass_values, alpha=0.8, color='orange')
    ax2.set_xlabel('Environment')
    ax2.set_ylabel('Insecure Pass (%)')
    ax2.set_title('Insecure Pass Percentage by Environment')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(insecure_pass_values + [1]) * 1.1)
    for bar in bars3:
        h = bar.get_height()
        if h > 0:
            ax2.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=8)

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
    for bar in bars4:
        h = bar.get_height()
        ax3.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    security_gaps = [p - s for p, s in zip(pass_at_k_values, sec_pass_at_k_values)]
    bars5 = ax4.bar(env_names, security_gaps, alpha=0.8, color='red')
    ax4.set_xlabel('Environment')
    ax4.set_ylabel('Security Gap (%)')
    ax4.set_title('Security Gap (Pass@1 - Secure Pass@1)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    for bar in bars5:
        h = bar.get_height()
        if abs(h) > 0.1:
            ax4.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                         xytext=(0, 3 if h >= 0 else -15), textcoords="offset points",
                         ha='center', va='bottom' if h >= 0 else 'top', fontsize=8)

    plt.tight_layout()
    plot_path = output_dir / f"{model_name}_evaluation_results.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {plot_path}")

    plot_cwe_vulnerabilities(data, model_name, output_dir)
    plt.show()


def plot_cwe_vulnerabilities(data: Dict, model_name: str, output_dir: Path):
    """Create a plot showing CWE vulnerability distribution (best-effort if present)."""
    cwe_counts = {}
    cwe_by_env = {}
    breakpoint()
    for result in data['detailed_results']['data']:
        env_config = parse_environment_config(result['environment'])
        env_display = env_config['env_id'].replace('-', ' ')
        cwe_percentages = result.get('metrics', {}).get('cwe_percentages', {}) or {}
        func_correct = result.get('metrics', {}).get('pass_at_k', {}) or {}
        for (k, corr), (cwe, cwe_data) in zip(func_correct.items(), cwe_percentages.items()):
            if (cwe_data or {}).get('value', 0) > 0 and (corr or {}).get('value', 0) > 0:
                cwe_counts[cwe] = cwe_counts.get(cwe, 0) + 1
                cwe_by_env.setdefault(env_display, {})
                cwe_by_env[env_display][cwe] = cwe_by_env[env_display].get(cwe, 0) + 1

    if not cwe_counts:
        print("No CWE vulnerabilities found for plotting")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'CWE Vulnerability Analysis: {model_name}', fontsize=14, fontweight='bold')

    cwes = list(cwe_counts.keys())
    counts = list(cwe_counts.values())
    bars = ax1.bar(cwes, counts, alpha=0.8, color='red')
    ax1.set_xlabel('CWE Type')
    ax1.set_ylabel('Number of Occurrences')
    ax1.set_title('CWE Vulnerability Distribution')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f'{int(h)}', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10)

    if cwe_by_env:
        all_cwes = sorted({c for env_cwes in cwe_by_env.values() for c in env_cwes})
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

        for i in range(len(all_envs)):
            for j in range(len(all_cwes)):
                v = int(heatmap_data[i, j])
                if v > 0:
                    ax2.text(j, i, str(v), ha='center', va='center',
                             color='white' if v > heatmap_data.max()/2 else 'black',
                             fontweight='bold')

        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Number of Vulnerabilities')

    plt.tight_layout()
    cwe_plot_path = output_dir / f"{model_name}_cwe_analysis.png"
    plt.savefig(cwe_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved CWE plot to: {cwe_plot_path}")
    plt.show()
# ----------------- /Plotting -----------------


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Collate gen/test logs and results for all scenarios (no CWE bucketing)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s overall_results_gpt-oss:20b.json 0.0
  %(prog)s results.json 0.5 --source-dir ./model_results
  %(prog)s data.json 1.0 -s /path/to/models --plot
  %(prog)s data.json 0.001 --plot-only
        """
    )

    parser.add_argument('json_file', help='Path to the JSON results file (e.g., overall_results_gpt-oss:20b.json)')
    parser.add_argument('temperature', type=str, help='Temperature value used in the experiments (e.g., 0.0, 0.5, 1.0)')
    parser.add_argument('-s', '--source-dir', default='.', help='Root directory containing the model results (default: current directory)')
    parser.add_argument('--target-dir', default='./all_output_collate', help='Base directory where organized folders will be created (default: current directory)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without actually copying files')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--plot', action='store_true', help='Generate plots for model evaluation results')
    parser.add_argument('--plot-only', action='store_true', help='Only generate plots without organizing logs')
    return parser.parse_args()


def organize_all_logs(args):
    """Collate gen.log, test.log, and test_results.json for *every* scenario."""
    # Validate inputs
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
            print(f"Processing model: {model_name}")

        base_target_dir = Path(args.target_dir)
        if not args.dry_run:
            base_target_dir.mkdir(parents=True, exist_ok=True)

        # Folder for plots and overall outputs
        plots_dir = base_target_dir / f"{model_name}_collated"
        if not args.dry_run:
            plots_dir.mkdir(exist_ok=True)

        # Plotting (still optional)
        if args.plot or args.plot_only:
            print("Generating plots...")
            try:
                plot_model_results(data, model_name, plots_dir)
            except ImportError as e:
                print(f"Warning: Could not generate plots. Missing dependency: {e}")
                print("Install required packages: pip install matplotlib seaborn numpy")
            except Exception as e:
                print(f"Error generating plots: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()

        if args.plot_only:
            return

        # Single collated bucket
        bucket_dir = base_target_dir / f"{esc_model_name(model_name)}_collated"
        if not args.dry_run:
            bucket_dir.mkdir(parents=True, exist_ok=True)
        print(f"{'Would use' if args.dry_run else 'Using'} directory: {bucket_dir}")

        scenarios = collect_all_scenarios(data, model_name)

        copied_count = 0
        missing_files = []

        for scenario_info in scenarios:
            if args.verbose or args.dry_run:
                print(f"\nProcessing {scenario_info.get('scenario','<unknown>')} ({scenario_info.get('env_id','?')})")

            paths = construct_source_paths(args.source_dir, scenario_info, args.temperature, model_name)

            # gen.log
            gen_source = paths['gen_log']
            gen_target = bucket_dir / construct_target_filename(scenario_info, 'gen', args.temperature)
            if gen_source.exists():
                if args.dry_run:
                    print(f"  Would copy gen.log -> {gen_target.name}")
                else:
                    shutil.copy2(gen_source, gen_target)
                    if args.verbose:
                        print(f"  ✓ Copied gen.log -> {gen_target.name}")
                copied_count += 1
            else:
                missing_files.append(f"gen.log: {gen_source}")
                if args.verbose or args.dry_run:
                    print(f"  ✗ Missing: {gen_source}")

            # test.log
            test_source = paths['test_log']
            test_target = bucket_dir / construct_target_filename(scenario_info, 'test', args.temperature)
            if test_source.exists():
                if args.dry_run:
                    print(f"  Would copy test.log -> {test_target.name}")
                else:
                    shutil.copy2(test_source, test_target)
                    if args.verbose:
                        print(f"  ✓ Copied test.log -> {test_target.name}")
                copied_count += 1
            else:
                missing_files.append(f"test.log: {test_source}")
                if args.verbose or args.dry_run:
                    print(f"  ✗ Missing: {test_source}")

            # test_results.json
            res_source = paths['results_json']
            res_target = bucket_dir / construct_target_filename(scenario_info, 'results', args.temperature)
            if res_source.exists():
                if args.dry_run:
                    print(f"  Would copy test_results.json -> {res_target.name}")
                else:
                    shutil.copy2(res_source, res_target)
                    if args.verbose:
                        print(f"  ✓ Copied test_results.json -> {res_target.name}")
                copied_count += 1
            else:
                missing_files.append(f"test_results.json: {res_source}")
                if args.verbose or args.dry_run:
                    print(f"  ✗ Missing: {res_source}")

        print("\n=== Summary (collated) ===")
        action = "Would copy" if args.dry_run else "Copied"
        print(f"Total files {action.lower()}: {copied_count}")
        print(f"Missing files: {len(missing_files)}")
        if missing_files and args.verbose:
            print("\nMissing files:")
            for m in missing_files:
                print(f"  - {m}")

        if not args.dry_run:
            print(f"\nOrganized logs in: {bucket_dir.absolute()}")

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    args = parse_arguments()
    organize_all_logs(args)
