#!/usr/bin/env bash
set -euo pipefail

PARENT_DIR='./SYS_naive_interventions'
MODEL='openai/gpt-oss-120b'
TEMP='1.0'
SPEC='openapi'
REASONING='high'
ENVS=('Python-Django' 'Python-Flask' 'Python-aiohttp' 'Python-FastAPI')
# ENVS=('Rust-Actix' 'Go-Fiber' 'Python-Django' 'Python-FastAPI')
PROMPTS=('generic')   # safety_prompt values
MODES=('test' 'evaluate')
THINKING_BUDGETS=('10000' '12000')

export LOCAL_API_BASE="http://localhost:8000"

join_by() { local IFS="$1"; shift; echo "$*"; }

run_main() {
  local mode="$1" prompt="$2" thinking_budget="$3" 
  local outdir="${PARENT_DIR}/${prompt}_${thinking_budget}"
  pipenv run python src/main.py \
    --models "${MODEL}" \
    --mode "${mode}" \
    --n_samples 5 \
    --temperature "${TEMP}" \
    --spec_type "${SPEC}" \
    --safety_prompt "${prompt}" \
    --max_concurrent_runs 16 \
    --results_dir "${outdir}" \
    --timeout 55 \
    --envs $(join_by ' ' "${ENVS[@]}") \
    --vllm \
    --force \
    --reasoning_effort=high \
    --completions \
    --max_thinking_tokens=${thinking_budget} \
    --trace_csv=${PARENT_DIR}/no_op/${MODEL//\//-}_${REASONING}_all_no_op_generic_intervention.csv
    # --envs $(join_by ' ' "${ENVS[@]}") \
}

gather_for_prompt() {
  local prompt="$1" thinking_budget="$2" 
  outdir="${PARENT_DIR}/${prompt}_${thinking_budget}"
  local json="${outdir}/overall_results_${MODEL//\//-}_${TEMP}_${SPEC}_${prompt}_${REASONING}.json"
  # If your tool emits a different name, update the template above accordingly.

  pipenv run python src_extra/gather_all_files.py \
    "${json}" "${TEMP}" \
    -s "${outdir}/" \
    --target-dir "${outdir}/" \
    --reasoning-suffix "${REASONING}" -v

  pipenv run python src_extra/gather_traces_into_csv.py \
    --json_file "${json}" \
    --target_dir "${outdir}/" \
    --models "${MODEL//\//-}_${REASONING}"
}

main() {
  for prompt in "${PROMPTS[@]}"; do
    for mode in "${MODES[@]}"; do
      for THINKING_BUDGET in "${THINKING_BUDGETS[@]}"; do
        echo "==> ${prompt} :: ${mode} :: budget ${THINKING_BUDGET}"
        run_main "${mode}" "${prompt}" "${THINKING_BUDGET}"
      done
    done
    for THINKING_BUDGET in "${THINKING_BUDGETS[@]}"; do
      echo "==> gather ${prompt}"
      gather_for_prompt "${prompt}" "${THINKING_BUDGET}"
    done
  done
}

main

# If you kept the same vars/functions from earlier:
# PARENT_DIR, MODEL, TEMP, SPEC, REASONING, ENVS[], EXTRA_ARGS, BASE_ENV, join_by()
# wrap this in if false block
if false; then

FINAL_PROMPT='generic'                 # safety_prompt for this phase

# give higher token budgets 
for max_tokens in '2000' '4000' '6000' '8000' '10000' '12000'; do
  FINAL_TAG="final_reminder"          # results subdir + CSV location

  final_json="${PARENT_DIR}/${FINAL_TAG}_${max_tokens}/overall_results_${MODEL//\//-}_${TEMP}_${SPEC}_${FINAL_PROMPT}_${REASONING}.json"
  final_csv="${PARENT_DIR}/${FINAL_TAG}_${max_tokens}/${MODEL//\//-}_${REASONING}_all_${FINAL_TAG}_intervention.csv"
  final_outdir="${PARENT_DIR}/${FINAL_TAG}_${max_tokens}"

  # # 1) Extract and add suffixes to intervention traces
  # pipenv run python src_extra/reasoning_rewrite/extract_and_add_suffix.py \
  #   --type "${FINAL_TAG}" \
  #   --parent_dir "${PARENT_DIR}" \
  #   --max_tokens "${max_tokens}" \
  #   --in_file "${PARENT_DIR}/${FINAL_PROMPT}/${MODEL//\//-}_${REASONING}_all_analysis.csv" \
  #   --out_file "${PARENT_DIR}/${FINAL_TAG}_${max_tokens}/${MODEL//\//-}_${REASONING}_all_${FINAL_TAG}_intervention.csv"

  # 2) Run intervene sweeps (generate → test → evaluate)
  for mode in generate test evaluate; do
    echo "==> ${FINAL_TAG} intervention :: ${mode} :: budget ${max_tokens}"
    pipenv run python src/main.py \
      --models "${MODEL}" \
      --mode "${mode}" \
      --n_samples 5 \
      --temperature "${TEMP}" \
      --spec_type "${SPEC}" \
      --safety_prompt "${FINAL_PROMPT}" \
      --max_concurrent_runs 16 \
      --results_dir "${final_outdir}"  \
      --timeout 55 \
      --envs $(join_by ' ' "${ENVS[@]}") \
      --vllm \
      --reasoning_effort "high" \
      --completions \
      --trace_csv=${final_csv} \
      --max_thinking_tokens=${max_tokens} \
      --trace_csv=${PARENT_DIR}/no_op/${MODEL//\//-}_${REASONING}_all_no_op_generic_intervention.csv \
      --intervention_str "But wait, we need to handle security as well:" \
      --force
      #
  done
  pipenv run python src_extra/gather_all_files.py \
  "${final_json}" "${TEMP}" \
  -s "${final_outdir}/" \
  --target-dir "${final_outdir}/" \
  --reasoning-suffix "${REASONING}" -v

  pipenv run python src_extra/gather_traces_into_csv.py \
  --json_file "${final_json}" \
  --target_dir "${final_outdir}/" \
  --models "${MODEL//\//-}_${REASONING}"

done 
fi 

# # # 3) Gather all files and traces for final_reminder
# # pipenv run python src_extra/gather_all_files.py \
# #   "${final_json}" "${TEMP}" \
# #   -s "${final_outdir}/" \
# #   --target-dir "${final_outdir}/" \
# #   --reasoning-suffix "${REASONING}" -v

# # pipenv run python src_extra/gather_traces_into_csv.py \
# #   --json_file "${final_json}" \
# #   --target_dir "${final_outdir}/" \
# #   --models "${MODEL//\//-}_${REASONING}"

