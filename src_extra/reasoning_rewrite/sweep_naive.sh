#!/usr/bin/env bash
set -euo pipefail

PARENT_DIR='./SYS_naive_interventions'
MODEL='openai/gpt-oss-120b'
TEMP='1.0'
SPEC='openapi'
REASONING='high'
ENVS=('Python-Django' 'Python-Flask' 'Python-aiohttp' 'Python-FastAPI')
PROMPTS=('none' 'generic' 'specific')   # safety_prompt values
MODES=('evaluate')

export LOCAL_API_BASE="http://localhost:8000"

join_by() { local IFS="$1"; shift; echo "$*"; }

run_main() {
  local mode="$1" prompt="$2" outdir="${PARENT_DIR}/${prompt}"
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
    --extra_args vllm=True reasoning_effort=high completions=False
    # --envs $(join_by ' ' "${ENVS[@]}") \
}

gather_for_prompt() {
  local prompt="$1" outdir="${PARENT_DIR}/${prompt}"
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
      echo "==> ${prompt} :: ${mode}"
      run_main "${mode}" "${prompt}"
    done
    echo "==> gather ${prompt}"
    gather_for_prompt "${prompt}"
  done
}

# main

# If you kept the same vars/functions from earlier:
# PARENT_DIR, MODEL, TEMP, SPEC, REASONING, ENVS[], EXTRA_ARGS, BASE_ENV, join_by()
FINAL_PROMPT='generic'                 # safety_prompt for this phase

for final_tag in prelim prelim_scaffold final_reminder; do
  FINAL_TAG_W_SUFFIX="${final_tag}_${FINAL_PROMPT}"
  final_json="${PARENT_DIR}/${FINAL_TAG_W_SUFFIX}/overall_results_${MODEL//\//-}_${TEMP}_${SPEC}_${FINAL_PROMPT}_${REASONING}.json"
  final_csv="${PARENT_DIR}/${FINAL_TAG_W_SUFFIX}/${MODEL//\//-}_${REASONING}_all_${final_tag}_intervention.csv"
  final_outdir="${PARENT_DIR}/${FINAL_TAG_W_SUFFIX}"

  # 1) Extract and add suffixes to intervention traces
  pipenv run python src_extra/reasoning_rewrite/extract_and_add_suffix.py \
    --type "${final_tag}" \
    --parent_dir "${PARENT_DIR}"

  # 2) Run intervene sweeps (generate → test → evaluate)
  for mode in generate test evaluate; do
    echo "==> ${FINAL_TAG_W_SUFFIX} intervention :: ${mode}"
    pipenv run python src/main.py \
      --models "${MODEL}" \
      --mode "${mode}" \
      --n_samples 5 \
      --temperature "${TEMP}" \
      --spec_type "${SPEC}" \
      --safety_prompt "${FINAL_PROMPT}" \
      --max_concurrent_runs 16 \
      --results_dir "${final_outdir}" \
      --timeout 55 \
      --envs $(join_by ' ' "${ENVS[@]}") \
      --extra_args vllm=True reasoning_effort=high completions=True trace_csv=${final_csv}
      # --envs $(join_by ' ' "${ENVS[@]}") \
      # --force \
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

