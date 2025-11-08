#!/usr/bin/env bash
set -euo pipefail

PARENT_DIR='./sanity_interventions'
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
    # echo "==> gather ${prompt}"
    # gather_for_prompt "${prompt}"
  done
}

main

# --- Final intervention phase -------------------------------------------------
FINAL_PROMPT='none'                 # safety_prompt for this phase
FINAL_TAG='final_enumerate'          # results subdir + CSV location

# If you kept the same vars/functions from earlier:
# PARENT_DIR, MODEL, TEMP, SPEC, REASONING, ENVS[], EXTRA_ARGS, BASE_ENV, join_by()

for final_tag in prelim prelim_scaffold final_enumerate; do
  FINAL_TAG="${final_tag}"
  final_json="${PARENT_DIR}/${FINAL_TAG}/overall_results_${MODEL//\//-}_${TEMP}_${SPEC}_${FINAL_PROMPT}_${REASONING}.json"
  final_csv="${PARENT_DIR}/${FINAL_TAG}/${MODEL//\//-}_${REASONING}_all_${FINAL_TAG}_intervention.csv"
  final_outdir="${PARENT_DIR}/${FINAL_TAG}"

  # 1) Extract and add suffixes to intervention traces
  pipenv run python src_extra/reasoning_rewrite/extract_and_add_suffix.py \
    --type "${FINAL_TAG}" \
    --parent_dir "${PARENT_DIR}"

  # 2) Run intervene sweeps (generate → test → evaluate)
  for mode in evaluate; do
    echo "==> ${FINAL_TAG} intervention :: ${mode}"
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
  # pipenv run python src_extra/gather_all_files.py \
  # "${final_json}" "${TEMP}" \
  # -s "${final_outdir}/" \
  # --target-dir "${final_outdir}/" \
  # --reasoning-suffix "${REASONING}" -v

  # pipenv run python src_extra/gather_traces_into_csv.py \
  # --json_file "${final_json}" \
  # --target_dir "${final_outdir}/" \
  # --models "${MODEL//\//-}_${REASONING}"

done 

# # 3) Gather all files and traces for final_reminder
# pipenv run python src_extra/gather_all_files.py \
#   "${final_json}" "${TEMP}" \
#   -s "${final_outdir}/" \
#   --target-dir "${final_outdir}/" \
#   --reasoning-suffix "${REASONING}" -v

# pipenv run python src_extra/gather_traces_into_csv.py \
#   --json_file "${final_json}" \
#   --target_dir "${final_outdir}/" \
#   --models "${MODEL//\//-}_${REASONING}"

