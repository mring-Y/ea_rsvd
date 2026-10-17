#!/usr/bin/env bash
# Standalone published-solver comparison, suitable for nohup/batch execution.
set -Eeuo pipefail
exec </dev/null
export PIP_NO_INPUT=1 PIP_DISABLE_PIP_VERSION_CHECK=1 GIT_TERMINAL_PROMPT=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "${RUN_ID}" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid RUN_ID' >&2; exit 64; }
RUN_ROOT="${ROOT}/results/${RUN_ID}"
ARCHIVE="${ROOT}/published_gpu_results_${RUN_ID}.tar.gz"
[[ ! -e "${RUN_ROOT}" && ! -e "${ARCHIVE}" ]] || { echo 'Existing run; choose another RUN_ID' >&2; exit 64; }
mkdir -p "${RUN_ROOT}/logs"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}/.python_packages${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${ROOT}/third_party/cpu_lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
MONITOR_PID=""

stage() {
  printf '%s\t%s\n' "$(date +%s)" "$1" >"${RUN_ROOT}/runner_state.tsv"
  echo "[stage $(date -u +%FT%TZ)] $1"
}

run_logged() {
  local label=$1 log=$2 rc=0
  shift 2
  stage "${label}"
  echo "[detail] ${log}"
  "$@" 2>&1 | tee "${log}" || rc=$?
  echo "[stage] ${label} exit=${rc}"
  return "${rc}"
}

finish() {
  local rc=$?
  trap - EXIT
  if [[ -n "${MONITOR_PID}" ]]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
  stage "archive"
  printf '{"runner_exit_code":%s,"run_id":"%s"}\n' "${rc}" "${RUN_ID}" >"${RUN_ROOT}/runner_completion.json"
  if tar -czf "${ARCHIVE}" -C "${ROOT}/results" "${RUN_ID}"; then
    sha256sum "${ARCHIVE}" >"${ARCHIVE}.sha256"
    echo "[result] ${ARCHIVE}"
    echo '[result] Send this results archive back for manuscript integration.'
  else
    echo "[warn] Packaging failed; retain ${RUN_ROOT}" >&2
  fi
  exit "${rc}"
}
trap finish EXIT
stage "environment"
echo "[run] profile=${PROFILE:-full} run_id=${RUN_ID} results=${RUN_ROOT}"
"${PYTHON}" -u "${ROOT}/watch_progress.py" --run "${RUN_ROOT}" --watch \
  --interval "${HEARTBEAT_SECONDS:-20}" --parent-pid "$$" &
MONITOR_PID=$!

for cmd in "${NVCC:-nvcc}" "${CC:-gcc}" "${CXX:-g++}" nvidia-smi; do
  command -v "${cmd}" >/dev/null || { echo "Missing command ${cmd}; see README_zh.md" >&2; exit 127; }
done
if ! "${PYTHON}" -c 'import numpy,scipy' >/dev/null 2>&1; then
  run_logged python_dependencies "${RUN_ROOT}/logs/python_dependencies.log" \
    "${PYTHON}" -m pip install --no-input --target "${ROOT}/.python_packages" numpy==2.2.6 scipy==1.15.2
fi
run_logged asset_verification "${RUN_ROOT}/logs/asset_verification.log" "${PYTHON}" -u "${ROOT}/verify_assets.py"
run_logged unit_tests "${RUN_ROOT}/logs/unit_tests.log" "${PYTHON}" -u -m unittest discover -s "${ROOT}/tests" -v
run_logged build "${RUN_ROOT}/logs/build.log" "${PYTHON}" -u "${ROOT}/build_gpu.py" --jobs "${BUILD_JOBS:-2}"
cp "${ROOT}/build/build_manifest.json" "${RUN_ROOT}/build_manifest.json"
run_logged gpu_correctness "${RUN_ROOT}/logs/gpu_correctness.log" \
  "${PYTHON}" -u "${ROOT}/tests/gpu_correctness.py" --binary "${ROOT}/build/gpu_compare" --out "${RUN_ROOT}/correctness"

if [[ "${RUN_MEMCHECK:-0}" == 1 ]]; then
  run_logged memcheck "${RUN_ROOT}/logs/memcheck.log" \
    "${PYTHON}" -u "${ROOT}/tests/gpu_memcheck.py" --binary "${ROOT}/build/gpu_compare" \
    --tool "${COMPUTE_SANITIZER:-compute-sanitizer}" --out "${RUN_ROOT}/memcheck"
fi

case "${PROFILE:-full}" in
  quality)
    profile_args=(--ranks 16 32 --seeds 20261013 20261014 --repeats 3 \
                  --oversampling 24 --powers 2 4 6 8 --eps 0.01 0.001 0.00001 --block-sizes 1 8)
    ;;
  paper)
    profile_args=(--ranks 16 32 --seeds 20261013 20261014 20261015 20261016 20261017 --repeats 5)
    ;;
  full)
    profile_args=(--ranks 16 32 64 --seeds 20261013 20261014 20261015 20261016 20261017 --repeats 5)
    ;;
  smoke)
    profile_args=(--matrices west2021 lp_pds_02 --ranks 16 --seeds 20261013 --repeats 2 \
                  --oversampling 8 24 --powers 0 2 --eps 0.0001 0.000001 --block-sizes 1 8)
    ;;
  *) echo 'PROFILE must be paper, full, quality or smoke' >&2; exit 64 ;;
esac
campaign_rc=0
run_logged campaign "${RUN_ROOT}/logs/campaign.log" "${PYTHON}" -u "${ROOT}/campaign.py" --binary "${ROOT}/build/gpu_compare" \
  --matrix-root "${MATRIX_ROOT:-${ROOT}/data}" --out "${RUN_ROOT}/campaign" \
  --timeout "${CASE_TIMEOUT:-180}" --precisions "${PRECISION:-fp32}" \
  "${profile_args[@]}" "$@" || campaign_rc=$?
if [[ "${campaign_rc}" == 0 || "${campaign_rc}" == 2 ]]; then
  run_logged analysis "${RUN_ROOT}/logs/analysis.log" "${PYTHON}" -u "${ROOT}/analyse_results.py" "${RUN_ROOT}/campaign"
fi
exit "${campaign_rc}"
