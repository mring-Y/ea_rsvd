#!/usr/bin/env bash
set -Eeuo pipefail

# Minimal cross-platform Ampere check for an A800 or A100 (sm_80).  The
# script is intentionally separate from the full campaign: it supplies a
# bounded platform comparison without running the long overlap and baseline
# sweeps.
exec </dev/null
export PIP_NO_INPUT=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export GIT_TERMINAL_PROMPT=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "${SCRIPT_DIR}/../src/cuda" && -f "${SCRIPT_DIR}/../src/cuda/gpu_sparse_reference.cu" ]]; then
  ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
  # Historical AutoDL bundles placed this runner at the bundle root.
  ROOT="${SCRIPT_DIR}"
fi
if [[ -f "${ROOT}/project/src/native/gpu_sparse_reference.cu" && \
      -f "${ROOT}/stage54_incremental_controller/controller_precision_replay.cu" && \
      -f "${ROOT}/normalize_matrix_market.py" && \
      -f "${ROOT}/collect_manifest.py" ]]; then
  BUNDLE_ROOT="${ROOT}"
  DATA_ROOT="${BUNDLE_ROOT}/project/data/suitesparse"
  SPARSE_SRC="${BUNDLE_ROOT}/project/src/native/gpu_sparse_reference.cu"
  CONTROLLER_SRC="${BUNDLE_ROOT}/stage54_incremental_controller/controller_precision_replay.cu"
  NORMALIZE="${BUNDLE_ROOT}/normalize_matrix_market.py"
  COLLECT="${BUNDLE_ROOT}/collect_manifest.py"
else
  DATA_ROOT="${ROOT}/data/suitesparse"
  SPARSE_SRC="${ROOT}/src/cuda/gpu_sparse_reference.cu"
  CONTROLLER_SRC="${ROOT}/src/cuda/controller_precision_replay.cu"
  NORMALIZE="${ROOT}/scripts/normalize_matrix_market.py"
  COLLECT="${ROOT}/scripts/collect_manifest.py"
fi
BUILD_ROOT="${ROOT}/build_a800_minimal"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
PLATFORM_LABEL="${EXPERIMENT_PLATFORM:-a800}"
case "${PLATFORM_LABEL}" in
  a100|a800) ;;
  *) PLATFORM_LABEL="a800" ;;
esac
OUT_ROOT="${ROOT}/results/${RUN_ID}_${PLATFORM_LABEL}_minimal"
ARCH="${CUDA_ARCH:-80}"
FIXED_SEED="${A800_FIXED_SEED:-20261003}"
PRECISION_MATRICES_TEXT="${A800_PRECISION_MATRICES:-west2021 lp_osa_30 ca-AstroPh}"
PRECISION_SEEDS_TEXT="${A800_PRECISION_SEEDS:-20261003 20261004 20261005}"
RUN_PRECISION="${RUN_A800_PRECISION:-1}"
MATRIX_NAMES_TEXT="${A800_MATRIX_NAMES:-west2021 lp_80bau3b lp_cre_d lp_fit2p lp_maros lp_osa_14 lp_osa_30 lp_pds_02 lp_pilot87 lp_stocfor3 ca-AstroPh ca-HepPh email-EuAll p2p-Gnutella31}"
MATRICES=( ${MATRIX_NAMES_TEXT} )

die() { echo "[FATAL] $*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"; }
for cmd in nvcc python3 curl tar sha256sum; do need_cmd "${cmd}"; done
[[ ! -e "${OUT_ROOT}" ]] || die "run output already exists: ${OUT_ROOT}; choose RUN_ID"
[[ ! -e "${ROOT}/autodl_results_${RUN_ID}_${PLATFORM_LABEL}_minimal.tar.gz" ]] || die "result archive already exists for RUN_ID=${RUN_ID}"
[[ -f "${SPARSE_SRC}" ]] || die "missing sparse reference source: ${SPARSE_SRC}"
[[ -f "${CONTROLLER_SRC}" ]] || die "missing controller source: ${CONTROLLER_SRC}"
mkdir -p "${DATA_ROOT}/_archives" "${BUILD_ROOT}" "${OUT_ROOT}/logs" "${OUT_ROOT}/raw"
printf 'experiment\tstatus\tlog_or_output\n' >"${OUT_ROOT}/status.tsv"
printf 'experiment\texit_code\n' >"${OUT_ROOT}/failures.tsv"
printf 'experiment\texit_code\tduration_s\tlog\tclassification\tlast_line\n' >"${OUT_ROOT}/failure_details.tsv"

curl_retry_all=()
if curl --help all 2>/dev/null | grep -q -- '--retry-all-errors'; then curl_retry_all=(--retry-all-errors); fi
download_one() {
  local name="$1" group="$2" archive="${DATA_ROOT}/_archives/${name}.tar.gz"
  local target="${DATA_ROOT}/${name}/${name}/${name}.mtx" log="${OUT_ROOT}/logs/download_${name}.log"
  local primary="https://sparse.tamu.edu/MM/${group}/${name}.tar.gz"
  local mirror="https://suitesparse-collection-website.herokuapp.com/MM/${group}/${name}.tar.gz"
  mkdir -p "$(dirname "${target}")"; : >"${log}"
  if [[ -s "${target}" ]]; then printf 'download_%s\t0\t%s (already present)\n' "${name}" "${target}" >>"${OUT_ROOT}/status.tsv"; return 0; fi
  if [[ -s "${archive}" ]] && ! tar -tzf "${archive}" >/dev/null 2>>"${log}"; then
    if ! mv "${archive}" "${archive}.invalid.$(date -u +%Y%m%dT%H%M%SZ)"; then
      printf 'download_%s\t64\t%s\n' "${name}" "${log}" >>"${OUT_ROOT}/status.tsv"
      printf 'download_%s\t64\n' "${name}" >>"${OUT_ROOT}/failures.tsv"
      return 1
    fi
  fi
  local rc=1 url
  if [[ ! -s "${archive}" && -s "${archive}.part" ]] && tar -tzf "${archive}.part" >/dev/null 2>>"${log}"; then
    if ! mv "${archive}.part" "${archive}"; then
      printf 'download_%s\t69\t%s\n' "${name}" "${log}" >>"${OUT_ROOT}/status.tsv"
      printf 'download_%s\t69\n' "${name}" >>"${OUT_ROOT}/failures.tsv"
      return 1
    fi
  fi
  if [[ -s "${archive}" ]] && tar -tzf "${archive}" >/dev/null 2>>"${log}"; then
    rc=0
  fi
  if [[ ${rc} -ne 0 ]]; then
    for url in "${primary}" "${mirror}"; do
      echo "[data] ${name} ${url}" | tee -a "${log}"
      if curl -L --fail --show-error -C - --retry 5 --retry-delay 5 --retry-connrefused "${curl_retry_all[@]}" --connect-timeout 30 -o "${archive}.part" "${url}" >>"${log}" 2>&1; then rc=0; break; else rc=$?; fi
    done
    if [[ ${rc} -eq 0 ]]; then mv "${archive}.part" "${archive}" || rc=69; fi
  fi
  if [[ ${rc} -ne 0 ]] || ! tar -tzf "${archive}" >/dev/null 2>>"${log}"; then
    echo "[warn] download failed for ${name}; continuing" | tee -a "${log}"
    printf 'download_%s\t%s\t%s\n' "${name}" "${rc}" "${log}" >>"${OUT_ROOT}/status.tsv"
    printf 'download_%s\t%s\n' "${name}" "${rc}" >>"${OUT_ROOT}/failures.tsv"
    return 1
  fi
  local tmp src
  tmp="$(mktemp -d)"
  if ! tar -xzf "${archive}" -C "${tmp}" >>"${log}" 2>&1; then
    rm -rf "${tmp}"
    printf 'download_%s\t66\t%s\n' "${name}" "${log}" >>"${OUT_ROOT}/status.tsv"
    printf 'download_%s\t66\n' "${name}" >>"${OUT_ROOT}/failures.tsv"
    return 1
  fi
  src="$(find "${tmp}" -type f -name "${name}.mtx" -print -quit 2>>"${log}" || true)"
  if [[ -z "${src}" ]]; then
    rm -rf "${tmp}"
    printf 'download_%s\t67\t%s\n' "${name}" "${log}" >>"${OUT_ROOT}/status.tsv"
    printf 'download_%s\t67\n' "${name}" >>"${OUT_ROOT}/failures.tsv"
    return 1
  fi
  if ! cp "${src}" "${target}"; then
    rm -rf "${tmp}"
    printf 'download_%s\t70\t%s\n' "${name}" "${log}" >>"${OUT_ROOT}/status.tsv"
    printf 'download_%s\t70\n' "${name}" >>"${OUT_ROOT}/failures.tsv"
    return 1
  fi
  rm -rf "${tmp}"; printf 'download_%s\t0\t%s\n' "${name}" "${log}" >>"${OUT_ROOT}/status.tsv"
}

DOWNLOADED=()
for name in "${MATRICES[@]}"; do
  case "${name}" in west2021) group=HB ;; lp_*) group=LPnetlib ;; *) group=SNAP ;; esac
  if download_one "${name}" "${group}"; then DOWNLOADED+=("${name}"); else echo "[warn] skipping ${name}" >&2; fi
done
MATRICES=("${DOWNLOADED[@]}")
[[ ${#MATRICES[@]} -gt 0 ]] || die "no matrices available after download"
if ! python3 "${NORMALIZE}" --root "${DATA_ROOT}" --matrices "${MATRICES[@]}" >"${OUT_ROOT}/logs/normalize.log" 2>&1; then
  echo "[warn] batch normalization failed; retrying one matrix at a time" | tee -a "${OUT_ROOT}/logs/normalize.log"
  NORMALIZED=()
  for name in "${MATRICES[@]}"; do
    if python3 "${NORMALIZE}" --root "${DATA_ROOT}" --matrices "${name}" >>"${OUT_ROOT}/logs/normalize.log" 2>&1; then
      NORMALIZED+=("${name}")
    else
      printf 'normalize_%s\t68\t%s\n' "${name}" "${OUT_ROOT}/logs/normalize.log" >>"${OUT_ROOT}/status.tsv"
      printf 'normalize_%s\t68\n' "${name}" >>"${OUT_ROOT}/failures.tsv"
    fi
  done
  MATRICES=("${NORMALIZED[@]}")
fi
[[ ${#MATRICES[@]} -gt 0 ]] || die "no matrices normalized successfully"

{
  echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "platform=${PLATFORM_LABEL}"
  echo "cuda_arch=${ARCH}"
  nvcc --version | tail -n 1
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
} >"${OUT_ROOT}/environment.txt" 2>&1
[[ -f "${DATA_ROOT}/autodl_matrix_manifest.json" ]] && cp "${DATA_ROOT}/autodl_matrix_manifest.json" "${OUT_ROOT}/"

NVCC_FLAGS=(--std=c++17 -O3 -lineinfo -arch="sm_${ARCH}")
CUDA_LIBS=(-lcusparse -lcusolver -lcublas -lcurand)
compile_cuda() {
  local src="$1" out="$2"; shift 2; local log="${OUT_ROOT}/logs/build_${out}.log" rc
  rm -f "${BUILD_ROOT}/${out}"
  set +e; nvcc "${NVCC_FLAGS[@]}" -Xcompiler=-pthread "$@" "${src}" -o "${BUILD_ROOT}/${out}" "${CUDA_LIBS[@]}" >"${log}" 2>&1; rc=$?; set -e
  printf 'build_%s\t%s\t%s\n' "${out}" "${rc}" "${log}" >>"${OUT_ROOT}/status.tsv"
  if [[ ${rc} -ne 0 ]]; then
    printf 'build_%s\t%s\n' "${out}" "${rc}" >>"${OUT_ROOT}/failures.tsv"
    printf 'build_%s\t%s\t0\t%s\tcompile_failure\t%s\n' "${out}" "${rc}" "${log}" "$(tail -n 1 "${log}" | tr '\t\r\n' '   ' | cut -c1-500)" >>"${OUT_ROOT}/failure_details.tsv"
  fi
}
compile_cuda "${SPARSE_SRC}" gpu_sparse_reference
if [[ "${RUN_PRECISION}" == "1" ]]; then
  for mode in 0 1 2; do compile_cuda "${CONTROLLER_SRC}" "controller_precision_${mode}" -DPRECISION_MODE="${mode}"; done
fi

MATRIX_PATH() { echo "${DATA_ROOT}/$1/$1/$1.mtx"; }
run_case() {
  local tag="$1"; shift; local log="${OUT_ROOT}/logs/${tag}.stdout" exe="$1" rc start end duration last classification
  if [[ ! -x "${exe}" ]]; then
    printf '%s\t127\t%s\n' "${tag}" "${log}" >>"${OUT_ROOT}/status.tsv"
    printf '%s\t127\n' "${tag}" >>"${OUT_ROOT}/failures.tsv"
    echo "missing executable ${exe}" >"${log}"
    printf '%s\t127\t0\t%s\tmissing_executable\t%s\n' "${tag}" "${log}" "${exe}" >>"${OUT_ROOT}/failure_details.tsv"
    return 0
  fi
  start="$(date +%s)"; set +e; "$@" >"${log}" 2>&1; rc=$?; set -e; end="$(date +%s)"; duration=$((end-start))
  printf '%s\t%s\t%s\n' "${tag}" "${rc}" "${log}" >>"${OUT_ROOT}/status.tsv"
  if [[ ${rc} -ne 0 ]]; then
    printf '%s\t%s\n' "${tag}" "${rc}" >>"${OUT_ROOT}/failures.tsv"; classification=runtime_or_validation_failure; [[ ${rc} -eq 1 ]] && classification=cuda_or_library_runtime_failure; [[ ${rc} -eq 2 ]] && classification=numerical_or_solver_validation_failure; last="$(tail -n 1 "${log}" | tr '\t\r\n' '   ' | cut -c1-500)"; printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${tag}" "${rc}" "${duration}" "${log}" "${classification}" "${last}" >>"${OUT_ROOT}/failure_details.tsv"
  fi
}

SPARSE_CSV="${OUT_ROOT}/same_cuda_sparse_reference.csv"
for name in "${MATRICES[@]}"; do run_case "sparse_reference_${name}" "${BUILD_ROOT}/gpu_sparse_reference" "$(MATRIX_PATH "${name}")" "${SPARSE_CSV}" 16 8 1 "${FIXED_SEED}"; done
if [[ "${RUN_PRECISION}" == "1" ]]; then
  for mode in 0 1 2; do
    for name in ${PRECISION_MATRICES_TEXT}; do
      matrix="$(MATRIX_PATH "${name}")"
      [[ -f "${matrix}" ]] || { echo "[warn] precision matrix missing: ${name}" >&2; continue; }
      for seed in ${PRECISION_SEEDS_TEXT}; do run_case "controller_p${mode}_${name}_seed${seed}" "${BUILD_ROOT}/controller_precision_${mode}" "${matrix}" "${OUT_ROOT}/controller_p${mode}_${name}_seed${seed}.csv" "${seed}" 40; done
    done
  done
else
  printf 'controller_precision\tdisabled\tRUN_A800_PRECISION=%s\n' "${RUN_PRECISION}" >>"${OUT_ROOT}/status.tsv"
fi

python3 "${COLLECT}" --root "${ROOT}" --out "${OUT_ROOT}/manifest.json" >"${OUT_ROOT}/logs/manifest.log" 2>&1 || true
tar -czf "${ROOT}/autodl_results_${RUN_ID}_${PLATFORM_LABEL}_minimal.tar.gz" -C "${OUT_ROOT}" .
sha256sum "${ROOT}/autodl_results_${RUN_ID}_${PLATFORM_LABEL}_minimal.tar.gz" >"${ROOT}/autodl_results_${RUN_ID}_${PLATFORM_LABEL}_minimal.tar.gz.sha256"
echo "[done] results: ${OUT_ROOT}"
echo "[done] archive: ${ROOT}/autodl_results_${RUN_ID}_${PLATFORM_LABEL}_minimal.tar.gz"
