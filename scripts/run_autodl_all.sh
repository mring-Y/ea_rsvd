#!/usr/bin/env bash
set -Eeuo pipefail

# One-shot Linux/AutoDL campaign for the remaining review experiments.
# Upload/unpack this folder as one bundle; all required CUDA sources are inside it.
# Typical invocation:
#   CUDA_ARCH=89 RUN_CUML=1 bash scripts/run_autodl_all.sh  # RTX 4090/4090D

# Never consume a terminal or inherited pipe.  All configuration comes from
# environment variables and command-line tools write to the recorded logs.
exec </dev/null
# Keep optional package and VCS helpers noninteractive as well.  This matters
# when the runner is started by nohup or a batch scheduler without a tty.
export PIP_NO_INPUT=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export GIT_TERMINAL_PROMPT=0

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${ROOT}/data/suitesparse"
BUILD_ROOT="${ROOT}/build"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${ROOT}/results/${RUN_ID}"
ARCH="${CUDA_ARCH:-}"
PLATFORM_LABEL="${EXPERIMENT_PLATFORM:-auto}"
SEEDS_TEXT="${SEEDS:-20261003 20261004 20261005 20261006 20261007}"
RUN_CUML="${RUN_CUML:-1}"
INCLUDE_ROADNET="${INCLUDE_ROADNET:-0}"
RUN_STRUCTURE="${RUN_STRUCTURE:-0}"
STRUCTURE_ONLY="${STRUCTURE_ONLY:-0}"
MATRIX_NAMES="${MATRIX_NAMES:-}"
if [[ -n "${MATRIX_NAMES}" ]]; then
  # Split the environment-provided list without reading from stdin.  This
  # keeps the runner fully non-interactive when launched through nohup.
  MATRICES=( ${MATRIX_NAMES} )
else
  MATRICES=(west2021 lp_80bau3b lp_cre_d lp_fit2p lp_maros lp_osa_14 lp_osa_30 lp_pds_02 lp_pilot87 lp_stocfor3 ca-AstroPh ca-HepPh email-EuAll p2p-Gnutella31)
fi
if [[ "${INCLUDE_ROADNET}" == "1" && " ${MATRICES[*]} " != *" roadNet-CA "* ]]; then
  MATRICES+=(roadNet-CA)
fi

die() { echo "[FATAL] $*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"; }
[[ ! -e "${OUT_ROOT}" ]] || die "run output already exists: ${OUT_ROOT}; choose a new RUN_ID"
[[ ! -e "${ROOT}/autodl_results_${RUN_ID}.tar.gz" ]] || die "result archive already exists for RUN_ID=${RUN_ID}"
mkdir -p "${DATA_ROOT}/_archives" "${BUILD_ROOT}" "${OUT_ROOT}/logs" "${OUT_ROOT}/raw"
printf 'experiment\tstatus\tlog_or_output\n' > "${OUT_ROOT}/status.tsv"
printf 'experiment\texit_code\n' > "${OUT_ROOT}/failures.tsv"
printf 'experiment\texit_code\tduration_s\tlog\tclassification\tlast_line\n' > "${OUT_ROOT}/failure_details.tsv"
need_cmd nvcc
need_cmd python3
need_cmd curl
need_cmd tar
if [[ -z "${ARCH}" ]]; then
  need_cmd nvidia-smi
  cap="$(python3 - <<'PY'
try:
    import torch
    major, minor = torch.cuda.get_device_capability(0)
    print(f"{major}{minor}")
except Exception:
    pass
PY
)"
  if [[ -z "${cap}" ]]; then
    cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '.[:space:]' || true)"
  fi
  if [[ "${cap}" =~ ^[0-9]{2,3}$ ]]; then
    ARCH="${cap}"
  else
    gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
    case "${gpu_name}" in
      *A100*|*A800*) ARCH=80 ;;
      *RTX\ 4090*|*L40S*|*L4*) ARCH=89 ;;
      *H100*|*H20*) ARCH=90 ;;
      *B200*|*B100*) ARCH=100 ;;
      *RTX\ 5090*) ARCH=120 ;;
      *) die "cannot infer CUDA architecture for '${gpu_name}'; set CUDA_ARCH explicitly" ;;
    esac
  fi
fi

echo "[env] checking and installing NumPy/SciPy when needed"
set +e
python3 -c 'import numpy, scipy' >/dev/null 2>&1
python_ready=$?
set -e
if [[ ${python_ready} -eq 0 ]]; then
  printf 'python_dependencies\t0\tpreinstalled\n' >> "${OUT_ROOT}/status.tsv"
fi
if [[ ${python_ready} -ne 0 ]]; then
  if ! python3 -m pip --version >/dev/null 2>&1; then
    set +e
    python3 -m ensurepip --upgrade >"${OUT_ROOT}/logs/ensurepip.stdout" 2>&1
    ensurepip_rc=$?
    set -e
    printf 'ensurepip\t%s\t%s\n' "${ensurepip_rc}" "${OUT_ROOT}/logs/ensurepip.stdout" >> "${OUT_ROOT}/status.tsv"
  fi
  if python3 -m pip --version >/dev/null 2>&1; then
    set +e
    python3 -m pip install --user --no-input numpy scipy >"${OUT_ROOT}/logs/python_dependencies.stdout" 2>&1
    python_ready=$?
    set -e
  else
    python_ready=127
    echo "pip/ensurepip unavailable" >"${OUT_ROOT}/logs/python_dependencies.stdout"
  fi
  printf 'python_dependencies\t%s\t%s\n' "${python_ready}" "${OUT_ROOT}/logs/python_dependencies.stdout" >> "${OUT_ROOT}/status.tsv"
  if [[ ${python_ready} -ne 0 ]]; then
    printf 'python_dependencies\t%s\n' "${python_ready}" >> "${OUT_ROOT}/failures.tsv"
    echo "[warn] NumPy/SciPy install failed; GPU experiments continue, CPU reference and cuML input loading will be marked unavailable"
  fi
fi
python3 - <<'PY' > "${OUT_ROOT}/python_environment.txt"
import sys
print("python=" + sys.version.replace("\n", " "))
for package in ("numpy", "scipy", "cupy", "cuml"):
    try:
        mod = __import__(package)
        print(f"{package}={getattr(mod, '__version__', 'available')}")
    except Exception as exc:
        print(f"{package}=unavailable: {type(exc).__name__}: {exc}")
PY

echo "[info] root=${ROOT}"
echo "[info] CUDA_ARCH=${ARCH}; seeds=${SEEDS_TEXT}"

CURL_RETRY_ALL_ERRORS=()
if curl --help all 2>/dev/null | grep -q -- '--retry-all-errors'; then
  CURL_RETRY_ALL_ERRORS=(--retry-all-errors)
fi

download_one() {
  local name="$1" group="$2"
  local archive="${DATA_ROOT}/_archives/${name}.tar.gz"
  local primary_url="https://sparse.tamu.edu/MM/${group}/${name}.tar.gz"
  local mirror_url="https://suitesparse-collection-website.herokuapp.com/MM/${group}/${name}.tar.gz"
  local target="${DATA_ROOT}/${name}/${name}/${name}.mtx"
  local log="${OUT_ROOT}/logs/download_${name}.log"
  mkdir -p "${DATA_ROOT}/_archives" "${OUT_ROOT}/logs"
  : >"${log}"

  if [[ -s "${target}" ]]; then
    printf 'download_%s\t0\t%s (already present)\n' "${name}" "${target}" >>"${OUT_ROOT}/status.tsv"
    return 0
  fi

  if [[ -s "${archive}" ]] && ! tar -tzf "${archive}" >/dev/null 2>>"${log}"; then
    echo "[data] existing archive is invalid; preserving it and retrying: ${archive}" | tee -a "${log}"
    if ! mv "${archive}" "${archive}.invalid.$(date -u +%Y%m%dT%H%M%SZ)"; then
      echo "[data] could not move invalid archive for ${name}; skipping" | tee -a "${log}"
      printf 'download_%s\t%s\t%s\n' "${name}" 64 "${log}" >>"${OUT_ROOT}/status.tsv"
      printf 'download_%s\t%s\n' "${name}" 64 >>"${OUT_ROOT}/failures.tsv"
      return 1
    fi
  fi

  # A previous curl process may have finished transferring the archive but
  # exited during the final connection close. Reuse that complete .part file
  # before opening another network request.
  if [[ ! -s "${archive}" && -s "${archive}.part" ]] && tar -tzf "${archive}.part" >/dev/null 2>>"${log}"; then
    echo "[data] using completed partial archive for ${name}" | tee -a "${log}"
    if ! mv "${archive}.part" "${archive}"; then
      echo "[data] could not finalize completed partial archive for ${name}; continuing" | tee -a "${log}"
      printf 'download_%s\t%s\t%s\n' "${name}" 69 "${log}" >>"${OUT_ROOT}/status.tsv"
      printf 'download_%s\t%s\n' "${name}" 69 >>"${OUT_ROOT}/failures.tsv"
      return 1
    fi
  fi

  if [[ ! -s "${archive}" ]]; then
    local download_rc=1
    local url
    for url in "${primary_url}" "${mirror_url}"; do
      echo "[data] downloading ${name} from ${url}" | tee -a "${log}"
      # Keep the .part file on failure so a later run can resume it.  The
      # explicit if-condition prevents set -e from aborting the whole campaign.
      if curl -L --fail --show-error -C - --retry 5 --retry-delay 5 \
          --retry-connrefused "${CURL_RETRY_ALL_ERRORS[@]}" --connect-timeout 30 \
          -o "${archive}.part" "${url}" >>"${log}" 2>&1; then
        download_rc=0
        break
      else
        download_rc=$?
        echo "[data] ${url} failed with exit code ${download_rc}; trying the next official endpoint" | tee -a "${log}"
      fi
    done
    if [[ ${download_rc} -ne 0 ]]; then
      echo "[data] download failed for ${name}; preserving ${archive}.part and continuing" | tee -a "${log}"
      printf 'download_%s\t%s\t%s\n' "${name}" "${download_rc}" "${log}" >>"${OUT_ROOT}/status.tsv"
      printf 'download_%s\t%s\n' "${name}" "${download_rc}" >>"${OUT_ROOT}/failures.tsv"
      return 1
    fi
    if ! mv "${archive}.part" "${archive}"; then
      echo "[data] could not finalize archive for ${name}; preserving .part and continuing" | tee -a "${log}"
      printf 'download_%s\t%s\t%s\n' "${name}" 69 "${log}" >>"${OUT_ROOT}/status.tsv"
      printf 'download_%s\t%s\n' "${name}" 69 >>"${OUT_ROOT}/failures.tsv"
      return 1
    fi
  fi

  if ! tar -tzf "${archive}" >/dev/null 2>>"${log}"; then
    echo "[data] archive validation failed for ${name}; skipping" | tee -a "${log}"
    printf 'download_%s\t%s\t%s\n' "${name}" 65 "${log}" >>"${OUT_ROOT}/status.tsv"
    printf 'download_%s\t%s\n' "${name}" 65 >>"${OUT_ROOT}/failures.tsv"
    return 1
  fi

  local tmp src
  tmp="$(mktemp -d)"
  if ! tar -xzf "${archive}" -C "${tmp}" >>"${log}" 2>&1; then
    rm -rf "${tmp}"
    echo "[data] extraction failed for ${name}; skipping" | tee -a "${log}"
    printf 'download_%s\t%s\t%s\n' "${name}" 66 "${log}" >>"${OUT_ROOT}/status.tsv"
    printf 'download_%s\t%s\n' "${name}" 66 >>"${OUT_ROOT}/failures.tsv"
    return 1
  fi
  src="$(find "${tmp}" -type f -name "${name}.mtx" -print -quit 2>>"${log}" || true)"
  if [[ -z "${src}" ]]; then
    rm -rf "${tmp}"
    echo "[data] ${name}.mtx not found in ${archive}; skipping" | tee -a "${log}"
    printf 'download_%s\t%s\t%s\n' "${name}" 67 "${log}" >>"${OUT_ROOT}/status.tsv"
    printf 'download_%s\t%s\n' "${name}" 67 >>"${OUT_ROOT}/failures.tsv"
    return 1
  fi
  mkdir -p "${DATA_ROOT}/${name}/${name}"
  if ! cp "${src}" "${target}"; then
    rm -rf "${tmp}"
    echo "[data] could not install ${name}.mtx; skipping" | tee -a "${log}"
    printf 'download_%s\t%s\t%s\n' "${name}" 70 "${log}" >>"${OUT_ROOT}/status.tsv"
    printf 'download_%s\t%s\n' "${name}" 70 >>"${OUT_ROOT}/failures.tsv"
    return 1
  fi
  rm -rf "${tmp}"
  printf 'download_%s\t0\t%s\n' "${name}" "${log}" >>"${OUT_ROOT}/status.tsv"
  return 0
}

echo "[data] downloading requested SuiteSparse matrices: ${MATRICES[*]}"
DOWNLOADED_MATRICES=()
for name in "${MATRICES[@]}"; do
  case "${name}" in
    west2021) group=HB ;;
    lp_*) group=LPnetlib ;;
    *) group=SNAP ;;
  esac
  if download_one "${name}" "${group}"; then
    DOWNLOADED_MATRICES+=("${name}")
  else
    echo "[warn] skipping ${name} after download failure" >&2
  fi
done

MATRICES=("${DOWNLOADED_MATRICES[@]}")
if [[ ${#MATRICES[@]} -eq 0 ]]; then
  die "no matrices were downloaded successfully; see ${OUT_ROOT}/logs/download_*.log"
fi

# The native reader requires general-real input.  This also records the source
# hash and expands symmetric pattern graph matrices without changing values.
if ! python3 "${ROOT}/scripts/normalize_matrix_market.py" \
  --root "${DATA_ROOT}" --matrices "${MATRICES[@]}" \
  >"${OUT_ROOT}/logs/normalize_matrix_market.log" 2>&1; then
  echo "[warn] batch normalization failed; retrying matrices individually" | tee -a "${OUT_ROOT}/logs/normalize_matrix_market.log"
  NORMALIZED_MATRICES=()
  for name in "${MATRICES[@]}"; do
    if python3 "${ROOT}/scripts/normalize_matrix_market.py" --root "${DATA_ROOT}" --matrices "${name}" \
      >>"${OUT_ROOT}/logs/normalize_matrix_market.log" 2>&1; then
      NORMALIZED_MATRICES+=("${name}")
    else
      echo "[warn] skipping ${name} after normalization failure" | tee -a "${OUT_ROOT}/logs/normalize_matrix_market.log"
      printf 'normalize_%s\t%s\n' "${name}" 68 >>"${OUT_ROOT}/failures.tsv"
    fi
  done
  MATRICES=("${NORMALIZED_MATRICES[@]}")
  [[ ${#MATRICES[@]} -gt 0 ]] || die "no matrices were normalized successfully"
fi

cat > "${OUT_ROOT}/environment.txt" <<EOF
utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
hostname=$(hostname)
platform=${PLATFORM_LABEL}
cuda_arch=${ARCH}
nvcc=$(nvcc --version | tail -n 1)
EOF
cp "${DATA_ROOT}/autodl_matrix_manifest.json" "${OUT_ROOT}/"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi >> "${OUT_ROOT}/environment.txt" 2>&1 || true
fi
if command -v git >/dev/null 2>&1 && git -C "${ROOT}" rev-parse HEAD >/dev/null 2>&1; then
  echo "git_commit=$(git -C "${ROOT}" rev-parse HEAD)" >> "${OUT_ROOT}/environment.txt"
fi

NVCC_FLAGS=(--std=c++17 -O3 -lineinfo -arch="sm_${ARCH}")
CUDA_LIBS=(-lcusparse -lcusolver -lcublas -lcurand)
compile_cuda() {
  local src="$1" out="$2"; shift 2
  local build_log="${OUT_ROOT}/logs/build_${out}.log"
  echo "[build] ${out}"
  rm -f "${BUILD_ROOT}/${out}"
  set +e
  nvcc "${NVCC_FLAGS[@]}" -Xcompiler=-pthread "$@" "${src}" -o "${BUILD_ROOT}/${out}" "${CUDA_LIBS[@]}" >"${build_log}" 2>&1
  local rc=$?
  set -e
  printf 'build_%s\t%s\t%s\n' "${out}" "${rc}" "${build_log}" >> "${OUT_ROOT}/status.tsv"
  if [[ ${rc} -ne 0 ]]; then
    printf 'build_%s\t%s\n' "${out}" "${rc}" >> "${OUT_ROOT}/failures.tsv"
    echo "[warn] build ${out} failed; dependent cases will be recorded as unavailable"
    return 0
  fi
}

if [[ "${STRUCTURE_ONLY}" == "1" ]]; then
  compile_cuda "${ROOT}/src/cuda/rsvd_mm_exact_noprobe.cu" exact_noprobe
else
  compile_cuda "${ROOT}/src/cuda/rsvd_mm_benchmark.cu" rsvd_mm_benchmark
  compile_cuda "${ROOT}/src/cuda/gpu_sparse_reference.cu" gpu_sparse_reference
  compile_cuda "${ROOT}/src/cuda/rsvd_mm_exact_noprobe.cu" exact_noprobe
  compile_cuda "${ROOT}/src/cuda/accumulation_stress.cu" accumulation_stress
  compile_cuda "${ROOT}/src/cuda/dual_lane_dynamic_overlap_residual.cu" dual_lane_dynamic_overlap
  compile_cuda "${ROOT}/src/cuda/atq_device_tree_partitioned.cu" atq_device_tree_partitioned
  compile_cuda "${ROOT}/src/cuda/end_to_end_overlap.cu" end_to_end_overlap
  for mode in 0 1 2; do
    compile_cuda "${ROOT}/src/cuda/controller_precision_replay.cu" \
      "controller_precision_${mode}" -DPRECISION_MODE="${mode}"
  done
fi

echo "[run] matrix experiments"
MATRIX_PATH() { echo "${DATA_ROOT}/$1/$1/$1.mtx"; }

run_case() {
  local tag="$1"; shift
  local stdout="${OUT_ROOT}/logs/${tag}.stdout"
  local executable="$1"
  local command_text=""
  local arg
  for arg in "$@"; do
    printf -v command_text '%s %q' "${command_text}" "${arg}"
  done
  if [[ ! -x "${executable}" ]]; then
    printf '%s\t%s\t%s\n' "${tag}" "127" "${stdout}" >> "${OUT_ROOT}/status.tsv"
    printf '%s\t%s\n' "${tag}" "127" >> "${OUT_ROOT}/failures.tsv"
    printf 'Executable unavailable: %s\nCommand:%s\n' "${executable}" "${command_text}" >"${stdout}"
    printf '%s\t127\t0\t%s\tmissing_executable\t%s\n' "${tag}" "${stdout}" "Executable unavailable: ${executable}" >> "${OUT_ROOT}/failure_details.tsv"
    echo "[warn] ${tag} skipped: missing executable ${executable}"
    return 0
  fi
  echo "[case-start] ${tag}"
  printf 'Command:%s\nHost:%s\nCUDA_LAUNCH_BLOCKING:%s\n' "${command_text}" "$(hostname)" "${CUDA_LAUNCH_BLOCKING:-0}" >"${stdout}"
  local start_epoch end_epoch duration_s
  start_epoch="$(date +%s)"
  set +e
  "$@" >>"${stdout}" 2>&1
  local rc=$?
  set -e
  end_epoch="$(date +%s)"
  duration_s=$((end_epoch - start_epoch))
  printf '%s\t%s\t%s\n' "${tag}" "${rc}" "${stdout}" >> "${OUT_ROOT}/status.tsv"
  if [[ ${rc} -ne 0 ]]; then
    printf '%s\t%s\n' "${tag}" "${rc}" >> "${OUT_ROOT}/failures.tsv"
    local classification="runtime_or_validation_failure"
    if [[ ${rc} -eq 1 ]]; then
      classification="cuda_or_library_runtime_failure"
    elif [[ ${rc} -eq 2 ]]; then
      classification="numerical_or_solver_validation_failure"
    elif [[ ${rc} -eq 65 ]]; then
      classification="invalid_input_or_rank"
    elif [[ ${rc} -eq 127 ]]; then
      classification="missing_executable"
    fi
    local last_line
    last_line="$(tail -n 1 "${stdout}" | tr '\t\r\n' '   ' | cut -c1-500)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${tag}" "${rc}" "${duration_s}" "${stdout}" "${classification}" "${last_line}" >> "${OUT_ROOT}/failure_details.tsv"
    echo "[warn] ${tag} failed with code ${rc} (${classification}) after ${duration_s}s; continuing"
    echo "       log: ${stdout}"
    echo "       tail: ${last_line}"
  fi
  echo "[case-done] ${tag} rc=${rc} duration=${duration_s}s"
}

if [[ "${STRUCTURE_ONLY}" == "1" ]]; then
  [[ "${RUN_STRUCTURE}" == "1" ]] || RUN_STRUCTURE=1
  echo "[run] non-LP structure-only sweep"
  STRUCTURE_ROOT="${OUT_ROOT}/nonlp_structure"
  mkdir -p "${STRUCTURE_ROOT}"
  for name in "${MATRICES[@]}"; do
    case "${name}" in
      ca-AstroPh|ca-HepPh|email-EuAll|p2p-Gnutella31) ;;
      *) echo "[warn] structure profile skips non-SNAP matrix ${name}"; continue ;;
    esac
    matrix="$(MATRIX_PATH "${name}")"
    for seed in ${SEEDS_TEXT}; do
      for q in 0 1 2; do
        run_case "structure_${name}_seed${seed}_q${q}" \
          "${BUILD_ROOT}/exact_noprobe" "${matrix}" \
          "${STRUCTURE_ROOT}/${name}_seed${seed}_q${q}.csv" 16 8 "${q}" "${seed}"
      done
    done
  done
  python3 "${ROOT}/scripts/collect_manifest.py" --root "${ROOT}" --out "${OUT_ROOT}/manifest.json"
  tar -czf "${ROOT}/autodl_results_${RUN_ID}.tar.gz" -C "${OUT_ROOT}" .
  sha256sum "${ROOT}/autodl_results_${RUN_ID}.tar.gz" >"${ROOT}/autodl_results_${RUN_ID}.tar.gz.sha256"
  echo "[done] structure-only results: ${OUT_ROOT}"
  echo "[done] archive: ${ROOT}/autodl_results_${RUN_ID}.tar.gz"
  exit 0
fi

BASELINE_CSV="${OUT_ROOT}/same_cuda_rsvd.csv"
SPARSE_CSV="${OUT_ROOT}/same_cuda_sparse_reference.csv"
EXACT_CSV="${OUT_ROOT}/same_cuda_exact_noprobe.csv"
for name in "${MATRICES[@]}"; do
  matrix="$(MATRIX_PATH "${name}")"
  for seed in ${SEEDS_TEXT}; do
    for q in 0 1 2; do
      run_case "baseline_${name}_seed${seed}_q${q}" \
        "${BUILD_ROOT}/rsvd_mm_benchmark" "${matrix}" "${BASELINE_CSV}" 16 8 "${q}" "${seed}"
    done
  done
  # Independent cuSPARSE/cuSOLVER path (one representative power depth).
  run_case "sparse_reference_${name}" \
    "${BUILD_ROOT}/gpu_sparse_reference" "${matrix}" "${SPARSE_CSV}" 16 8 1 20261003
  run_case "exact_noprobe_${name}" \
    "${BUILD_ROOT}/exact_noprobe" "${matrix}" "${EXACT_CSV}" 16 8 1 20261003
done

if [[ "${RUN_STRUCTURE}" == "1" ]]; then
  echo "[run] non-LP structure sweep"
  STRUCTURE_ROOT="${OUT_ROOT}/nonlp_structure"
  mkdir -p "${STRUCTURE_ROOT}"
  for name in "${MATRICES[@]}"; do
    case "${name}" in
      ca-AstroPh|ca-HepPh|email-EuAll|p2p-Gnutella31) ;;
      *) continue ;;
    esac
    matrix="$(MATRIX_PATH "${name}")"
    for seed in ${SEEDS_TEXT}; do
      for q in 0 1 2; do
        run_case "structure_${name}_seed${seed}_q${q}" \
          "${BUILD_ROOT}/exact_noprobe" "${matrix}" \
          "${STRUCTURE_ROOT}/${name}_seed${seed}_q${q}.csv" 16 8 "${q}" "${seed}"
      done
    done
  done
fi

echo "[run] FP32/FP64/Kahan controller decision replay"
for mode in 0 1 2; do
  for name in "${MATRICES[@]}"; do
    matrix="$(MATRIX_PATH "${name}")"
    for seed in ${SEEDS_TEXT}; do
      run_case "controller_p${mode}_${name}_seed${seed}" \
        "${BUILD_ROOT}/controller_precision_${mode}" "${matrix}" \
        "${OUT_ROOT}/controller_p${mode}_${name}_seed${seed}.csv" "${seed}" 40
    done
  done
done

echo "[run] accumulation and overlap experiments"
run_case accumulation_stress_power2 "${BUILD_ROOT}/accumulation_stress" 1048576 8 256 10 "${OUT_ROOT}/accumulation_stress_power2.csv"
# Odd number of row blocks exercises the tree's carry path (255 leaves).
run_case accumulation_stress_odd "${BUILD_ROOT}/accumulation_stress" 65280 8 256 10 "${OUT_ROOT}/accumulation_stress_odd.csv"
for name in west2021 lp_fit2p ca-AstroPh p2p-Gnutella31; do
  matrix="$(MATRIX_PATH "${name}")"
  for block in 128 257 512; do
    for parts in 2 4; do
      for pool in 2 4; do
        run_case "dynamic_overlap_${name}_b${block}_p${parts}_m${pool}" \
          "${BUILD_ROOT}/dual_lane_dynamic_overlap" "${matrix}" \
          "${OUT_ROOT}/raw/dynamic_overlap_${name}_b${block}_p${parts}_m${pool}.csv" 16 "${block}" "${parts}" "${pool}"
      done
    done
  done
  for integrated in 0 1; do
    for dynamic in 0 1; do
      run_case "tree_pool_${name}_i${integrated}_d${dynamic}" \
        "${BUILD_ROOT}/atq_device_tree_partitioned" "${matrix}" \
        "${OUT_ROOT}/raw/tree_pool_${name}_i${integrated}_d${dynamic}.csv" 257 16 4 4 "${integrated}" "${dynamic}"
    done
  done
  run_case "end_to_end_overlap_${name}" \
    "${BUILD_ROOT}/end_to_end_overlap" "${matrix}" "${OUT_ROOT}/raw/end_to_end_overlap_${name}.csv" 16
done

echo "[run] CPU float64 Eckart--Young references (optional SciPy dependency)"
if python3 -c 'import numpy, scipy' >/dev/null 2>&1; then
  set +e
python3 "${ROOT}/scripts/high_precision_reference.py" \
    --data-root "${DATA_ROOT}" --output "${OUT_ROOT}/high_precision_reference.csv" \
    --matrices "${MATRICES[@]}" >"${OUT_ROOT}/logs/high_precision.stdout" 2>&1
  hp_rc=$?
  set -e
else
  hp_rc=127
  echo "NumPy/SciPy unavailable; float64 reference not run" >"${OUT_ROOT}/logs/high_precision.stdout"
fi
printf 'high_precision\t%s\t%s\n' "${hp_rc}" "${OUT_ROOT}/logs/high_precision.stdout" >> "${OUT_ROOT}/status.tsv"
if [[ ${hp_rc} -ne 0 ]]; then
  printf 'high_precision\t%s\n' "${hp_rc}" >> "${OUT_ROOT}/failures.tsv"
fi

if [[ "${RUN_CUML}" == "1" ]]; then
  echo "[run] optional cuML TruncatedSVD API probe"
  set +e
  if ! python3 -c 'import cuml, cupy' >/dev/null 2>&1; then
    echo "[run] cuML missing; trying the RAPIDS cuML CUDA 12 wheel"
    set +e
    python3 -m pip install --user --no-input --extra-index-url https://pypi.nvidia.com cuml-cu12 \
      >"${OUT_ROOT}/logs/cuml_install.stdout" 2>&1
    install_rc=$?
    set -e
    printf 'cuml_install\t%s\t%s\n' "${install_rc}" "${OUT_ROOT}/logs/cuml_install.stdout" >> "${OUT_ROOT}/status.tsv"
    if [[ ${install_rc} -ne 0 ]]; then
      printf 'cuml_install\t%s\n' "${install_rc}" >> "${OUT_ROOT}/failures.tsv"
    fi
  fi
  set +e
  python3 "${ROOT}/scripts/cuml_randomized_baseline.py" \
    --data-root "${DATA_ROOT}" --output "${OUT_ROOT}/cuml_truncated_svd.csv" \
    --matrices "${MATRICES[@]}" >"${OUT_ROOT}/logs/cuml.stdout" 2>&1
  cuml_rc=$?
  set -e
  printf 'cuml_truncated_svd\t%s\t%s\n' "${cuml_rc}" "${OUT_ROOT}/logs/cuml.stdout" >> "${OUT_ROOT}/status.tsv"
  if [[ ${cuml_rc} -ne 0 ]]; then
    printf 'cuml_truncated_svd\t%s\n' "${cuml_rc}" >> "${OUT_ROOT}/failures.tsv"
  fi
else
  echo "cuML disabled (RUN_CUML=${RUN_CUML})" >"${OUT_ROOT}/logs/cuml.stdout"
  printf 'cuml_truncated_svd\tdisabled\t%s\n' "${OUT_ROOT}/logs/cuml.stdout" >> "${OUT_ROOT}/status.tsv"
fi

python3 - <<'PY' > "${OUT_ROOT}/python_environment_final.txt"
import sys
print("python=" + sys.version.replace("\n", " "))
for package in ("numpy", "scipy", "cupy", "cuml"):
    try:
        mod = __import__(package)
        print(f"{package}={getattr(mod, '__version__', 'available')}")
    except Exception as exc:
        print(f"{package}=unavailable: {type(exc).__name__}: {exc}")
PY
python3 "${ROOT}/scripts/collect_manifest.py" --root "${ROOT}" --out "${OUT_ROOT}/manifest.json"
tar -czf "${ROOT}/autodl_results_${RUN_ID}.tar.gz" -C "${OUT_ROOT}" .
sha256sum "${ROOT}/autodl_results_${RUN_ID}.tar.gz" >"${ROOT}/autodl_results_${RUN_ID}.tar.gz.sha256"
echo "[done] results: ${OUT_ROOT}"
echo "[done] archive: ${ROOT}/autodl_results_${RUN_ID}.tar.gz"
if [[ "$(wc -l < "${OUT_ROOT}/failures.tsv")" -gt 1 ]]; then
  echo "[done] some cases failed; inspect ${OUT_ROOT}/failures.tsv and logs before upload"
fi
