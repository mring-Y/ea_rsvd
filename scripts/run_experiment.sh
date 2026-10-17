#!/usr/bin/env bash
set -Eeuo pipefail

# One entry point for reproducible profiles. All profiles are non-interactive.
exec </dev/null
export PIP_NO_INPUT=1
export GIT_TERMINAL_PROMPT=0

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${EXPERIMENT_MODE:-all}"
PLATFORM="${EXPERIMENT_PLATFORM:-auto}"

case "${MODE}" in
  all|rank|precision|reference|overlap|structure) ;;
  *) echo "unsupported EXPERIMENT_MODE=${MODE}; use all, rank, precision, reference, overlap, or structure" >&2; exit 64 ;;
esac

case "${PLATFORM}" in
  a800|a100) export CUDA_ARCH="${CUDA_ARCH:-80}" ;;
  4090|4090d|auto) export CUDA_ARCH="${CUDA_ARCH:-89}" ;;
  h100|h20) export CUDA_ARCH="${CUDA_ARCH:-90}" ;;
  *) echo "unsupported EXPERIMENT_PLATFORM=${PLATFORM}" >&2; exit 64 ;;
esac

if [[ "${PLATFORM}" == "a800" || "${PLATFORM}" == "a100" ]] && [[ "${MODE}" != "structure" ]]; then
  exec bash "${ROOT}/scripts/run_autodl_a800_minimal.sh"
fi

case "${MODE}" in
  all) exec bash "${ROOT}/scripts/run_autodl_all.sh" ;;
  rank) export RUN_CUML=0; export MATRIX_NAMES="${MATRIX_NAMES:-west2021 lp_osa_30 lp_pds_02 ca-HepPh email-Enron}"; exec bash "${ROOT}/scripts/run_autodl_all.sh" ;;
  precision) export RUN_CUML=0; export MATRIX_NAMES="${MATRIX_NAMES:-west2021 lp_osa_30 ca-AstroPh ca-HepPh email-EuAll p2p-Gnutella31}"; exec bash "${ROOT}/scripts/run_autodl_all.sh" ;;
  reference|overlap) export RUN_CUML=0; export MATRIX_NAMES="${MATRIX_NAMES:-west2021 lp_80bau3b lp_cre_d lp_fit2p lp_maros lp_osa_14 lp_osa_30 lp_pds_02 lp_pilot87 lp_stocfor3 ca-AstroPh ca-HepPh email-EuAll p2p-Gnutella31}"; exec bash "${ROOT}/scripts/run_autodl_all.sh" ;;
  structure) export RUN_CUML=0; export MATRIX_NAMES="${MATRIX_NAMES:-ca-AstroPh ca-HepPh email-EuAll p2p-Gnutella31}"; export RUN_STRUCTURE=1; export STRUCTURE_ONLY=1; exec bash "${ROOT}/scripts/run_autodl_all.sh" ;;
esac
