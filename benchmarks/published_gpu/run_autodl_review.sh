#!/usr/bin/env bash
# Recommended two-part review comparison. Each part has a separate archive.
set -Eeuo pipefail
exec </dev/null
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON="${PYTHON:-python3}"
[[ "${PREFIX}" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid RUN_ID' >&2; exit 64; }
status=0
for profile in paper quality; do
  rc=0
  echo "[review] Running ${profile}"
  PROFILE="${profile}" RUN_ID="${PREFIX}_${profile}" \
    bash "${ROOT}/run_autodl_all.sh" "$@" || rc=$?
  if [[ ${rc} == 2 ]]; then
    completion="${ROOT}/results/${PREFIX}_${profile}/campaign/completion.json"
    if [[ -f "${completion}" ]] && "${PYTHON}" -c \
      'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("completed") is True else 1)' "${completion}"; then
      status=2
    else
      echo "[review] ${profile} stopped before completing its campaign; retain its archive and logs" >&2
      exit "${rc}"
    fi
  elif [[ ${rc} != 0 ]]; then
    echo "[review] ${profile} could not complete; retain its archive and logs" >&2
    exit "${rc}"
  fi
done
echo '[review] Finished both parts; return both results archives'
exit "${status}"
