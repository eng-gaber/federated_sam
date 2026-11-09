#!/usr/bin/env bash
# Convenience root runner. Delegates to scripts/run_mode_experiments.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/scripts/run_mode_experiments.sh" "$@"
