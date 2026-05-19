#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

notebook_source="wildfire_susceptibility_rf_intro.py"
notebook_file="wildfire_susceptibility_rf_intro.ipynb"
pdf_file="wildfire_susceptibility_rf_intro.pdf"
timeout_seconds="${NOTEBOOK_TIMEOUT:-1200}"

echo "Converting ${notebook_source} to ${notebook_file}"
uv run jupytext --to ipynb "${notebook_source}"

echo "Executing ${notebook_file}"
uv run jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout="${timeout_seconds}" \
  "${notebook_file}"

echo "Converting ${notebook_file} to ${pdf_file}"
uv run jupyter nbconvert \
  --to webpdf \
  --output "$(basename "${pdf_file}" .pdf)" \
  "${notebook_file}"
