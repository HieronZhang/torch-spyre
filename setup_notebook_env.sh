#!/usr/bin/env bash
# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# One-shot setup for running notes/cost_model_demo.ipynb on ANY machine.
# The notebook is self-contained (embeds all data) — it needs only numpy,
# matplotlib and Jupyter; NO torch / NO Spyre runtime required.
#
#   bash setup_notebook_env.sh
#
# Requires `uv` (https://docs.astral.sh/uv/). The script creates an isolated
# venv `.venv-nb` and installs the notebook dependencies into it.

set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' not found. Install it first:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

VENV=".venv-nb"
echo ">> creating uv venv: $VENV (Python 3.12)"
uv venv "$VENV" --python 3.12

echo ">> installing notebook dependencies (numpy, matplotlib, jupyterlab, ipykernel)"
VIRTUAL_ENV="$VENV" uv pip install numpy matplotlib jupyterlab ipykernel

cat <<EOF

Done. To open the notebook:

  source $VENV/bin/activate
  jupyter lab notes/cost_model_demo.ipynb

Or run it headless to regenerate all figures:

  source $VENV/bin/activate
  jupyter nbconvert --to notebook --execute --inplace notes/cost_model_demo.ipynb
EOF
