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

"""Locate the measurement database the cost model is scored against.

The database is ~14 MB of pretty-printed JSON that grows every time the hardware is
re-swept, so it is NOT committed: it would dominate the repository's history and every
re-sweep would rewrite hundreds of thousands of lines for a file no diff can usefully
show. It is fetched or pointed at instead.

Resolution order, first hit wins:

1. an explicit path passed by the caller (``--records``)
2. ``$SPYRE_COST_MODEL_RECORDS`` -- a path to a local copy
3. ``sweep_records.json`` beside this file, if a previous run already downloaded it
4. a download from ``$SPYRE_COST_MODEL_RECORDS_URL``, or ``DEFAULT_URL`` below,
   cached to (3) so it happens once

Nothing here needs hardware; the database holds measured times, and re-scoring only
re-runs the model over them.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(_HERE, "sweep_records.json")

#: Where the database is fetched from when nothing local is set. This points at the
#: branch the measurements were collected on, which is public but is NOT part of this
#: repository's history -- if it is ever deleted or rewritten, set
#: ``SPYRE_COST_MODEL_RECORDS_URL`` to wherever the data lives instead. A release asset
#: is the more durable home if this becomes load-bearing.
DEFAULT_URL = (
    "https://raw.githubusercontent.com/HieronZhang/torch-spyre/"
    "prepare_pr/tools/cost_model/sweep_records.json"
)

_HELP = f"""\
The cost-model measurement database was not found.

Point at a local copy:
    export SPYRE_COST_MODEL_RECORDS=/path/to/sweep_records.json

or point the download somewhere else (it defaults to the branch the
measurements were collected on) and it will cache to {CACHE}:
    export SPYRE_COST_MODEL_RECORDS_URL=<url of sweep_records.json>

or rebuild it on Spyre hardware, which takes a few hours:
    python3 docs/source/user_guide/examples/run_cost_model_sweep.py
"""


def records_path(explicit=None, download=True):
    """Absolute path to the database, fetching it once if needed.

    Raises SystemExit with instructions rather than a traceback: every caller is a
    command-line tool, and a missing database is a setup problem, not a bug.
    """
    if explicit:
        if not os.path.exists(explicit):
            sys.exit(f"no such records file: {explicit}")
        return explicit

    env = os.environ.get("SPYRE_COST_MODEL_RECORDS")
    if env:
        if not os.path.exists(env):
            sys.exit(f"SPYRE_COST_MODEL_RECORDS points at a missing file: {env}")
        return env

    if os.path.exists(CACHE):
        return CACHE

    url = os.environ.get("SPYRE_COST_MODEL_RECORDS_URL", DEFAULT_URL)
    if url and download:
        import urllib.request

        print(f"fetching the measurement database from {url}", file=sys.stderr)
        tmp = CACHE + ".part"
        try:
            urllib.request.urlretrieve(url, tmp)  # noqa: S310 - operator-supplied URL
            os.replace(tmp, CACHE)
        except Exception as exc:  # noqa: BLE001 - report the cause, not a traceback
            if os.path.exists(tmp):
                os.unlink(tmp)
            sys.exit(f"could not fetch {url}: {exc}\n\n{_HELP}")
        return CACHE

    sys.exit(_HELP)
