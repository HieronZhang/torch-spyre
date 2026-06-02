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

"""Opt-in dumping of the ATen FX graph for inspecting the Spyre pipeline.

Everything here is a no-op unless the environment variable ``SPYRE_DUMP_IR``
is set to a truthy value (``1``, ``true``, ``yes`` or ``on``), so this module
is safe to leave wired into the pass pipeline permanently.

Output goes to stderr by default, or is appended to the file named by
``SPYRE_DUMP_IR_FILE`` when that variable is set.
"""

import os
import sys

import torch
import torch.fx

_TRUTHY = {"1", "true", "yes", "on"}


def dump_enabled() -> bool:
    """Return True when SPYRE_DUMP_IR requests dumping."""
    return os.environ.get("SPYRE_DUMP_IR", "").strip().lower() in _TRUTHY


def _emit(text: str) -> None:
    """Write one dump record to the configured sink (file or stderr)."""
    dest = os.environ.get("SPYRE_DUMP_IR_FILE")
    if dest:
        with open(dest, "a", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
    else:
        sys.stderr.write(text)
        sys.stderr.write("\n")
        sys.stderr.flush()


def _banner(title: str) -> str:
    bar = "=" * 78
    return f"{bar}\n==== {title}\n{bar}"


def _format_fx_graph(graph: torch.fx.Graph) -> str:
    """Render each FX node, annotated with its fake-tensor metadata."""
    lines = []
    for node in graph.nodes:
        line = node.format_node()
        if line is None:
            line = f"{node.op}: {node.name}"
        val = node.meta.get("val")
        if isinstance(val, torch.Tensor):
            line += f"    # {val.dtype} {tuple(val.shape)} {val.device}"
        lines.append(line)
    return "\n".join(lines)


def dump_fx_graph(
    graph: torch.fx.Graph,
    label: str = "ATen FX graph (post-grad, pre-lowering)",
) -> None:
    """Print the ATen FX graph; no-op unless SPYRE_DUMP_IR is set.

    Wired into ``CustomPostPasses`` so it runs on the post-grad FX graph just
    before Inductor lowers it to LoopLevel IR. A debug dump must never break
    compilation, so any formatting error is reported and swallowed.
    """
    if not dump_enabled():
        return
    try:
        body = _format_fx_graph(graph)
        _emit(f"{_banner(label)}\n{body}\n[{len(graph.nodes)} nodes]\n")
    except Exception as exc:  # noqa: BLE001 - instrumentation must not raise
        _emit(f"[SPYRE_DUMP_IR] failed to dump FX graph: {exc!r}")
