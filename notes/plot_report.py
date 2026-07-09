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

"""Regenerate the figures for notes/cost_model_report.md from sweep_records.json.

Each figure is a function ``fig_<name>`` that reads the stored measured times and writes
a PNG to notes/figures/. Reproducible -- the data is the committed sweep records, not
hand-drawn. Run one figure or all:

    python notes/plot_report.py                 # all figures
    python notes/plot_report.py pointwise_baseline
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIGDIR = os.path.join(_HERE, "figures")
_RECORDS = os.path.join(_HERE, "sweep_records.json")


def _load(current_only=True):
    recs = json.load(open(_RECORDS, encoding="utf-8"))["records"]
    recs = [r for r in recs if not r.get("failed") and r.get("kernel_us")]
    if current_only:
        recs = [r for r in recs if r.get("is_current")]
    return recs


def _save(fig, name):
    os.makedirs(_FIGDIR, exist_ok=True)
    path = os.path.join(_FIGDIR, f"{name}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, _HERE)}")


# ============================================================================
# §1 -- pointwise is memory-I/O bound: neg time is linear in bytes through the
# origin (no fixed per-kernel cost).
# ============================================================================
def fig_pointwise_baseline(recs):
    pts = [
        (int(r["io_hbm_bytes"]) / 1e6, r["kernel_us"])
        for r in recs
        if r["op"] == "neg" and r.get("io_hbm_bytes")
    ]
    allx = np.array([x for x, _ in pts])
    ally = np.array([y for _, y in pts])
    # linear fit T[us] = a*MB + b (b -> fixed cost; a -> 1/BW)
    a, b = np.polyfit(allx, ally, 1)
    r2 = 1 - np.sum((ally - (a * allx + b)) ** 2) / np.sum((ally - ally.mean()) ** 2)
    bw = 1e3 / a  # MB/us = GB/s

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    xline = np.array([0, allx.max() * 1.03])
    ax.plot(
        xline,
        a * xline + b,
        "-",
        color="0.5",
        lw=1.2,
        zorder=1,
        label=f"linear fit: {bw:.0f} GB/s, $R^2$={r2:.4f}",
    )
    ax.scatter(
        allx,
        ally,
        s=42,
        color="#1f77b4",
        label="neg",
        zorder=3,
        edgecolors="white",
        linewidths=0.5,
    )
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel("HBM traffic  (MB, device / stick-padded)")
    ax.set_ylabel("kernel time  (µs)")
    ax.set_title("§1  Pointwise is memory-I/O bound (neg, 1R:1W)")
    ax.annotate(
        f"intercept b = {b:+.1f} µs  ≈ 0\n(no fixed per-kernel cost)",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        va="top",
        ha="left",
        fontsize=9,
        color="0.25",
        bbox=dict(boxstyle="round", fc="#f5f5f5", ec="0.8"),
    )
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax.margins(x=0.02)
    _save(fig, "fig1_pointwise_baseline")


_FIGS = {
    "pointwise_baseline": fig_pointwise_baseline,
}


def main():
    import sys

    recs = _load()
    want = sys.argv[1:] or list(_FIGS)
    for name in want:
        if name not in _FIGS:
            raise SystemExit(f"unknown figure {name!r} (have: {', '.join(_FIGS)})")
        print(f"figure {name}:")
        _FIGS[name](recs)


if __name__ == "__main__":
    main()
