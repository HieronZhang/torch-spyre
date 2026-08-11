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

"""Port the cost-model research onto `dev1`, the torch 2.11 line where flash still tiles.

WHY. Coarse-tiled flash attention compiled and measured on 2026-07-30 and fails on the
current line -- every tiled configuration dies in `insert_restickify.finalize_layouts`, and
the `merge` branch's coarse-tile rewrite did not fix it. `dev1` predates the regression, and
the 2828-record measurement database was collected on that toolchain, so the recorded
features and the compiler agree there. No re-sweep is wanted: the database is already right
for this tree.

WHAT MOVES, and why each is safe:

* `research/`, `tools/cost_model/` -- offline only. They import the cost model by file path
  and never touch the compiler, so the branch cannot affect them.
* `torch_spyre/_inductor/cost_model.py` -- the scoring function every prediction came from.
  Verified safe to move: all 78 CostParams fields shared with dev1 hold IDENTICAL values,
  and dev1 merely carries 8 extra parameters (`overlap_gamma*`, `write_reread_*`) whose
  terms were later removed. Taking HEAD's version changes no prediction already made.
* `profile_ops.py` -- the bench harness, including the LX-residency dump P1 reads and the
  `prefix_block` workload.
* The `LX_FORCE_ONLY` hook in `scratchpad/allocator.py` -- applied as a PATCH, never copied.
  dev1's allocator differs by 122 lines, so overwriting it would silently revert real work.

WHAT DOES NOT MOVE: `dump_cost_model.py` (dev1's differs only in comment reflow plus one
divide-guard), the `wsr/` package (that is the rewrite we are stepping away from), the sweep
driver (no re-sweep), and CI workflows.

THE ONE INCOMPATIBILITY is structural: `propagate_named_dims` and `coarse_tile` live at
`_inductor/wsr/` on the 2.13 line and flat at `_inductor/` on dev1. Rather than fork the
files, the imports in `profile_ops.py` and `research/workloads.py` resolve either layout, so
one copy serves both trees. This script verifies that rather than trusting it.

    git checkout dev1
    python3 research/migrate_to_dev1.py --from dev2-pr3364 --dry-run
    python3 research/migrate_to_dev1.py --from dev2-pr3364
    python3 research/migrate_to_dev1.py --verify-only

Nothing is committed. Files land in the working tree for review.
"""

import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

#: Whole directories to bring across verbatim. Offline tooling with no compiler coupling.
TREES = ["research", "tools/cost_model"]

#: Individual files, with a note on why each is safe to take wholesale.
FILES = [
    (
        "torch_spyre/_inductor/cost_model.py",
        "scoring function; CostParams verified identical on the shared fields",
    ),
    (
        "docs/source/user_guide/examples/profile_ops.py",
        "bench harness: LX residency dump, span guard, prefix_block",
    ),
    ("docs/source/compiler/cost_model_report.md", "the report"),
]

#: Brought only if present on the source ref; absence is not an error.
OPTIONAL = [
    "docs/source/user_guide/examples/run_cost_model_sweep.py",
    "docs/source/user_guide/examples/run_layout_cube.sh",
]

#: The allocator hook, applied by text insertion so dev1's own changes survive.
HOOK_CALL_ANCHOR = "        if op is None or not self._op_output_good_for_lx_reuse(op):"
HOOK_CALL = """        forced = _lx_force_override(name)
        if forced is not None:
            return forced
"""
HOOK_DEFS_ANCHOR = "def _lx_planning_size() -> int:"


def git(*args, ref_ok=True):
    p = subprocess.run(["git", *args], cwd=_ROOT, capture_output=True, text=True)
    if p.returncode and not ref_ok:
        raise RuntimeError(f"git {' '.join(args)}: {p.stderr.strip()}")
    return p.stdout


def current_branch():
    return git("rev-parse", "--abbrev-ref", "HEAD").strip()


def show(ref, path):
    p = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=_ROOT, capture_output=True)
    return p.stdout if p.returncode == 0 else None


def files_in(ref, tree):
    out = git("ls-tree", "-r", "--name-only", ref, "--", tree)
    return [ln for ln in out.splitlines() if ln.strip()]


def write(path, data, dry):
    full = os.path.join(_ROOT, path)
    if dry:
        return "would write"
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as fh:
        fh.write(data)
    return "written"


def apply_hook(dry):
    """Insert the LX_FORCE_ONLY override into dev1's allocator. Idempotent.

    A patch, not a copy: dev1's allocator.py differs from the 2.13 line by 122 lines, and
    replacing it wholesale would revert those silently. The hook is two pieces -- a call at
    the top of the residency gate and the two functions it needs -- both anchored on text
    that exists in both trees.
    """
    path = "torch_spyre/_inductor/scratchpad/allocator.py"
    full = os.path.join(_ROOT, path)
    src = open(full, encoding="utf-8").read()
    if "_lx_force_override" in src:
        return "already present"
    if HOOK_CALL_ANCHOR not in src:
        return "FAILED: residency gate anchor not found"
    if HOOK_DEFS_ANCHOR not in src:
        return "FAILED: _lx_planning_size anchor not found"

    defs = show("HEAD", path)
    if defs is None:
        return "FAILED: cannot read the hook source"
    text = defs.decode()
    start = text.index("_LX_BUF_ID = re.compile")
    end = text.index(HOOK_DEFS_ANCHOR)
    hook_defs = text[start:end]

    out = src.replace(HOOK_CALL_ANCHOR, HOOK_CALL + HOOK_CALL_ANCHOR, 1)
    out = out.replace(HOOK_DEFS_ANCHOR, hook_defs + HOOK_DEFS_ANCHOR, 1)
    if "\nimport os\n" not in out:
        out = out.replace("import logging\n", "import logging\nimport os\n", 1)
    if "import regex as re" not in out:
        out = out.replace("import sympy\n", "import regex as re\nimport sympy\n", 1)
    if dry:
        return "would patch"
    with open(full, "w", encoding="utf-8") as fh:
        fh.write(out)
    return "patched"


_SHIM_FN = '''

def _named_dims_module():
    """``propagate_named_dims``, wherever this tree keeps it.

    The module moved from ``_inductor/propagate_named_dims.py`` into the ``_inductor/wsr``
    package. Both layouts are in active use -- the torch 2.13 line has the package, the
    torch 2.11 line (``dev1``) has the flat module -- and the API is identical either way,
    so resolving it here lets one copy of this file serve both instead of forking it.
    """
    try:
        from torch_spyre._inductor.wsr import propagate_named_dims as mod
    except ImportError:
        from torch_spyre._inductor import propagate_named_dims as mod  # torch 2.11 line
    return mod

'''

_WSR_IMPORT = "    import torch_spyre._inductor.wsr.propagate_named_dims as pnd"


def apply_import_shim(dry):
    """Make the named-dims import resolve either layout. Idempotent.

    Applied HERE rather than relied upon from the source ref, because the shim may be an
    uncommitted working-tree edit -- and ``git show <ref>:<path>`` would then hand back the
    wsr-only version, which imports nothing on dev1. Re-applying costs nothing when the
    file already carries it and rescues the port when it does not.
    """
    out = []
    path = "docs/source/user_guide/examples/profile_ops.py"
    full = os.path.join(_ROOT, path)
    src = open(full, encoding="utf-8").read()
    if "_named_dims_module" not in src:
        anchor = "\ndef _rand("
        if anchor not in src:
            out.append("profile_ops.py: FAILED, no anchor for the shim")
        else:
            src = src.replace(anchor, _SHIM_FN + anchor, 1)
    n = src.count(_WSR_IMPORT)
    src = src.replace(_WSR_IMPORT, "    pnd = _named_dims_module()")
    if not dry:
        with open(full, "w", encoding="utf-8") as fh:
            fh.write(src)
    out.append(f"profile_ops.py: {n} wsr import(s) rewritten")

    path = "research/workloads.py"
    full = os.path.join(_ROOT, path)
    src = open(full, encoding="utf-8").read()
    old = """    from torch_spyre._inductor.wsr.propagate_named_dims import (
        declare_tensor_dim,
        name_tensor_dims,
    )"""
    new = """    try:  # the module moved into the `wsr` package on the torch 2.13 line
        from torch_spyre._inductor.wsr.propagate_named_dims import (
            declare_tensor_dim,
            name_tensor_dims,
        )
    except ImportError:  # torch 2.11 line (`dev1`) keeps it flat
        from torch_spyre._inductor.propagate_named_dims import (
            declare_tensor_dim,
            name_tensor_dims,
        )"""
    if old in src:
        src = src.replace(old, new, 1)
        if not dry:
            with open(full, "w", encoding="utf-8") as fh:
                fh.write(src)
        out.append("workloads.py: import made layout-agnostic")
    else:
        out.append("workloads.py: already agnostic")
    return out


def apply_overlay(overlay, dry):
    """Copy working-tree files saved before the checkout over the ported ref content.

    The migration reads from a git ref, so any edit that was never committed would be lost.
    Point this at a directory holding those files (same relative paths) to carry them.
    """
    if not overlay or not os.path.isdir(overlay):
        return []
    done = []
    for base, _, names in os.walk(overlay):
        for n in names:
            src = os.path.join(base, n)
            rel = os.path.relpath(src, overlay)
            if not dry:
                dst = os.path.join(_ROOT, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(src, "rb") as a, open(dst, "wb") as b:
                    b.write(a.read())
            done.append(rel)
    return done


def verify():
    """Check the things whose failure would be silent."""
    print("\n=== verify ===")
    ok = True

    import ast

    bad = []
    for base, _, names in os.walk(os.path.join(_ROOT, "research")):
        for n in names:
            if n.endswith(".py"):
                f = os.path.join(base, n)
                try:
                    ast.parse(open(f, encoding="utf-8").read())
                except SyntaxError as exc:
                    bad.append(f"{os.path.relpath(f, _ROOT)}: {exc}")
    print(f"  research/*.py parse            {'ok' if not bad else 'FAILED'}")
    for b in bad:
        ok = False
        print(f"     {b}")

    alloc = os.path.join(_ROOT, "torch_spyre/_inductor/scratchpad/allocator.py")
    src = open(alloc, encoding="utf-8").read()
    has = "_lx_force_override" in src and "forced = _lx_force_override(name)" in src
    print(f"  LX_FORCE_ONLY hook present    {'ok' if has else 'MISSING'}")
    try:
        ast.parse(src)
        print("  allocator.py parses           ok")
    except SyntaxError as exc:
        ok = False
        print(f"  allocator.py parses           FAILED: {exc}")
    ok &= has

    # The structural difference this port exists to absorb.
    pos = os.path.join(_ROOT, "docs/source/user_guide/examples/profile_ops.py")
    ps = open(pos, encoding="utf-8").read()
    agnostic = (
        "_named_dims_module" in ps and "wsr.propagate_named_dims as pnd" not in ps
    )
    print(f"  profile_ops import agnostic   {'ok' if agnostic else 'STILL wsr-ONLY'}")
    ok &= agnostic
    flat = os.path.exists(
        os.path.join(_ROOT, "torch_spyre/_inductor/propagate_named_dims.py")
    )
    pkg = os.path.exists(
        os.path.join(_ROOT, "torch_spyre/_inductor/wsr/propagate_named_dims.py")
    )
    print(
        f"  named-dims layout             {'flat (dev1)' if flat else ''}"
        f"{' + wsr package' if pkg else ''}"
    )

    for f in (
        "tools/cost_model/eval_model.py",
        "tools/cost_model/sweep_records.json",
        "research/lx_choice.py",
        "research/run_lx_experiments.py",
    ):
        e = os.path.exists(os.path.join(_ROOT, f))
        ok &= e
        print(f"  {f:<30}{'ok' if e else 'MISSING'}")

    # The offline model must still load and score, which is the whole basis of the study.
    try:
        sys.path.insert(0, os.path.join(_ROOT, "research"))
        sys.path.insert(0, os.path.join(_ROOT, "tools", "cost_model"))
        import lx_choice as L

        print(
            f"  cost model loads offline      ok "
            f"(LX budget {L.LX_CAPACITY_BYTES // 1024} KB/core)"
        )
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  cost model loads offline      FAILED: {type(exc).__name__}: {exc}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--from",
        dest="src",
        default="dev2-pr3364",
        help="ref to take the research tree from",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument(
        "--overlay",
        default="",
        help="directory of pre-checkout working-tree files to copy over the "
        "ported content (carries UNCOMMITTED edits the ref lacks)",
    )
    args = ap.parse_args()

    if args.verify_only:
        return 0 if verify() else 1

    branch = current_branch()
    print(f"on branch {branch}, porting from {args.src}\n")
    if branch == args.src:
        print(f"WARNING: already on {args.src}. Check out dev1 first:")
        print("    git checkout dev1")
        print("Nothing to do.")
        return 1
    if not git("rev-parse", "--verify", args.src).strip():
        print(f"ERROR: no such ref {args.src!r}")
        return 1

    n = 0
    for tree in TREES:
        paths = files_in(args.src, tree)
        print(f"  {tree:<28} {len(paths)} file(s)")
        for path in paths:
            data = show(args.src, path)
            if data is not None:
                write(path, data, args.dry_run)
                n += 1
    for path, why in FILES:
        data = show(args.src, path)
        status = (
            "MISSING on source" if data is None else write(path, data, args.dry_run)
        )
        print(f"  {path:<58} {status}")
        print(f"      ({why})")
        n += data is not None
    for path in OPTIONAL:
        data = show(args.src, path)
        if data is not None:
            write(path, data, args.dry_run)
            n += 1
            print(f"  {path:<58} written (optional)")

    for line in apply_overlay(args.overlay, args.dry_run):
        print(f"  overlay: {line}")
    for line in apply_import_shim(args.dry_run):
        print(f"  shim: {line}")
    print(f"\n  allocator hook: {apply_hook(args.dry_run)}")
    print(f"\n{n} file(s) {'would be ' if args.dry_run else ''}ported")
    if args.dry_run:
        print("\ndry run: nothing written. Re-run without --dry-run to apply.")
        return 0

    good = verify()
    print("\nNothing was committed. Review with `git status` / `git diff`.")
    if good:
        print("\nNext: rebuild (torch 2.11 line), then")
        print("    python3 research/probe_flash.py --extra")
        print(
            "    python3 research/run_lx_experiments.py --records "
            "tools/cost_model/sweep_records.json --phases 1,2,3"
        )
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
