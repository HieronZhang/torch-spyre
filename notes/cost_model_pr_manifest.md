# Cost-model PR manifest

**Nothing here has been committed, staged, or pushed.** Every path is a working-tree
change for you to branch and commit yourself.

## Before branching

The real upstream remote is configured but **not fetched**, so `upstream/main` is an
unknown revision locally:

```bash
git remote -v          # upstream -> git@github.com:torch-spyre/torch-spyre.git
git fetch upstream     # needed before you can branch from real main
```

This matters: every `git diff main` in this checkout currently resolves against your
**fork's** main, which already contains this work. That makes the change look far larger
than it is. Restricted to commits you authored, the whole cost-model footprint inside
`torch_spyre/` is two new files plus **two spliced lines** in `passes.py`.

## Belongs in the PR

| path | state | why |
|---|---|---|
| `torch_spyre/_inductor/cost_model.py` | modified | the model itself; pure Python, no torch import |
| `torch_spyre/_inductor/dump_cost_model.py` | modified | IR → features extraction |
| `torch_spyre/_inductor/cost_model_pass.py` | **new** | the pass: grouping, pricing, `CostReport` |
| `torch_spyre/_inductor/dump_common.py` | unchanged | hard dependency — `dump_cost_model` imports `banner`/`emit` from it |
| `torch_spyre/_inductor/config.py` | modified | the `cost_model` flag (one entry) |
| `torch_spyre/_inductor/passes.py` | modified | import, `last_cost_report`, the call site, the post-fusion entry |
| `tests/inductor/test_cost_model_pass.py` | **new** | 18 tests: grouping, attribution, disabled path |
| `tests/inductor/test_cost_model_decode.py` | unchanged | pre-existing, but see below |
| `tests/configs/.../test_cost_model_pass_config.yaml` | **new** | CI registration |
| `tests/configs/.../test_cost_model_decode_config.yaml` | **new** | CI registration — **this test has never run in CI**; 70 yamls exist and none covered it |
| `docs/source/compiler/cost_model.md` | **new** | the page, with the pipeline diagram |
| `docs/source/compiler/index.rst` | modified | one toctree line |
| `CLAUDE.md` | modified | one row in the environment-variable table |

## Does NOT belong in this PR

| path | why |
|---|---|
| `torch_spyre/_inductor/dump_loop_ir.py` | separate concern (IR dumping), and it needs `pass_utils.format_operations`, which exists only on the `layout211` branch |
| `torch_spyre/_inductor/dump_fx_graph.py` | separate concern; land with the IR dumper |
| `notes/**` | research artifacts — the model report, the direction study, the analysis scripts |
| `haoyang_logs/**` | raw measurement logs |
| `docs/source/user_guide/examples/run_*.sh` | sweep drivers, not product |

## Reviewer notes worth putting in the PR description

- **Off by default, and off is free.** `config.cost_model` defaults to `""`; the pass
  returns before touching the graph.
- **Naming.** `passes.py` already imports `cost_model_matmul_division`, an unrelated model
  that chooses a matmul work division. The new code carries a comment at both the config
  flag and the module docstring pointing this out, since reviewers will otherwise conflate
  them.
- **Cache key.** The pre-scheduling call is deliberately *outside* `self.passes`, so it is
  not hashed into the Inductor cache key by `_uuid`. A read-only report must not invalidate
  caches. The post-fusion entry *is* a list member, so it does affect the key — flag this
  if that is unwanted.
- **`SPYRE_DUMP_COST` now drives two things.** The pre-existing `dump_cost_model` still
  reads it directly, so at `=1` both the old per-op feature dump and the new grouped report
  print. If only one is wanted, say which.
- **Non-additivity.** `predict_ops` is per fused bundle and not additive over its ops; the
  per-op column in the report is an attribution of the group total, not independent
  predictions. Two tests guard this.

## Verification run before handing over

```bash
python3 notes/eval_model.py --all        # gold categories unchanged
pre-commit run --files <every path above> # ruff, pymarkdown, yamlfmt clean
```

`mypy` reports two pre-existing errors in `cost_model.py:1589` that are not from this work
(confirmed by checking the file before these changes); `cost_model_pass.py` reports none.
