# AGENTS.md

Orientation for agents working in this repository. Humans should start at
[`my_utils/profiling/docs/PERFORMANCE_ANALYSIS_HANDBOOK.md`](my_utils/profiling/docs/PERFORMANCE_ANALYSIS_HANDBOOK.md).

## What this repo is

Profiling and analysis utilities for GPU training workloads. The centre of gravity
is `my_utils/profiling/`, which turns Nsight Systems and Nsight Compute output into
diagnoses: not "here are your metrics" but "this kernel is latency-bound on global
memory, here is the evidence, here is the fix, here is the ceiling on the win".

## Read this before touching the analysis code

**The analysis modules are pure Python — no torch, no CUDA, no GPU.** That is a
deliberate constraint, not an accident:

```
my_utils/profiling/hardware/gpu_specs.py       GPU peaks, ridge points
my_utils/profiling/sources/kernel_taxonomy.py  kernel-name classification
my_utils/profiling/ncu/metric_catalog.py       metric names, stall taxonomy
my_utils/profiling/ncu/ncu_diagnostics.py      the ncu rule engine
my_utils/profiling/analyzers/triage.py         the nsys top-down tree
```

Keep it that way. It is what makes `tests/profiling/test_analysis_engine.py` (80
tests) runnable in CI and on a laptop. If you need torch in an analysis path, you
have almost certainly put the code in the wrong layer.

Note that `my_utils/__init__.py` imports torch, so those tests load modules by file
path rather than through the package. Follow that pattern for new analysis tests.

```bash
# These run anywhere:
python -m pytest tests/profiling/test_analysis_engine.py -q

# The rest of tests/ needs torch installed.
```

## Non-obvious invariants

Breaking any of these produces confidently wrong output, which is worse than no
output. They are enforced by tests — if a test fails on one of these, the test is
probably right.

1. **GPU peaks are DENSE.** Datasheets quote tensor TFLOPS *with sparsity*.
   Grading achieved FLOP/s against a 2x-inflated peak halves every utilisation
   number. L40S is the trap: datasheet 362 TFLOPS BF16, dense figure 181.

2. **`uses_tensor_cores()` is tri-state.** `None` means the kernel name carries no
   evidence. Do not collapse it to `False` — "not using tensor cores" is only
   actionable when the name actually says so (`simt`, `sgemm`).

3. **Never hard-code an ncu metric string.** Ada appends `_v2` to tensor pipe
   metrics; Blackwell renames `dram__bytes_read` to `dram__bytes_op_read`. Go
   through `MetricView.get("catalog_key")`, which tries every known spelling.

4. **`selected` and `not_selected` are not bottlenecks.** `selected` is productive
   time; `not_selected` means you have *surplus* warps. They are in
   `BENIGN_STALL_KEYS` and must stay excluded from stall rankings.

5. **CUDA-core FLOP counters do not see tensor cores.** If the tensor pipe was busy
   but no `sm__ops_path_tensor_*` was collected, the FLOP total is a floor. The
   engine detects this (`flops_undercounted`) and reports "roofline unreliable"
   instead of a bogus percentage. Do not remove that guard.

6. **Communication is judged on exposure, not volume.** Collective time fully
   hidden under compute is not a finding. `exposed = comm_union - (comm ∩ compute)`.

7. **`speedup_ceiling` is a ceiling, never a prediction.** It follows GPA's
   `T/(T-M)` model. Latency-hiding fixes cap at 2x.

8. **Thresholds are cited and overridable.** `SOL_THRESHOLDS` (ncu) and
   `TriageThresholds` (nsys) both carry their source in a comment. NVIDIA
   calibrated theirs on LLM inference; loosen rather than fork.

9. **Demangled kernel names start with a return type.** Anchor patterns with `\b`,
   not `^` — `^kernel_dispatch_token` never matches
   `void kernel_dispatch_token(...)`. Mangled symbols are case-sensitive and are
   matched against the raw name, not the lowercased one.

10. **Per-kernel attribution is meaningless for megakernels.** Check
    `classify_kernel(name).megakernel` before reporting per-kernel percentages.

## How to extend

| Task | Where | Watch out for |
|---|---|---|
| New GPU | `_GPU_SPECS` in `hardware/gpu_specs.py` | Most-specific first (`h100 sxm` before `h100`); dense peaks |
| New kernel pattern | `KERNEL_CATEGORIES` / `_FRAMEWORK_PATTERNS` | Ordered, first match wins; use `\b` |
| New metric | `_CATALOG_LIST` in `ncu/metric_catalog.py` | List every arch spelling in `names` |
| New rule | `analyze_*(view)` in `ncu/ncu_diagnostics.py` | Return `{"findings": [Finding(...)]}`; register in `diagnose_kernel` |
| New triage branch | `triage_step` in `analyzers/triage.py` | Order matters — it is a decision tree |

Rules must read metrics through the catalog, put the producing numbers in
`evidence`, and **stay silent when the required metrics are absent** rather than
assuming zero. A missing metric is not a zero.

## Python compatibility

Target **Python 3.10**. The training runtime this supports is 3.10, and PEP 701
f-string features (backslashes inside replacement fields, quote reuse) parse only
on 3.12+. A regression here breaks `import my_utils.profiling` entirely on the
target runtime.

```bash
# Detector for the whole package (no 3.10 interpreter needed):
#   walks tokens, flags backslashes / quote reuse / multi-line exprs inside f-strings
```

If you are adding string-heavy code, prefer hoisting constants out of f-strings
over escaping inside them.

## Repository layout

```
my_utils/
├── profiling/          <- the main body of work
│   ├── hardware/       GPU capability tables
│   ├── ncu/            Nsight Compute: collection configs, catalog, rule engine
│   ├── sources/        Nsight Systems: SQLite parsing, SQL skills, taxonomy
│   ├── analyzers/      cross-source analysis and triage
│   ├── adapters/       framework integrations (Megatron, DeepSpeed, vLLM, ...)
│   ├── runtime/        in-process profiler control
│   ├── nccl/           NCCL inspector plugin output
│   ├── visualization/  HTML report generation
│   └── docs/           design docs and the handbook
├── core/               general utilities (imports torch)
├── distributed/        distributed helpers
└── hooks/              PyTorch hooks
```

## Conventions

- Match the surrounding code's style; these files favour explicit names and
  comments that explain *why*, not *what*.
- Cite the source for any threshold or hardware constant you introduce.
- When a tool's behaviour is version-dependent, say which version you verified
  against.
- Prefer adding a test over adding a comment claiming something works.
