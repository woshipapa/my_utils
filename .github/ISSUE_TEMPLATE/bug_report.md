---
name: Bug report
about: Report something that does not work as expected
title: "[BUG]"
labels: ''
assignees: ''

---

**Describe the bug**
What happened, and what you expected instead.

**Command run**
The exact command (e.g. the `ncu`/`nsys` invocation or the analysis CLI call) that triggered the problem.

**Environment**
- Nsight Compute (`ncu --version`) / Nsight Systems (`nsys --version`) version, if relevant
- GPU model
- Python version; torch version if the code path uses it

**Report file**
Can you share the profiler report (or a redacted excerpt / CSV export) that reproduces the issue? Bugs with a reproducible report get fixed much faster.

**Logs / stack trace**
If applicable.
