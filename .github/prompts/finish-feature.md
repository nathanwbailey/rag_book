---
description: 'Run the finish-feature checklist: tests + ruff lint/format'
agent: 'agent'
tools: ['runInTerminal', 'edit', 'search/codebase']
---

# Finish Feature Checklist

You are closing out a feature implementation. Do not report the feature as
complete until every step below passes.

## Steps

1. **Identify scope**: determine which files changed for this feature
   (use `git diff --name-only` if unclear).
2. **Tests**: ensure there are tests covering the new/changed behavior.
   - If none exist, write them now under the appropriate `tests/` path.
   - Run the test suite and confirm it passes.
3. **Lint**: run `uv run ruff check .` and fix every finding.
4. **Format**: run `uv run ruff format .`.
5. **Re-verify**: re-run tests and `ruff check .` one final time after any
   fixes to confirm a clean pass.

## Output

Report exactly:
- Which tests were added or already existed
- Test result (pass/fail)
- Ruff check result (clean/fixed N issues)
- Any remaining issues you could not resolve, with a brief explanation