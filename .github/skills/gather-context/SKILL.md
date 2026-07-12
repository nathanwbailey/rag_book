---
name: gather-context
description: Gather the nearest useful file context before answering questions or making changes. Use when you need a matching instruction file, an existing summary, or fresh local exploration for a file.
---

# Gather Context

Follow this order and stop at the first source that is sufficient:

1. Check whether a `.github/instructions/*.instructions.md` file with an
   `applyTo` pattern matching the target file is already in this conversation's
   context.
2. If the target is a repository file, check whether a summary already exists
   at the mirrored path under `summaries/`.
3. If neither source is sufficient, invoke the Explore skill with the target
   file as the anchor and the smallest useful local context as the goal, then
   invoke the Summarize skill to write a reusable summary.

## Completion Criterion

Stop when you can point to the instruction file, summary, or explored context
that answers the question, or when Explore plus Summarize has produced it.