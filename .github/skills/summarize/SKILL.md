---
name: summarize
description: Summarize a file. Use when the user wants a concise explanation of a code file, asks what a file does, or wants the contents, purpose, and I/O of its classes and functions.
---

# Summarize

Use the Explore subagent to summarize one provided file at a time. If no Explore subagent is available, perform the file read and analysis directly.

## Process

1. Anchor on the file path the user provided.
2. Invoke Explore with that file as the entry point and allow it to read nearby imports and referenced definitions as needed.
3. Ask for:
   - a short file-level summary of contents and purpose
   - every class, function, method, and nested helper in scope
   - for each callable or class, the signature, purpose, inputs, outputs/returns, side effects, dependencies, and notable exceptions
   - any ambiguity or unknowns that should be called out instead of guessed
4. Synthesize the subagent findings into a concise human-readable summary.
5. Write the same summary to a markdown artifact in `summaries/`, using a filename that mirrors the source directory structure inside. If the summary already exists, overwrite it.

## Output shape

Return the summary in this order:

1. File overview
2. Classes
3. Functions and methods
4. Notable side effects or dependencies
5. Open questions or ambiguities

Keep the order stable so the same file produces the same section layout on repeated runs.

## Rules

- Include private helpers, nested functions, and module-level definitions.
- Treat I/O as signature plus behavior: parameters, returns, side effects, dependencies, and notable exceptions.
- Do not widen the scope beyond the single requested file unless the file imports or references something that is necessary to explain it.
- If the file cannot be found, return a clear failure instead of inferring the path.
- If the file has no callable definitions, still write a file-level summary and note that explicitly.