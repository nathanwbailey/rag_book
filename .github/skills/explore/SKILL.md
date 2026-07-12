---
name: explore
description: Walk a codebase and return focused context about structure, ownership, and flow. Use when the user asks to explore the codebase, find where something lives, trace how parts connect, or gather edit context before making changes.
---

# Explore

Use the Explore subagent for read-only codebase exploration.

## Quick Start

Start from the user's anchor, then ask the subagent for the smallest useful map.

1. Pick the anchor: file, symbol, feature, error, command, config key, or directory.
2. Invoke the Explore agent with the topic, the likely entry point if known, and the desired thoroughness: quick, medium, or thorough.
3. Ask for concise findings: key files, the main relationships or flow, the next best place to look, and any uncertainty.

## Workflows

### Structure Discovery

Use when the user wants to know where something lives or how a feature is laid out.
(where does X live / how is this feature organized?)

1. Start from the narrowest concrete anchor available.
2. Ask for the owning files, nearby types or modules, and the next file worth opening.
3. Stop when the subagent can name the main entry points and the file that most directly owns the topic.

### Call-Flow Tracing

Use when the user wants to follow how a request, event, or value moves through the code.
(how does a request/event/value move through the code?)

1. Start from the call site, handler, or command the user named.
2. Ask the subagent to trace the path through the controlling code and report each hop that changes behavior.
3. Stop when the subagent can explain the controlling path and where the flow terminates or branches.

### Edit-Context Lookup

Use when the user is about to modify code and needs the nearest safe place to edit.
(I'm about to change something, where's the safest place to do it?")

1. Start from the failing behavior or the feature being changed.
2. Ask for the smallest local context that controls the behavior and the file that should be edited first.
3. Stop when the subagent can point to the controlling abstraction and the likely edit surface.

## Prompt shape

- Include the topic to explore.
- Include any likely entry point or file if known.
- Say whether the goal is structure, ownership, call flow, or edit context.
- Ask for file references in the response.

## Output

Return a short summary with the most relevant file references, the main relationship or flow found, the next best place to look, and any uncertainty that remains.