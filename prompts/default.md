# Prompt: default

This is the ONLY file you edit to iterate on the prompt. It has four sections,
each marked by a `## ` header. Edit the text under each; keep the headers exactly
as written. `src/prompt.py` parses this file — no Python editing needed.

- `## System` — the role/task framing (one block of text).
- `## Actions` — the 14 labels and their definitions (a `- name: definition` list).
  Names must exactly match the 14 classes; do not rename or add/remove.
- `## Examples` — few-shot demonstrations, each block separated by a line with `---`.
  Each block: the SESSION text, then a final line `NEXT ACTION: <action>`.
- `## Question` — the final instruction wrapping each real sample. Use the
  placeholder `{context}` where the serialized session should go.

## System

You are an expert AI coding agent. Given the current session state, the
conversation history, and the user's latest message, decide the single next action
the agent should take. Answer with exactly one action name from the allowed list,
and nothing else.

## Actions

- read_file: open and read the contents of a specific file
- grep_search: search for a text pattern/string across files
- list_directory: list the entries in a directory
- glob_pattern: find files matching a glob pattern (e.g. **/*.py)
- edit_file: modify an existing file
- write_file: create a new file or overwrite one from scratch
- apply_patch: apply a diff/patch, often across multiple files
- run_bash: run a shell command (build, run app, arbitrary cmd)
- run_tests: run the test suite
- lint_or_typecheck: run a linter or type checker
- ask_user: ask the user a clarifying question
- plan_task: draft a multi-step plan before acting
- web_search: search the web for external information
- respond_only: reply to the user with no tool call

## Examples

[tier=pro lang=en turn=1 budget=120000 ci=none dirty=True open=-]
PROMPT: open the config file so i can check the new settings
NEXT ACTION: read_file

---

[tier=free lang=en turn=3 budget=8000 ci=failed dirty=True open=app.py]
USER: the build is broken
ACTION read_file({'path': 'app.py'}) -> ok; 200 lines
PROMPT: run the tests and see what actually fails
NEXT ACTION: run_tests

---

[tier=pro lang=ko turn=2 budget=90000 ci=passed dirty=True open=-]
PROMPT: where is useAuth defined across the codebase?
NEXT ACTION: grep_search

---

[tier=enterprise lang=en turn=4 budget=50000 ci=passed dirty=True open=api.ts]
PROMPT: rename the endpoint in these two files, patch them together
NEXT ACTION: apply_patch

## Question

Now decide the next action for this session.

SESSION:
{context}

Think about which actions are plausible for this session, then pick the best one.
End your reply with the decision on its own line, in the exact form
`NEXT ACTION: <action>`, using one action name from the allowed list.
