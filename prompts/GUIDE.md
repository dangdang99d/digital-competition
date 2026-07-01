# Prompt Engineering — a practical guide (for this task)

You edit `prompts/default.md` (or a copy), run the harness, compare Macro-F1. This
guide teaches the ideas you need, from zero, grounded in *our* task (pick 1 of 14
coding-agent actions). Read top to bottom once; then use it as a checklist.

---

## 0. Mental model: what a prompt actually is

A language model predicts the next tokens given the text you feed it. "Prompt
engineering" = arranging that text so the most likely next tokens are the answer
you want. You are not programming the model; you are **giving it context and
constraints** so the correct answer becomes the path of least resistance.

For us the model never free-writes — the harness constrains output to the 14
labels (rank-by-logprob). So your prompt's job is: make the *right* label score
highest. Everything below serves that.

---

## 1. The high-leverage levers (in rough order of impact)

1. **Clear task instruction** — say exactly what to decide and the output format.
   (`## System` + the final `## Question`.)
2. **Good label definitions** — the model must know what each action *means*,
   especially confusable ones. This is usually the #1 lever for us. (`## Actions`.)
3. **Few-shot examples** — 2–8 labeled demonstrations. Often the single biggest
   accuracy jump. (`## Examples`.) See §3.
4. **Disambiguation hints** — explicit rules for the pairs the model confuses
   (e.g. "prefer grep_search when searching for a symbol, read_file when opening a
   named file"). See §4.
5. **Reasoning** (chain-of-thought) — ask it to think before answering. Powerful
   for hard tasks, but see §5 — it interacts awkwardly with our constrained scorer.

---

## 2. Instruction writing — concrete rules

- **Be specific and imperative.** "Decide the single next action" beats "what
  should happen next?".
- **State the output format explicitly.** "Answer with exactly one action name
  from the allowed list." Even though we constrain decoding, telling the model the
  format still shifts its internal probabilities toward clean answers.
- **Put the most important instruction last.** Models weight the end of the prompt
  heavily. Our `## Question` ends with `NEXT ACTION:` — the model is now primed to
  emit an action.
- **Don't contradict yourself.** Conflicting instructions (e.g. "be brief" +
  "explain your reasoning") confuse the model. Pick one.

---

## 3. Few-shot examples — the biggest practical lever

**What:** include solved examples in the prompt so the model pattern-matches.

**How to choose good ones (this matters more than how many):**
- **Cover the hard/confusable cases**, not the easy ones. One example that
  disambiguates `read_file` vs `grep_search` is worth ten obvious ones.
- **Match the real input format** — our examples use the same `[meta...] USER: ...
  PROMPT: ...` serialization the real samples use. Keep it consistent.
- **Cover rare classes.** Macro-F1 weights all 14 classes equally. If the model
  never predicts `web_search` (1.8% of data), your score tanks. Put a `web_search`
  and an `ask_user` example in.
- **Balance the label variety.** Don't make 3 of your 4 examples `edit_file`, or
  the model over-predicts it.

**How many:** start with 2–4, try 6–8. More = better coverage but longer prompt =
slower + can dilute. Our harness has a `--shots N` flag so you can measure it
directly. Diminishing returns usually kick in by ~8.

**Ordering effect:** models are slightly biased toward labels seen recently /
frequently in the examples ("recency" and "majority-label" bias). Shuffle example
order and keep the label mix even.

---

## 4. Disambiguation rules — cheap accuracy for known confusions

Our EDA found the model most confuses *within-family* pairs. Add a short rules
block (in `## System` or before `## Question`) like:

```
Guidelines:
- Searching for a symbol/pattern across files -> grep_search (not read_file).
- Opening one named file to view it -> read_file.
- Changing an existing file -> edit_file; a diff across several files -> apply_patch.
- Creating a brand-new file / full overwrite -> write_file.
- Running the app or an arbitrary command -> run_bash; running the tests -> run_tests.
- Only reply, no tool needed -> respond_only.
```

Target the specific mistakes you SEE in the per-class F1 / confusions the harness
prints. Don't write rules for confusions that aren't happening.

---

## 5. Chain-of-thought (reasoning) — powerful but handle with care

"Let's think step by step" before answering often helps hard classification. BUT
our guardrail scores the label directly after the prompt, so free reasoning
doesn't fit the simple rank-by-logprob path. Two options if you want it:
- Keep it simple first (no CoT) — get a baseline.
- If you want CoT later, we'd switch the harness to *generate* a short rationale
  then a final `NEXT ACTION:` and parse it. Tell me and I'll add that mode.

Start without CoT. Add it only if the simple version plateaus.

---

## 6. The iteration loop (how to actually work)

1. Edit `prompts/default.md` (or copy to `prompts/v2.md`).
2. Run on the **dev subset** (fast, ~1000 rows):
   `python -m src.harness --prompt prompts/v2.md --n_dev 1000 --shots 4`
3. Read the **per-class F1** the harness prints. Find the worst classes.
4. Improve the prompt for those specific classes (better definition, a targeted
   example, a disambiguation rule). Change **one thing at a time** so you know what
   helped.
5. Repeat. When a prompt clearly wins on dev, confirm on full val: add `--full`.

**Change one variable at a time.** If you edit definitions AND add examples AND
change instructions together, you won't know which helped. Scientific method.

**Keep versions.** Copy to `prompts/v1.md`, `v2.md`, ... so you can compare and
revert. The harness logs which prompt file produced each score in
`output/harness_results.csv`.

---

## 7. Common failure modes to watch for

- **Over-predicting the majority class** (`edit_file`): add examples of the classes
  it's missing; add disambiguation rules.
- **Rare classes at F1=0**: the model never picks them → add a clear definition +
  one example each. Biggest Macro-F1 killer.
- **Format drift**: not an issue for us (constrained decoding), but keep the
  `NEXT ACTION:` cue so the model is primed correctly.
- **Prompt too long**: dilutes attention and slows inference. Trim examples that
  aren't earning their keep.

---

## 8. External resources (learn more)

- **Anthropic prompt engineering guide** — the best practical intro:
  https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview
- **OpenAI prompt engineering guide** — concise, general:
  https://platform.openai.com/docs/guides/prompt-engineering
- **Google prompt engineering whitepaper** (Kaggle, 2024) — thorough, free PDF.
- **learnprompting.org** — free structured course, beginner-friendly.
- **Brex prompt engineering guide** (GitHub) — good for the "why", engineer-focused.
- Papers if curious: "Language Models are Few-Shot Learners" (GPT-3, few-shot),
  "Chain-of-Thought Prompting" (Wei et al. 2022), "Calibrate Before Use"
  (Zhao et al. 2021 — the majority/recency bias in few-shot).

---

## TL;DR checklist for each prompt edit

- [ ] Clear task + output format, important bit last.
- [ ] All 14 definitions clear; extra care on confusable/rare ones.
- [ ] 2–8 few-shot examples: cover hard + rare classes, balanced labels, real format.
- [ ] Disambiguation rules only for confusions you actually observe.
- [ ] Change ONE thing per iteration; test on dev subset; confirm winner on --full.
- [ ] Watch per-class F1 for rare classes stuck at 0.
