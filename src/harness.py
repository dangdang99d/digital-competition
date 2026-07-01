"""Guardrail harness — prompt a code LLM to classify the next action, constrained
to the 14 valid labels. No fine-tuning; you iterate on src/prompt.py only.

GUARDRAIL = constrained decoding by rank-by-logprob: for each sample we score all
14 candidate labels by the model's total log-probability of emitting that label
string as the answer, and pick the argmax. The output is therefore ALWAYS one of
the 14 valid actions — no free generation, no parsing, no retries.

Usage:
  python -m src.harness --model Qwen/Qwen2.5-Coder-1.5B-Instruct --n_dev 1000 --shots 2
  python -m src.harness --model ... --full          # evaluate on the whole 14k val split
Outputs: output/harness_results.csv  (+ per-class F1 and confusions to stdout/log).

You should NOT need to edit this file — edit src/prompt.py.
"""
import argparse
import csv as _csv
import json
import os
import re
import time

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import f1_score

from src.data import ALL_CLASSES, CLASS_TO_ID, load_samples, macro_f1, split_indices
from src.prompt import LABELS, build_prompt, load_prompt


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def load_model(model_id, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model = model.to(device).eval()
    return tok, model


def _render(tok, messages):
    """Apply the chat template, leaving the generation prompt open for the answer."""
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def score_labels(tok, model, messages, label_token_ids, device):
    """Return log-prob of each of the 14 labels as the answer continuation.

    We build the prompt once, then for each label compute the summed log-prob of
    its answer tokens conditioned on the prompt. Labels tokenize to few tokens, so
    this is cheap. Returns np.array of shape (14,).
    """
    prompt_text = _render(tok, messages)
    prompt_ids = tok(prompt_text, return_tensors="pt").input_ids.to(device)
    plen = prompt_ids.shape[1]

    scores = np.empty(len(label_token_ids), dtype=np.float32)
    for i, ans_ids in enumerate(label_token_ids):
        ans = torch.tensor([ans_ids], device=device)
        full = torch.cat([prompt_ids, ans], dim=1)
        logits = model(full).logits  # (1, L, V)
        # logits at position t predict token t+1; answer tokens sit at [plen-1 : -1]
        logp = torch.log_softmax(logits[0, plen - 1:-1], dim=-1)
        tok_logp = logp[torch.arange(len(ans_ids)), torch.tensor(ans_ids, device=device)]
        # sum = log P(label | prompt), the correct quantity. Do NOT length-normalize:
        # labels have different token counts (2-4), and dividing by length biases the
        # ranking toward multi-token labels (e.g. lint_or_typecheck).
        scores[i] = tok_logp.sum().item()
    return scores


def predict_logprob(tok, model, samples, device, spec, shots, max_hist):
    """Fast mode: rank the 14 labels by log P(label|prompt), argmax. No reasoning.

    Returns (preds, records) where records[j] holds top-3 labels+scores for debugging.
    """
    label_token_ids = [tok(" " + lbl, add_special_tokens=False).input_ids for lbl in LABELS]
    preds = np.empty(len(samples), dtype=np.int64)
    records = []
    t0 = time.time()
    for j, s in enumerate(samples):
        msgs = build_prompt(spec, s, shots=shots, max_hist=max_hist)
        sc = score_labels(tok, model, msgs, label_token_ids, device)
        preds[j] = int(sc.argmax())
        order = np.argsort(-sc)[:3]
        records.append({
            "reasoning": "",  # logprob mode has no rationale
            "top3": "; ".join(f"{LABELS[i]}:{sc[i]:.2f}" for i in order),
        })
        if j % 100 == 0:
            logger.info(f"  scored {j+1}/{len(samples)}  ({(j+1)/(time.time()-t0):.1f}/s)")
    return preds, records


_ACTION_RE = re.compile(r"(?i)NEXT ACTION:\s*([a-z_]+)")


def _parse_action(text):
    """Extract the chosen action from generated text. Returns (label or None, raw)."""
    matches = _ACTION_RE.findall(text)
    for cand in reversed(matches):            # last 'NEXT ACTION:' wins
        if cand in LABELS:
            return cand, text
    # fallback: any bare label mentioned in the text (last one)
    found = [lbl for lbl in LABELS if re.search(rf"\b{lbl}\b", text)]
    return (found[-1] if found else None), text


@torch.no_grad()
def predict_generative(tok, model, samples, device, spec, shots, max_hist, max_new_tokens):
    """Generative mode: model writes a short rationale then 'NEXT ACTION: <label>'.

    We parse + validate the label against the 14 classes (fallback to logprob score
    if the model emits nothing valid, so a prediction always exists). Captures the
    full reasoning text per sample for debugging. Returns (preds, records).
    """
    label_token_ids = [tok(" " + lbl, add_special_tokens=False).input_ids for lbl in LABELS]
    preds = np.empty(len(samples), dtype=np.int64)
    records = []
    n_fallback = 0
    t0 = time.time()
    for j, s in enumerate(samples):
        msgs = build_prompt(spec, s, shots=shots, max_hist=max_hist)
        prompt_text = _render(tok, msgs)
        ids = tok(prompt_text, return_tensors="pt").input_ids.to(device)
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        label, raw = _parse_action(gen)
        if label is None:                      # guardrail fallback: no valid action emitted
            n_fallback += 1
            sc = score_labels(tok, model, msgs, label_token_ids, device)
            label = LABELS[int(sc.argmax())]
        preds[j] = CLASS_TO_ID[label]
        records.append({"reasoning": gen.strip().replace("\n", " ⏎ "), "top3": ""})
        if j % 25 == 0:
            logger.info(f"  generated {j+1}/{len(samples)}  ({(j+1)/(time.time()-t0):.1f}/s)")
    if n_fallback:
        logger.warning(f"  {n_fallback}/{len(samples)} samples emitted no valid action -> logprob fallback")
    return preds, records


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------
def evaluate(y_true_ids, y_pred_ids):
    mf1 = macro_f1(y_true_ids, y_pred_ids)
    per_class = f1_score(
        y_true_ids, y_pred_ids, labels=list(range(len(ALL_CLASSES))),
        average=None, zero_division=0,
    )
    return mf1, per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--prompt", default="prompts/default.md",
                    help="markdown prompt file to use (edit THIS, not the code)")
    ap.add_argument("--n_dev", type=int, default=1000,
                    help="stratified dev subset size for cheap prompt iteration")
    ap.add_argument("--full", action="store_true",
                    help="evaluate on the FULL val split instead of the dev subset")
    ap.add_argument("--shots", type=int, default=2)
    ap.add_argument("--max_hist", type=int, default=6)
    ap.add_argument("--mode", choices=["generative", "logprob"], default="generative",
                    help="generative = model writes a rationale then the action (captures "
                         "reasoning for debugging); logprob = fast label ranking, no reasoning")
    ap.add_argument("--max_new_tokens", type=int, default=160,
                    help="generation budget for the rationale + answer (generative mode)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="./output")
    ap.add_argument("--tag", default="", help="label appended to output filenames")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device={device}  model={args.model}  prompt={args.prompt}  "
                f"mode={args.mode}  shots={args.shots}")

    spec = load_prompt(args.prompt)   # parse + validate the markdown prompt
    samples, y = load_samples(args.data_dir)
    y_ids = np.array([CLASS_TO_ID[a] for a in y])
    tr, va = split_indices(y, seed=args.seed)

    # choose eval set: full val, or a stratified dev subset of it (cheap iteration)
    if args.full:
        eval_idx = va
    else:
        rng = np.random.default_rng(args.seed)
        # stratified subsample of the val split so rare classes are represented
        per_class = max(1, args.n_dev // len(ALL_CLASSES))
        picks = []
        for c in range(len(ALL_CLASSES)):
            pool = va[y_ids[va] == c]
            if len(pool):
                picks.append(rng.choice(pool, size=min(per_class, len(pool)), replace=False))
        eval_idx = np.concatenate(picks)
        rng.shuffle(eval_idx)
    logger.info(f"eval on {len(eval_idx)} samples ({'FULL val' if args.full else 'dev subset'})")

    tok, model = load_model(args.model, device)
    eval_samples = [samples[i] for i in eval_idx]
    if args.mode == "generative":
        preds, records = predict_generative(tok, model, eval_samples, device, spec,
                                            args.shots, args.max_hist, args.max_new_tokens)
    else:
        preds, records = predict_logprob(tok, model, eval_samples, device, spec,
                                         args.shots, args.max_hist)
    y_eval = y_ids[eval_idx]

    mf1, per_class = evaluate(y_eval, preds)
    logger.success(f"Macro-F1 = {mf1:.4f}  on {len(eval_idx)} samples")
    print("\nper-class F1 (worst first):")
    order = np.argsort(per_class)
    for c in order:
        flag = "  <-- weak" if per_class[c] < 0.3 else ""
        print(f"  {ALL_CLASSES[c]:18s} {per_class[c]:.3f}{flag}")

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""

    # ---- per-sample DEBUG log: every sample, with reasoning + correctness ----
    # Sorted so wrong predictions come first — read the top of the file to debug.
    from src.data import serialize
    debug_rows = []
    for k, gi in enumerate(eval_idx):
        t_id, p_id = int(y_eval[k]), int(preds[k])
        debug_rows.append({
            "id": samples[gi]["id"],
            "correct": t_id == p_id,
            "true": ALL_CLASSES[t_id],
            "pred": ALL_CLASSES[p_id],
            "reasoning": records[k]["reasoning"],
            "top3": records[k]["top3"],
            "context": serialize(samples[gi], max_hist=args.max_hist).replace("\n", " ⏎ "),
        })
    debug_rows.sort(key=lambda r: r["correct"])   # wrong (False) first
    dbg_csv = os.path.join(args.out_dir, f"harness_debug{tag}.csv")
    with open(dbg_csv, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["id", "correct", "true", "pred", "reasoning", "top3", "context"])
        w.writeheader()
        w.writerows(debug_rows)
    n_wrong = sum(1 for r in debug_rows if not r["correct"])
    logger.info(f"per-sample debug log ({n_wrong} wrong) -> {dbg_csv}")

    # ---- confusion: most common true->pred error pairs ----
    from collections import Counter
    conf = Counter((r["true"], r["pred"]) for r in debug_rows if not r["correct"])
    print("\ntop error pairs (true -> pred):")
    for (t, p), n in conf.most_common(10):
        print(f"  {t:18s} -> {p:18s} x{n}")

    # ---- aggregate result row ----
    row = {
        "model": args.model, "prompt": args.prompt, "mode": args.mode, "shots": args.shots,
        "n_eval": len(eval_idx), "eval_set": "full_val" if args.full else "dev_subset",
        "macro_f1": round(float(mf1), 4),
    }
    out_csv = os.path.join(args.out_dir, "harness_results.csv")
    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    logger.info(f"appended -> {out_csv}")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
