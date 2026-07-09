"""Reconstruct full sessions from the overlapping per-step training samples.

train.jsonl slices each simulated session into one sample per step: step N's
history is a sliding window of the last 12 events before its prompt, and each
step appends exactly one (user, assistant_action) pair.  Placement is therefore
pure arithmetic (verified: adjacent samples' windows agree exactly):

  step N sample:  history -> absolute event indices 2(N-1)-len(h) .. 2(N-1)-1
                  prompt  -> user event at index 2(N-1)
                  label   -> action event at index 2N-1 (name only; args appear
                             once a step>=N+1 sample's history covers it)

Writes one JSON per session to --out_dir:

  {
    "session_id": ...,
    "n_steps":    highest step seen,
    "events":     [ {role:user|assistant_action, ...} ... ]   full timeline;
                  action events carry args+result_summary when any sample's
                  history observed them, else {"known_from": "label_only"}
    "steps":      [ {step, id, split, label, session_meta} ... ]  one entry per
                  sample, with that step's full metadata snapshot
  }

Usage:  python -m src.rebuild_sessions [--data_dir ./data] [--out_dir ./output/sessions]
"""
import argparse
import csv
import json
import os
from collections import defaultdict

from loguru import logger


def load_split(path, split, sessions):
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            sid, step = o["id"].rsplit("-step_", 1)
            o["_step"], o["_split"] = int(step), split
            sessions[sid].append(o)
            n += 1
    return n


def place(events, idx, ev, conflicts):
    """Insert event at absolute index, verifying agreement on overlap."""
    old = events.get(idx)
    if old is None:
        events[idx] = ev
    elif old != ev:
        # a fuller action event (args from history) beats a label-only stub
        if old.get("known_from") == "label_only" and ev["role"] == "assistant_action":
            events[idx] = ev
        elif ev.get("known_from") == "label_only":
            pass
        else:
            conflicts.append((idx, old, ev))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--out_dir", default="./output/sessions")
    args = ap.parse_args()
    from src.runlog import log_cmd
    log_cmd()

    labels = {}
    with open(os.path.join(args.data_dir, "train_labels.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["id"]] = row["action"]

    sessions = defaultdict(list)
    n_tr = load_split(os.path.join(args.data_dir, "train.jsonl"), "train", sessions)
    n_te = load_split(os.path.join(args.data_dir, "test.jsonl"), "test", sessions)
    logger.info(f"{n_tr} train + {n_te} test samples across {len(sessions)} sessions")

    os.makedirs(args.out_dir, exist_ok=True)
    all_conflicts, n_label_only, n_actions, n_missing = 0, 0, 0, 0
    for sid, samples in sessions.items():
        samples.sort(key=lambda o: o["_step"])
        events, conflicts, steps = {}, [], []
        for o in samples:
            base = 2 * (o["_step"] - 1)
            for j, ev in enumerate(o["history"]):
                place(events, base - len(o["history"]) + j, ev, conflicts)
            place(events, base, {"role": "user", "content": o["current_prompt"]},
                  conflicts)
            lab = labels.get(o["id"])
            if lab:
                place(events, base + 1,
                      {"role": "assistant_action", "name": lab, "args": None,
                       "result_summary": None, "known_from": "label_only"},
                      conflicts)
            steps.append({"step": o["_step"], "id": o["id"], "split": o["_split"],
                          "label": lab, "session_meta": o["session_meta"]})
        # not every step of a session is sampled; slots the 12-event window
        # never reached stay unobserved — keep them as explicit placeholders
        n_slots = max(events) + 1
        timeline = [events.get(i, {"role": "missing", "known_from": "unobserved"})
                    for i in range(n_slots)]
        n_actions += sum(e["role"] == "assistant_action" for e in timeline)
        n_label_only += sum(e.get("known_from") == "label_only" for e in timeline)
        n_missing += sum(e["role"] == "missing" for e in timeline)
        all_conflicts += len(conflicts)
        with open(os.path.join(args.out_dir, f"{sid}.json"), "w", encoding="utf-8") as f:
            json.dump({"session_id": sid, "n_steps": samples[-1]["_step"],
                       "events": timeline, "steps": steps},
                      f, ensure_ascii=False, indent=2)

    logger.info(f"action events: {n_actions}, of which label-only "
                f"(args never observed): {n_label_only}; unobserved slots: {n_missing}")
    if all_conflicts:
        logger.warning(f"{all_conflicts} overlap conflicts — data not as regular as assumed!")
    logger.success(f"{len(sessions)} session files -> {args.out_dir}")


if __name__ == "__main__":
    main()
