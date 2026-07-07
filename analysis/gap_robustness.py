"""E6 robustness — is the two-model per-class-tau fallback (champion qwen3 -> hist0
when qwen3's per-class gap < tau_c) robust, or was its +0.0049 gain slice-specific?

Two robustness slices, all UNCALIBRATED, honest (fit tau on disjoint data from eval):
  1. by generator : fit per-class tau on ONE generator subset (sim | au), evaluate on
                    the OTHER held-out generator. Cross-generator generalization.
  2. by step      : first-step / zero-history (step_01) vs later steps (step>=2).
                    Honest 2-fold for tau WITHIN each stratum.

Baseline throughout: single-model qwen3 argmax = 0.7682 (global, uncal). We also
report each slice's own within-slice qwen3 base, since the honest question is whether
the fallback still ADDS gain over single-model qwen3 on that held-out slice.

Reuses top12/macro/TAUS and the per-class tau fit/apply logic from gap_fix.py.
"""
import re
import numpy as np
import matplotlib.pyplot as plt

from src.data import ALL_CLASSES, CLASS_TO_ID, load_samples, split_indices
from analysis.gap_fix import top12, macro, TAUS

QWEN3_GLOBAL_BASE = 0.7682  # single-model champion, uncal, held-out (project baseline)

_ID = re.compile(r"sess_([a-z]+)_.*-step_(\d+)")


# ---- per-class tau fallback (identical logic to gap_fix.perclass_variants) ----
def fit_perclass_tau(y, t1, alt, gap, fit):
    """Greedy per-class tau: for each predicted class, pick tau maximizing fit-half
    macro-F1 (other classes' taus fixed, one pass). When gap<tau_c, use alt argmax."""
    taus = np.zeros(len(ALL_CLASSES))
    p = t1[fit].copy()
    best = macro(y[fit], p)
    for c in range(len(ALL_CLASSES)):
        cur_best, cur_tau = best, 0.0
        for tau in TAUS:
            p_try = p.copy()
            m = (t1[fit] == c) & (gap[fit] < tau)
            p_try[m] = alt[fit][m]
            f = macro(y[fit], p_try)
            if f > cur_best:
                cur_best, cur_tau = f, tau
        taus[c] = cur_tau
        m = (t1[fit] == c) & (gap[fit] < cur_tau)
        p[m] = alt[fit][m]
        best = cur_best
    return taus


def apply_perclass(t1, alt, gap, ev, taus):
    p = t1[ev].copy()
    for c in range(len(ALL_CLASSES)):
        m = (t1[ev] == c) & (gap[ev] < taus[c])
        p[m] = alt[ev][m]
    return p


def fallback_fit_eval(y, t1a, gapa, altb, fit, ev):
    """Fit per-class tau on `fit` indices, evaluate qwen3->hist0 fallback on `ev`.
    Returns (base_ev_macroF1, fallback_ev_macroF1)."""
    taus = fit_perclass_tau(y, t1a, altb, gapa, fit)
    base = macro(y[ev], t1a[ev])
    fb = macro(y[ev], apply_perclass(t1a, altb, gapa, ev, taus))
    return base, fb


def main():
    samples, y_all = load_samples("./data")
    _, va = split_indices(y_all, seed=42)
    y = np.array([CLASS_TO_ID[y_all[i]] for i in va])

    gen = np.array([_ID.match(samples[i]["id"]).group(1) for i in va])
    step = np.array([int(_ID.match(samples[i]["id"]).group(2)) for i in va])

    L = {m: np.load(f"analysis/cache/{m}_val_logits.npz")["logits"] for m in ("hist0", "qwen3")}
    T = {m: top12(L[m]) for m in L}
    # champion: qwen3 top-1 with qwen3 gap, fall back to hist0 top-1
    t1q, _, gpq = T["qwen3"]
    t1h = T["hist0"][0]

    idx = np.arange(len(va))
    rng = np.random.RandomState(0)
    fold = rng.randint(0, 2, size=len(va))

    rows = []  # (slice_label, n_eval, base_f1, fallback_f1)

    # ---- reference: overall honest 2-fold (reproduce champion) ----
    sc_b, sc_f = [], []
    for e in (0, 1):
        fit, ev = idx[fold != e], idx[fold == e]
        b, f = fallback_fit_eval(y, t1q, gpq, t1h, fit, ev)
        sc_b.append(b); sc_f.append(f)
    print("=== reference (overall, honest 2-fold) ===")
    print(f"qwen3 base (global)         : {np.mean(sc_b):.4f}")
    print(f"fallback (per-class tau)    : {np.mean(sc_f):.4f}  ({np.mean(sc_f)-np.mean(sc_b):+.4f})")
    rows.append(("overall", len(va), np.mean(sc_b), np.mean(sc_f)))

    # ---- 1. by generator: fit on one generator, eval on the OTHER (held-out) ----
    print("\n=== 1. by generator (fit on one, eval on held-out other) ===")
    gen_rows = []
    for fit_g, ev_g in (("sim", "au"), ("au", "sim")):
        fit = idx[gen == fit_g]
        ev = idx[gen == ev_g]
        b, f = fallback_fit_eval(y, t1q, gpq, t1h, fit, ev)
        print(f"tau fit on {fit_g:>3} (n={len(fit):5d}) -> eval on {ev_g:>3} (n={len(ev):5d}): "
              f"base={b:.4f}  fallback={f:.4f}  ({f-b:+.4f} vs slice base, "
              f"{f-QWEN3_GLOBAL_BASE:+.4f} vs global 0.7682)")
        gen_rows.append((f"gen: eval={ev_g}\n(tau fit on {fit_g})", len(ev), b, f))
    rows += gen_rows

    # ---- 2. by step: first-step (zero-history) vs later, honest 2-fold within stratum ----
    print("\n=== 2. by step / history length (honest 2-fold within stratum) ===")
    strata = [("first-step (step==1, zero-hist)", step == 1),
              ("later steps (step>=2)", step >= 2)]
    step_rows = []
    for label, mask in strata:
        s_idx = idx[mask]
        s_fold = fold[mask]
        b_list, f_list = [], []
        for e in (0, 1):
            fit = s_idx[s_fold != e]
            ev = s_idx[s_fold == e]
            b, f = fallback_fit_eval(y, t1q, gpq, t1h, fit, ev)
            b_list.append(b); f_list.append(f)
        bmean, fmean = np.mean(b_list), np.mean(f_list)
        print(f"{label:34s} n={int(mask.sum()):5d}: base={bmean:.4f}  fallback={fmean:.4f}  "
              f"({fmean-bmean:+.4f} vs slice base, {fmean-QWEN3_GLOBAL_BASE:+.4f} vs global 0.7682)")
        short = "first-step\n(zero-hist)" if "first" in label else "later steps\n(step>=2)"
        step_rows.append((short, int(mask.sum()), bmean, fmean))
    rows += step_rows

    make_figure(gen_rows, step_rows, np.mean(sc_b), np.mean(sc_f))
    return rows


def make_figure(gen_rows, step_rows, overall_base, overall_fb):
    BG, BLUE, RED = "#fcfcfb", "#3987e5", "#d4553f"
    plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG,
                         "savefig.facecolor": BG, "font.size": 10})

    labels = ["overall"] + [r[0] for r in gen_rows] + [r[0] for r in step_rows]
    bases = [overall_base] + [r[2] for r in gen_rows] + [r[2] for r in step_rows]
    fbs = [overall_fb] + [r[3] for r in gen_rows] + [r[3] for r in step_rows]

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.4))
    b1 = ax.bar(x - w / 2, bases, w, label="single-model qwen3 (slice base)",
                color=BLUE, alpha=0.55, edgecolor="white", linewidth=0.6)
    b2 = ax.bar(x + w / 2, fbs, w, label="two-model per-class-tau fallback",
                color=BLUE, edgecolor="white", linewidth=0.6)

    ax.axhline(QWEN3_GLOBAL_BASE, color=RED, lw=1.6, ls="--",
               label=f"global qwen3 baseline {QWEN3_GLOBAL_BASE:.4f}")

    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", (rect.get_x() + rect.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=7.5, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.3)
    ax.set_ylabel("held-out macro-F1 (uncalibrated)")
    ax.set_title("E6 robustness: two-model per-class-τ fallback vs single-model qwen3\n"
                 "(honest — τ fit on data disjoint from each eval slice)", fontsize=11)
    lo = min(min(bases), min(fbs), QWEN3_GLOBAL_BASE)
    hi = max(max(bases), max(fbs))
    ax.set_ylim(lo - 0.03, hi + 0.03)
    ax.grid(axis="y", color="#cfcfcf", lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    out = "experiments/deferral/figures/e6_robustness.png"
    import os
    os.makedirs("experiments/deferral/figures", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nsaved figure -> {out}")


if __name__ == "__main__":
    main()
