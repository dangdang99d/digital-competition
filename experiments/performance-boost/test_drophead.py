"""E49 correctness tests for the DropHead variants — no training, ~1 min on any GPU/CPU.
Loads a real granite (small) and asserts the hook math. Run:
  PYTHONPATH=. python experiments/performance-boost/test_drophead.py
"""
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.finetune import _drophead_keep, install_drophead

M = "ibm-granite/granite-embedding-311m-multilingual-r2"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def fresh():
    m = AutoModelForSequenceClassification.from_pretrained(
        M, num_labels=14, torch_dtype=torch.float32, trust_remote_code=True,
        reference_compile=False, ignore_mismatched_sizes=True).to(DEV).eval()
    return m


def logits(model, enc):
    with torch.no_grad():
        return model(**enc).logits.float().cpu().numpy()


def ok(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    assert cond, name


def main():
    torch.manual_seed(0)
    nH = 12
    print("== unit: _drophead_keep ==")
    x3 = torch.randn(4, 7, 768)          # padded (B,L,H)
    ks = _drophead_keep(x3, nH, 0.3, "sequence", "cpu")
    kt = _drophead_keep(x3, nH, 0.3, "token", "cpu")
    ok("seq mask shape (B,1,nH,1)", tuple(ks.shape) == (4, 1, nH, 1))
    ok("token mask shape (B,L,nH,1)", tuple(kt.shape) == (4, 7, nH, 1))
    ok("never drops all heads (seq)", int(ks.sum(-2).min()) >= 1)
    # drop-rate ≈ p over many samples
    rates = [1 - _drophead_keep(x3, nH, 0.3, "token", "cpu").float().mean().item() for _ in range(200)]
    ok(f"empirical drop-rate≈0.3 ({np.mean(rates):.3f})", abs(np.mean(rates) - 0.3) < 0.03)

    tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
    enc = tok(["hello world foo bar", "another test input here", "third row of text",
               "fourth example sentence"], padding=True, truncation=True,
              max_length=64, return_tensors="pt").to(DEV)

    # ONE model reused: install with infer=False, capture clean base, then toggle the shared
    # state to exercise eval/MC on identical weights (fresh() re-randomizes the head, so a new
    # model would have different weights — the earlier test bug).
    print("== eval invariance (default state: mask off in eval) ==")
    m = fresh(); st = install_drophead(m, 0.3)     # infer=False
    base = logits(m, enc)
    ok("eval == clean (hook no-ops in eval)", np.allclose(logits(m, enc), base, atol=1e-5))

    print("== train mode drops (output changes) ==")
    m2 = fresh(); install_drophead(m2, 0.5); m2.train()
    with torch.no_grad():
        a, b = m2(**enc).logits, m2(**enc).logits
    ok("train forwards differ (stochastic)", not torch.allclose(a, b))

    print("== MC-DropHead: infer=True active at eval, stochastic ==")
    st["infer"] = True                             # same model m, now MC-active
    l1, l2 = logits(m, enc), logits(m, enc)
    ok("infer=True → stochastic at eval", not np.allclose(l1, l2, atol=1e-5))

    # NOTE: DropHead's nH/kept rescaling is NOT per-head unbiased, and downstream LayerNorm/
    # FFN/softmax nonlinearities break E[f(x)]=f(E[x]) — so E[MC]≈clean is NOT expected (and
    # not the point: MC averaging yields a DIFFERENT ensembled prediction). The correct
    # invariant is variance REDUCTION: a K-pass mean is more stable than single passes.
    print("== MC averaging reduces variance (~1/K) ==")
    K = 64

    def mc_mean(kk):
        acc = np.zeros_like(base)
        for _ in range(kk):
            acc += logits(m, enc)
        return acc / kk
    d_single = np.abs(logits(m, enc) - logits(m, enc)).max()
    d_mc = np.abs(mc_mean(K) - mc_mean(K)).max()
    ok(f"K={K} mean more stable than single (d_mc {d_mc:.4f} < d_single {d_single:.4f})",
       d_mc < d_single)

    print("== correlated: same head mask across ALL layers ==")
    m = fresh(); st = install_drophead(m, 0.5, state={"p": 0.5, "infer": True},
                                       correlated=True)
    with torch.no_grad():
        m(**enc)
    ok("correlated cmask populated after forward", st.get("cmask") is not None)
    # after a fresh forward the model-level pre-hook resets, then first layer repopulates
    cm1 = st["cmask"].clone()
    with torch.no_grad():
        m(**enc)
    ok("correlated cmask re-sampled each forward", not torch.equal(cm1, st["cmask"]))

    print("== layer_ramp: deeper layers drop more (per-layer empirical rate) ==")
    m = fresh(); install_drophead(m, 0.6, state={"p": 0.6, "infer": True}, layer_ramp=True)
    perlayer = {}                        # name -> [dropped_head_slots, total] across forwards
    projs = [(n, mod) for n, mod in m.named_modules() if n.endswith(".attn.Wo")]
    def mk(name):
        def h(mod, inp):
            xv = inp[0].view(*inp[0].shape[:-1], 12, -1)
            dropped = (xv.abs().sum(-1) == 0).float().sum().item()
            tot = xv[..., 0].numel()
            a, b = perlayer.get(name, (0.0, 0.0)); perlayer[name] = (a + dropped, b + tot)
        return h
    handles = [mod.register_forward_pre_hook(mk(n)) for n, mod in projs]  # runs AFTER drophead hook (registered later)
    for _ in range(40):
        with torch.no_grad():
            m(**enc)
    for hh in handles:
        hh.remove()
    rates = [perlayer[n][0] / perlayer[n][1] for n, _ in projs]
    ok(f"ramp: last-layer rate {rates[-1]:.3f} > first {rates[0]:.3f}", rates[-1] > rates[0] + 0.1)

    print("== mutual exclusion guard ==")
    try:
        install_drophead(fresh(), 0.3, correlated=True, layer_ramp=True); bad = False
    except AssertionError:
        bad = True
    ok("correlated+layer_ramp raises", bad)

    print("\nALL DROPHEAD TESTS PASSED")


if __name__ == "__main__":
    main()
