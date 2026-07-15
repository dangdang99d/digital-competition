"""FactoredLinear + whitened-SVD FFN factorization for the Qwen3-0.6B champion (E4).

Materializes E3's training-free whitened-SVD FFN truncation (analysis/palu_probe_ffn.py:
gate/up/down -> rank r costs only -0.26pt @ r=512, +0.4pt @ r=768) as REAL factored
Linear pairs, so a short low-LR recovery fine-tune can close the (small) remaining gap.

Each FFN projection  nn.Linear(in, out)  with weight W is replaced by a
    FactoredLinear:  x -> B(A(x)),   A: in->r (no bias),   B: r->out (orig bias)
with A, B initialised so that

    B.weight @ A.weight  ==  truncate_r(W . L) . L^-1

i.e. the exact E3 whitened-SVD reconstruction, where  L L^T = E[x x^T]  is the
calibration-pass input second-moment of that projection. Because matmul is
associative,  B(A(x))  is byte-identical to replacing W with E3's `reconstruct`
weight -> the factored model reproduces E3's r-truncation val macro-F1 exactly
(training-free), and recovery FT then trains A, B to recover the -0.26pt.

`whitened_svd` / `reconstruct` are copied VERBATIM from analysis/palu_probe_ffn.py
(the E3-validated math -- do NOT edit) so this module is import-self-contained.

Reload story (config.factored_ffn + rebuild_factored/load_factored_model): a plain
AutoModel.from_pretrained rebuilds the STOCK Qwen3 FFN and silently drops the A/B
weights. To reload a factored checkpoint, use `load_factored_model(path)` (or build
the stock model, call `rebuild_factored(model, config.factored_ffn)`, then
load_state_dict) -- mirrors how replace_head's config.custom_head is reloaded.
"""
import os

import torch
import torch.nn as nn

FFN_PROJS = ("gate_proj", "up_proj", "down_proj")   # qwen3 SwiGLU
# ⚠️ mlp-qualified: bare "Wo" also matches ModernBERT's attn.Wo (attention output) —
# that would factor attention too, conflating the FFN axis. Keep the "mlp." prefix.
FFN_PROJS_MODERNBERT = ("mlp.Wi", "mlp.Wo")          # granite/ModernBERT GeGLU (FFN only)


def detect_ffn_projs(model):
    """Pick the FFN Linear suffixes present in this backbone (qwen3 vs granite).
    Low-rank factorization is generic over the Linear shape, so only the module
    names differ — no GeGLU-vs-SwiGLU special-casing needed for the SVD itself."""
    names = {n for n, _ in model.named_modules()}
    if any(n.endswith("gate_proj") for n in names):
        return FFN_PROJS
    if any(n.endswith(("mlp.Wi", "mlp.Wo")) for n in names):
        return FFN_PROJS_MODERNBERT
    raise RuntimeError("no known FFN projection modules (gate/up/down or Wi/Wo) found")


# ---- E3-validated whitened-SVD math (VERBATIM from analysis/palu_probe_ffn.py) ----
def whitened_svd(W, S):
    """Precompute whitened-SVD components once: S=L.L^T, M=W.L=U.diag(s).Vh, Linv=L^-1.
    Truncating to any rank r is then a cheap slice (see reconstruct)."""
    W = W.double()
    dm = S.diag().mean()
    L = None
    for ridge in (1e-4, 1e-3, 1e-2, 1e-1):
        try:
            L = torch.linalg.cholesky(S + ridge * dm * torch.eye(S.shape[0], device=S.device, dtype=S.dtype))
            break
        except Exception:
            continue
    M = W @ L
    U, s, Vh = torch.linalg.svd(M, full_matrices=False)
    Linv = torch.linalg.solve_triangular(L, torch.eye(L.shape[0], device=L.device, dtype=L.dtype), upper=False)
    return U, s, Vh, Linv


def reconstruct(comp, r):
    """Ŵ = truncate_r(W.L).L^-1 from cached components. Returns (What fp32, energy)."""
    U, s, Vh, Linv = comp
    r = min(r, s.shape[0])
    energy = (s[:r].pow(2).sum() / s.pow(2).sum()).item()
    What = (U[:, :r] * s[:r]) @ Vh[:r] @ Linv
    return What.float(), energy
# -----------------------------------------------------------------------------------


class FactoredLinear(nn.Module):
    """Two-matmul, low-rank replacement for nn.Linear(in, out): forward = B(A(x)),
    with A: in->rank (no bias) and B: rank->out (carries the original bias if any),
    so the effective (out, in) weight is  B.weight @ A.weight."""

    def __init__(self, in_features, out_features, rank, bias=False):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.A = nn.Linear(self.in_features, self.rank, bias=False)
        self.B = nn.Linear(self.rank, self.out_features, bias=bias)

    def forward(self, x):
        return self.B(self.A(x))

    @property
    def weight(self):
        """Effective dense weight (out, in) = B @ A -- for introspection/parity checks."""
        return self.B.weight.data @ self.A.weight.data

    @classmethod
    def from_whitened(cls, linear, S, rank):
        """Build from a dense nn.Linear + its input second-moment S = E[x x^T].
        Splits E3's whitened-SVD reconstruction  Ŵ = (U s)[:, :r] @ Vh[:r] @ Linv
        into  B = (U s)[:, :r]  (out, r)  and  A = Vh[:r] @ Linv  (r, in), so
        B @ A == reconstruct(comp, r) exactly (matmul associativity)."""
        W = linear.weight.data                              # (out, in)
        out_f, in_f = W.shape
        r = min(int(rank), out_f, in_f)
        U, s, Vh, Linv = whitened_svd(W.float(), S)
        B = (U[:, :r] * s[:r]).float()                      # (out, r)
        A = (Vh[:r] @ Linv).float()                         # (r, in)
        mod = cls(in_f, out_f, r, bias=linear.bias is not None)
        dtype = W.dtype
        mod.A.weight.data = A.to(dtype)
        mod.B.weight.data = B.to(dtype)
        if linear.bias is not None:
            mod.B.bias.data = linear.bias.data.clone()
        return mod


def _set_submodule(model, qualified_name, new_mod):
    """Replace the module at a dotted name (e.g. 'model.layers.3.mlp.gate_proj')."""
    if "." in qualified_name:
        parent = model.get_submodule(qualified_name.rsplit(".", 1)[0])
        child = qualified_name.rsplit(".", 1)[-1]
    else:
        parent, child = model, qualified_name
    setattr(parent, child, new_mod)


@torch.no_grad()
def collect_ffn_grams(model, enc_list, tok, remap=None, projs=FFN_PROJS, bs=8, device=None):
    """Input second-moment S = E[x x^T] per FFN projection, from a calibration pass
    (same activation-aware whitening as E3's palu_probe_ffn.collect_grams).

    enc_list: list of tokenizer encodings (dicts with input_ids[, attention_mask]).
    remap:    optional LongTensor mapping full-vocab ids -> pruned-vocab ids; pass it
              ONLY for the vocab-pruned champion (E3 setup). None for a full model.
    Returns (grams: {name: (in,in) float64}, targets: {name: nn.Module}).
    """
    from transformers import DataCollatorWithPadding

    device = device or next(model.parameters()).device
    coll = DataCollatorWithPadding(tokenizer=tok)
    targets = {n: m for n, m in model.named_modules() if n.endswith(tuple(projs))}
    assert targets, f"no FFN projections {projs} found in the model"
    grams = {n: torch.zeros(m.weight.shape[1], m.weight.shape[1], dtype=torch.float64, device=device)
             for n, m in targets.items()}
    cnt = {n: 0 for n in targets}

    def mk(n):
        def hook(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).double()
            grams[n] += x.t() @ x
            cnt[n] += x.shape[0]
        return hook

    hooks = [m.register_forward_hook(mk(n)) for n, m in targets.items()]
    try:
        for s in range(0, len(enc_list), bs):
            batch = {k: v.to(device) for k, v in coll(enc_list[s:s + bs]).items()}
            if remap is not None:
                batch["input_ids"] = remap[batch["input_ids"]]
            model(**batch)
    finally:
        for h in hooks:
            h.remove()
    for n in grams:
        grams[n] /= max(1, cnt[n])
    return grams, targets


def factor_ffn_from_grams(model, grams, targets, rank, projs=FFN_PROJS):
    """Replace each FFN projection (in `targets`) with a FactoredLinear initialised
    from its whitened SVD at the given rank. Records model.config.factored_ffn so the
    checkpoint can be reloaded (see load_factored_model). Returns [(name, actual_r)]."""
    done = []
    for name, lin in targets.items():
        fl = FactoredLinear.from_whitened(lin, grams[name], rank)
        _set_submodule(model, name, fl)
        done.append((name, fl.rank))
    if hasattr(model, "config"):
        model.config.factored_ffn = {"rank": int(rank), "projs": list(projs)}
    return done


def build_factored_model(model, tok, calib_texts, rank, max_len=512, remap=None,
                         projs=None, n_calib=256, bs=8, device=None):
    """End-to-end: tokenize calibration texts, collect FFN input grams, and replace
    every FFN projection with its whitened-SVD FactoredLinear at `rank`.

    Mutates `model` in place and returns [(name, actual_r)]. `remap` only for the
    vocab-pruned champion (E3). projs=None -> auto-detect (qwen3 vs granite Wi/Wo).
    """
    if projs is None:
        projs = detect_ffn_projs(model)
    device = device or next(model.parameters()).device
    calib = list(calib_texts)[:n_calib]
    enc = [tok(t, truncation=True, max_length=max_len) for t in calib]
    grams, targets = collect_ffn_grams(model, enc, tok, remap=remap, projs=projs,
                                       bs=bs, device=device)
    return factor_ffn_from_grams(model, grams, targets, rank, projs=projs)


def rebuild_factored(model, factored_cfg):
    """Given a freshly built STOCK model and a recorded `config.factored_ffn` dict,
    swap each FFN projection to an (empty) FactoredLinear of the correct shape so a
    saved factored state_dict loads back. Call BEFORE load_state_dict."""
    rank = int(factored_cfg["rank"])
    projs = tuple(factored_cfg.get("projs", FFN_PROJS))
    for name, lin in list(model.named_modules()):
        if name.endswith(projs) and isinstance(lin, nn.Linear):
            out_f, in_f = lin.weight.shape
            r = min(rank, out_f, in_f)
            _set_submodule(model, name,
                           FactoredLinear(in_f, out_f, r, bias=lin.bias is not None))
    return model


def load_factored_model(path, torch_dtype=torch.float32, **from_config_kwargs):
    """The from_pretrained story: build the stock arch from the saved config, swap in
    FactoredLinear shells per config.factored_ffn, then load the saved weights.
    Returns (model, missing_keys, unexpected_keys). CPU/offline safe (no download)."""
    from transformers import AutoConfig, AutoModelForSequenceClassification

    cfg = AutoConfig.from_pretrained(path)
    factored_cfg = getattr(cfg, "factored_ffn", None)
    model = AutoModelForSequenceClassification.from_config(cfg, **from_config_kwargs)
    model = model.to(torch_dtype)
    if factored_cfg:
        rebuild_factored(model, factored_cfg)
        model = model.to(torch_dtype)
    st_path = os.path.join(path, "model.safetensors")
    if os.path.exists(st_path):
        from safetensors.torch import load_file
        sd = load_file(st_path)
    else:
        sd = torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval(), missing, unexpected
