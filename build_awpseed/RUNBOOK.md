# AWP seed-fleet submission runbook (deadline build)

Everything runs ON THE BOX where the seed models trained (repo at `/workspace/repo`,
this folder arrives via `git pull`). Two shapes; pick when scores are in.

## 0. Collect scores when runs finish (`ls /workspace/logs/awpfd_*.DONE | wc -l` = 16)

```bash
cd /workspace/repo
grep -H "best val Macro-F1" /workspace/logs/awpfd_s*.log | sed 's/.*awpfd_/awpfd_/' | sort -t= -k2 -rn
```
Top lines = best seeds by the 3.5k held-out.

## A. 3-seed uniform ensemble (RECOMMENDED — ~910MB zip, ~6:39 server)

```bash
cd /workspace/repo
export PYTHONPATH=. HF_HOME=/workspace/.hf_home
# EDIT the three seeds:
S0=42; S1=7; S2=13
P=output/awp_seeds/ft_ibm-granite__granite-embedding-311m-multilingual-r2_awpfd_s
for i in 0 1 2; do
  eval S=\$S$i
  /venv/main/bin/python build_awpseed/prep_member.py \
    --ckpt ${P}${S} --out build_awpseed/model/member_${i}_awpfd_s${S} --verify
done
cd build_awpseed
rm -f ../submit_0715_awpseed3.zip
zip -r -0 -X ../submit_0715_awpseed3.zip script.py requirements.txt model \
  -x "model/*template*"
cd ..; ls -la submit_0715_awpseed3.zip; unzip -l submit_0715_awpseed3.zip | tail -5
```
Filename `submit_0715_awpseed3.zip` = 23 chars (cap 32) ✓. Size must print < 1GB.

## B. best single seed (~630MB zip, ~5:04 server)

```bash
cd /workspace/repo
BEST=42   # EDIT: best seed from step 0
P=output/awp_seeds/ft_ibm-granite__granite-embedding-311m-multilingual-r2_awpfd_s${BEST}
mkdir -p build_single/model && cp -r ${P} build_single/model/granite-311m-e8a-ls-awpseed
# single-model script/req/variant come from the drophd/awp_t031 lineage:
cp build_drophd/script.py build_drophd/requirements.txt build_drophd/serialize_variant.json build_single/ 2>/dev/null \
  || echo "on-box: pull script.py etc from the repo build_drophd/ (committed)"
cd build_single
rm -f ../submit_0715_awpbest.zip
zip -r -0 -X ../submit_0715_awpbest.zip serialize_variant.json requirements.txt script.py model
cd ..; ls -la submit_0715_awpbest.zip
```

## Integrity gate (either shape) — MUST pass before download
- `prep_member.py --verify` printed `PARITY OK` for every member (shape A), and/or
- run the zip's script against data/: `cd build_awpseed && ln -s ../data data && mkdir -p output && /venv/main/bin/python script.py` → prints `Saved output/submission.csv rows=5`.

## Notes
- Members are pruned with `keep_ids_ordered.npy` — recovered from the SHIPPED e38 pair
  member and verified byte-exact; remap/tokenizer are the same files that already ran
  on the server (e38 pair/trio zips) → server-side behavior is proven.
- NO logit calibration anywhere; raw argmax; richargs serialization.
- Download the zip via the vast console / your fast path, upload to DACON. 32-char
  filename cap includes `.zip`.

## Challenger models (fd_* on the other box)
Same flows work — for shape A point `--ckpt` at `output/e50fd/ft_..._fd_<tag>`; members
can MIX (e.g. 2×AWP seed + 1×ramp) — the script averages softmax uniformly regardless.
