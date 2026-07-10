# -*- coding: utf-8 -*-
"""Builds train_and_analyze_4gpu.ipynb: parameterized (papermill) fold-parallel version
of the user's train_and_analyze.ipynb. Original logic preserved; training split into
per-fold, and OOF/analysis/ensemble/submission gated behind RUN_FINALIZE."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import sys

nb = new_notebook()
cells = []

def md(src): cells.append(new_markdown_cell(src))
def code(src, tags=None):
    c = new_code_cell(src)
    if tags: c.metadata["tags"] = tags
    cells.append(c)

md("""# Action 분류 — 4-GPU 폴드병렬 실행판 (papermill 파라미터화)

원본 `train_and_analyze.ipynb`와 **동일 로직**. 차이점만:
- **papermill 파라미터**로 fold 하나만 학습(`FOLD=k`)하거나, 저장된 fold들로 최종화(`RUN_FINALIZE=True`).
- 결과물은 전부 `RESULTS_DIR` 아래에 저장 (models / oof / analysis / submission).
- sbatch가 fold0~4를 GPU 4대에 뿌리고, 마지막에 finalize 1회 실행.

리키지-프리 세션 그룹 5-fold + OOF + logit_bias + 분석 + 앙상블 → submission (원본과 동일).""")

# ---- parameters cell (papermill injects overrides right after this) ----
code('''# === papermill parameters (런타임에 덮어씀) ===
DEV_MODE        = False   # True면 소규모 점검
FOLD            = None    # int -> 해당 fold만 학습 / None -> 학습 안 함
RUN_FINALIZE    = True    # True -> 저장된 fold들로 OOF/분석/앙상블/submission
RESULTS_DIR     = None    # 결과 저장 루트 (None이면 ./results/local)
BASE_MODEL_PATH = None    # 로컬 베이스 모델 경로 (있으면 오프라인 로드)
DEV_SAMPLES     = 3000
DROP_META_FIELDS = []     # 입력에서 뺄 meta 피처 키 목록 (ablation). 예: ["git","ci","pref"]
FOLDS_TO_FINALIZE = None  # finalize에서 조립할 fold 목록 (None이면 0..N_SPLITS-1 자동)
TRAIN_BS = 16             # per_device train batch (멀티GPU DataParallel이면 ×n_gpu가 글로벌 배치)
EVAL_BS  = 64             # per_device eval batch
GRAD_ACCUM = 1            # gradient accumulation (큰 모델이면 배치↓ + accum↑로 유효배치 유지)
ADD_DERIVED = False       # 파생피처(A) 접두: [LAST][PREV3][NHIST][NEDIT][NREAD][NSEARCH][NTEST][STEP]
CLASS_WEIGHT = False      # sqrt 역빈도 class weight (WeightedTrainer)
OPENFILES = "off"         # open_files 추가 모드: "off"|"ext"(확장자만)|"names"(파일명만)|"both"
OPENDIR = False           # True면 open_files 디렉토리 경로를 opendir= 로 추가 (파일명과 별개)
HISTFILES = "off"         # "names"면 history args에서 언급된 파일 basename을 histfiles= 로 추가 (openfiles 확장)
HISTPATH = "full"         # history args 디렉토리 제거 모드:
#   "full" = 그대로 / "base" = path/file/scope/target 전부 basename
#   "files" = 파일경로만(read/edit/write.path, run_tests/lint.target) basename. list_directory.path·grep scope는 보존(디렉토리가 의미).
TURN_REPR = "raw"         # turn 표현: "raw"(현재 숫자)|"cap8"(min(n,8))|"bucket"(1..8,9_11,12_14,15p)
LORA = False              # True면 LoRA로 학습(어댑터만) 후 merge 저장 (SOTA 방식)
LORA_R = 16               # LoRA rank
LORA_ALPHA = 32           # LoRA alpha
LR = 2e-5                 # 학습률 (LoRA면 2e-4 권장)
EPOCHS = 3                # 에폭 (early stop과 함께 늘려 씀)
EARLY_STOP = 0            # >0이면 EarlyStoppingCallback patience (eval_macro_f1 기준)
SEED_OVERRIDE = None      # None이면 42. 값 주면 학습 seed(헤드 init+shuffle)만 바꿈. split은 항상 42 고정.
MAX_HISTORY_P = None      # None이면 12. full history면 크게(예: 40) -> 모든 이벤트 포함.
MAX_LENGTH_P = None       # None이면 512. history 늘릴 때 키움(모델 max_position 이내).
DATA_DIR_P = None         # None이면 PROJECT_DIR/data. 값 주면 그 경로에서 train/test 읽음(예: 영어 번역본).
''', tags=["parameters"])

# ---- config ----
code('''# ===== Config =====
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from pathlib import Path

PROJECT_DIR = Path("/data/kimin10866/repos/데이콘")
DATA_DIR    = Path(DATA_DIR_P) if globals().get("DATA_DIR_P") else (PROJECT_DIR / "data")

# 베이스 모델: 로컬 경로가 주어지면 오프라인 로드, 아니면 HF 허브
if BASE_MODEL_PATH:
    BASE_MODEL = str(BASE_MODEL_PATH)
    os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
    LOCAL_ONLY = True
else:
    BASE_MODEL = "ibm-granite/granite-embedding-311m-multilingual-r2"
    os.environ["HF_HUB_OFFLINE"] = "0"; os.environ["TRANSFORMERS_OFFLINE"] = "0"
    LOCAL_ONLY = False

# 결과 디렉토리 구조
RESULTS_DIR = Path(RESULTS_DIR) if RESULTS_DIR else (PROJECT_DIR / "results" / "local")
MODEL_ROOT   = RESULTS_DIR / "models"
OOF_DIR      = RESULTS_DIR / "oof"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
SUB_DIR      = RESULTS_DIR / "submission"
for d in (MODEL_ROOT, OOF_DIR, ANALYSIS_DIR, SUB_DIR):
    d.mkdir(parents=True, exist_ok=True)

MODEL_PREFIX = "granite-311m-v2"
MAX_LENGTH          = 512 if globals().get("MAX_LENGTH_P") is None else int(MAX_LENGTH_P)
MAX_HISTORY_EVENTS  = 12 if globals().get("MAX_HISTORY_P") is None else int(MAX_HISTORY_P)
print("MAX_LENGTH:", MAX_LENGTH, "| MAX_HISTORY_EVENTS:", MAX_HISTORY_EVENTS)
N_SPLITS = 5
SEED     = 42          # split(KFold) seed — 항상 고정 (fold별 val 동일 = 공정 비교)
TRAIN_SEED = SEED if globals().get("SEED_OVERRIDE") is None else int(SEED_OVERRIDE)  # 학습 randomness만
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
# LR / EPOCHS / TRAIN_BS / EVAL_BS 는 파라미터 셀에서 정의됨

ACTION_CLASSES = ["read_file","grep_search","list_directory","glob_pattern","edit_file",
                  "write_file","apply_patch","run_bash","run_tests","lint_or_typecheck",
                  "ask_user","plan_task","web_search","respond_only"]
label2id = {c:i for i,c in enumerate(ACTION_CLASSES)}
id2label = {i:c for c,i in label2id.items()}
NUM_CLASSES = len(ACTION_CLASSES)
print("classes:", NUM_CLASSES, "| DEV_MODE:", DEV_MODE, "| FOLD:", FOLD, "| FINALIZE:", RUN_FINALIZE)
print("BASE_MODEL:", BASE_MODEL, "| LOCAL_ONLY:", LOCAL_ONLY)
print("RESULTS_DIR:", RESULTS_DIR)
''')

# ---- repro + device ----
code('''# ===== 재현성 + 디바이스 =====
import json, csv, re, random
import numpy as np, torch
from transformers import set_seed
set_seed(TRAIN_SEED); random.seed(TRAIN_SEED); np.random.seed(TRAIN_SEED); torch.manual_seed(TRAIN_SEED)
print("TRAIN_SEED:", TRAIN_SEED, "| split SEED:", SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
if device.type != "cuda":
    print("\\u26a0\\ufe0f GPU 아님 — 학습이 매우 느림.")
''')

# ---- render_sample (identical to original) ----
code('''# ===== 입력 렌더링 (baseline script.py와 동일 로직) =====
def _safe_text(v):
    return "" if v is None else (v if isinstance(v, str) else str(v))

def _compact_json(v):
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def _budget_bucket(t):
    try: t = int(t)
    except (TypeError, ValueError): return "unknown"
    return "very_low" if t<2000 else "low" if t<10000 else "medium" if t<50000 else "high"

def _elapsed_bucket(s):
    try: s = int(s)
    except (TypeError, ValueError): return "unknown"
    return "early" if s<120 else "mid" if s<900 else "late"

def _turn_tok(t):
    # TURN_REPR: raw(현재)/cap8(min(n,8))/bucket(1..8,9_11,12_14,15p)
    mode = globals().get("TURN_REPR", "raw")
    try: n = int(t)
    except (TypeError, ValueError): return _safe_text(t)
    if mode == "cap8":   return str(min(n, 8))
    if mode == "bucket":
        if n <= 8:  return str(n)
        if n <= 11: return "9_11"
        if n <= 14: return "12_14"
        return "15p"
    if mode == "sem":    # 의미 버킷: 세션 단계 (분포 패턴 반영)
        if n <= 1: return "start"
        if n <= 3: return "early"
        if n <= 8: return "mid"
        return "late"
    return str(n)

# 파생피처(A) — train_derived_cw.py 와 동일 로직 (각 샘플 history/id 에서만 계산 = 누수 아님)
_EDIT   = {"edit_file", "write_file", "apply_patch"}
_READ   = {"read_file"}
_SEARCH = {"grep_search", "glob_pattern", "list_directory"}
_TEST   = {"run_tests", "lint_or_typecheck"}

def derived_fields(sample):
    hist = sample.get("history") or []
    actions = [it.get("name") for it in hist
               if it.get("role") == "assistant_action" and it.get("name")]
    last  = actions[-1] if actions else "START"
    prev3 = ">".join(actions[-3:]) if actions else "START"
    n_edit   = sum(a in _EDIT   for a in actions)
    n_read   = sum(a in _READ   for a in actions)
    n_search = sum(a in _SEARCH for a in actions)
    n_test   = sum(a in _TEST   for a in actions)
    m = re.search(r"step_(\\d+)", _safe_text(sample.get("id")))
    step = m.group(1) if m else "?"
    return " ".join([
        f"[LAST={last}]", f"[PREV3={prev3}]", f"[NHIST={len(hist)}]",
        f"[NEDIT={n_edit}]", f"[NREAD={n_read}]", f"[NSEARCH={n_search}]",
        f"[NTEST={n_test}]", f"[STEP={step}]",
    ])

def render_sample(sample, max_history_events=12):
    meta = sample.get("session_meta") or {}
    ws   = meta.get("workspace") or {}
    hist = (sample.get("history") or [])[-max_history_events:]
    open_files = ws.get("open_files") or []
    lm = ws.get("language_mix") or {}
    main_lang = max(lm.items(), key=lambda x: x[1])[0] if lm else ""
    _drop = set(globals().get("DROP_META_FIELDS", []) or [])
    _meta_items = [
        ("tier",   f"tier={_safe_text(meta.get('user_tier'))}"),
        ("pref",   f"pref={_safe_text(meta.get('language_pref'))}"),
        ("turn",   f"turn={_turn_tok(meta.get('turn_index'))}"),
        ("budget", f"budget={_budget_bucket(meta.get('budget_tokens_remaining'))}"),
        ("elapsed",f"elapsed={_elapsed_bucket(meta.get('elapsed_session_sec'))}"),
        ("lang",   f"lang={main_lang}"),
        ("ci",     f"ci={_safe_text(ws.get('last_ci_status'))}"),
        ("git",    f"git={'dirty' if ws.get('git_dirty') else 'clean'}"),
        ("open",   f"open={len(open_files)}"),
        ("loc",    f"loc={_safe_text(ws.get('loc'))}"),
    ]
    meta_text = " ".join(tok for key, tok in _meta_items if key not in _drop)
    _ofm = globals().get("OPENFILES", "off")
    if _ofm and _ofm != "off":
        # 파일명/확장자를 '텍스트'로 추가 -> 사전학습 인코더가 의미로 읽음 (원핫 아님).
        _names = [f.split("/")[-1] for f in open_files[:6]]
        _exts  = sorted({n.rsplit(".", 1)[-1] for n in _names if "." in n})
        _add = []
        if _ofm in ("ext", "both"):   _add.append(f"openext={','.join(_exts)}")
        if _ofm in ("names", "both"): _add.append(f"openfiles={' '.join(_names)}")
        if _add: meta_text = meta_text + " " + " ".join(_add)
    if globals().get("OPENDIR", False):
        # 디렉토리 경로를 텍스트로 (예: tests/ -> run_tests, src/ -> edit 힌트). root 파일은 스킵.
        _dirs = sorted({"/".join(f.split("/")[:-1]) for f in open_files[:6] if "/" in f})
        if _dirs: meta_text = meta_text + " opendir=" + " ".join(_dirs)
    if globals().get("HISTFILES", "off") not in ("off", False, None, ""):
        # history args(read/edit/write path, grep scope 등)에서 파일 basename 추출 -> histfiles= 로 명시.
        #   openfiles(현재 열림)의 확장: 과거에 건드린(닫힌) 파일까지 포함. 확장자 있는 토큰만, 중복 제거.
        _hn = []
        for _it in (sample.get("history") or []):
            if _it.get("role") == "assistant_action":
                for _v in (_it.get("args") or {}).values():
                    if isinstance(_v, str):
                        for _tok in re.findall(r"[\w./-]*[\w-]\.[a-zA-Z][a-zA-Z0-9]{0,4}", _v):
                            _hn.append(_tok.split("/")[-1])
        _seen = set(); _uniq = []
        for _f in _hn:
            if _f not in _seen: _seen.add(_f); _uniq.append(_f)
        if _uniq: meta_text = meta_text + " histfiles=" + " ".join(_uniq[:12])
    _histpath = globals().get("HISTPATH", "full")
    _PATH_KEYS = {"path", "file", "scope", "target"}          # base 모드: 넓게
    # files 모드: (action, key)가 '파일'을 가리키는 것만. list_directory.path·grep scope는 제외(디렉토리가 의미).
    _FILE_AK = {("read_file","path"), ("edit_file","path"), ("write_file","path"),
                ("run_tests","target"), ("lint_or_typecheck","target")}
    def _basename(v): return v.rstrip("/").split("/")[-1]
    def _strip_args(name, a):
        if not a or _histpath == "full": return a
        if _histpath == "base":
            return {k: (_basename(v) if (k in _PATH_KEYS and isinstance(v, str) and "/" in v) else v) for k, v in a.items()}
        if _histpath == "files":
            return {k: (_basename(v) if ((name, k) in _FILE_AK and isinstance(v, str) and "/" in v) else v) for k, v in a.items()}
        return a
    parts = []
    for it in hist:
        if it.get("role") == "user":
            parts.append(f"U: {_safe_text(it.get('content'))}")
        elif it.get("role") == "assistant_action":
            _nm = _safe_text(it.get('name'))
            parts.append(f"A[{_nm}] {_compact_json(_strip_args(_nm, it.get('args') or {}))} -> {_safe_text(it.get('result_summary'))}")
    _head = [derived_fields(sample)] if globals().get("ADD_DERIVED", False) else []
    return " ".join(_head + ["[META]", meta_text, "[HIST]", " | ".join(parts), "[CUR]", _safe_text(sample.get("current_prompt"))])
''')

# ---- data load + session (identical) ----
code('''# ===== 데이터 로드 + 세션 추출 =====
def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

samples  = load_jsonl(DATA_DIR / "train.jsonl")
label_of = {r["id"]: r["action"] for r in csv.DictReader(open(DATA_DIR/"train_labels.csv", encoding="utf-8"))}

ids      = [s["id"] for s in samples]
texts    = [render_sample(s, MAX_HISTORY_EVENTS) for s in samples]
y        = np.array([label2id[label_of[i]] for i in ids], dtype=np.int64)
sessions = np.array([re.sub(r"-step_\\d+$", "", i) for i in ids])
N = len(ids)

if DEV_MODE:
    uniq = list(dict.fromkeys(sessions.tolist()))
    keep = set(uniq[: max(2, DEV_SAMPLES // 7)])
    idx  = [i for i in range(N) if sessions[i] in keep][:DEV_SAMPLES]
    samples=[samples[i] for i in idx]; ids=[ids[i] for i in idx]; texts=[texts[i] for i in idx]
    y=y[idx]; sessions=sessions[idx]; N=len(ids)

print("N:", N, "| 세션:", len(set(sessions)), "| 세션당 평균 step:", round(N/len(set(sessions)),2))
import collections
for c in ACTION_CLASSES:
    print(f"  {c:18s} {collections.Counter(y).get(label2id[c],0):6d}")
''')

# ---- split (identical) ----
code('''# ===== 세션 그룹 5-fold (누수 검사) =====
from sklearn.model_selection import StratifiedGroupKFold
sgkf  = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = list(sgkf.split(np.zeros(N), y, groups=sessions))
for k,(tr,va) in enumerate(folds):
    leak = len(set(sessions[va]) & set(sessions[tr]))
    print(f"fold{k}: train {len(tr):6d}  val {len(va):6d}  | 세션 누수 겹침: {leak}")
    assert leak == 0, "세션 누수 발생!"
print("\\u2705 모든 fold 세션 누수 0 — 리키지 프리")
''')

# ---- tokenizer + dataset ----
code('''# ===== 토크나이저 + Dataset =====
from transformers import AutoTokenizer, DataCollatorWithPadding
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=LOCAL_ONLY)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token   # Qwen3 등 decoder 토크나이저 pad 없음

class ActionDataset(torch.utils.data.Dataset):
    def __init__(self, idxs): self.idxs = list(idxs)
    def __len__(self): return len(self.idxs)
    def __getitem__(self, j):
        i = self.idxs[j]
        enc = tokenizer(texts[i], truncation=True, max_length=MAX_LENGTH)
        enc["labels"] = int(y[i])
        return enc

collator = DataCollatorWithPadding(tokenizer=tokenizer)
print("tokenizer ready:", tokenizer.__class__.__name__)
''')

# ---- metrics / new_model / tune_bias ----
code('''# ===== 지표 / 모델 생성 / logit_bias 튜닝 =====
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return {"macro_f1": f1_score(labels, logits.argmax(-1), average="macro")}

def new_model():
    _lora = globals().get("LORA", False)
    _is_mbert = ("granite" in str(BASE_MODEL).lower() or "modernbert" in str(BASE_MODEL).lower())
    _kw = dict(num_labels=NUM_CLASSES, id2label=id2label, label2id=label2id, local_files_only=LOCAL_ONLY)
    if _is_mbert:  # reference_compile 은 ModernBERT 전용 (DataParallel/PEFT면 끔)
        _kw["reference_compile"] = (torch.cuda.device_count() <= 1) and not _lora
    m = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, **_kw)
    if getattr(m.config, "pad_token_id", None) is None:
        m.config.pad_token_id = tokenizer.pad_token_id   # decoder 모델(Qwen3) 분류 헤드용
    m = m.float()
    if _lora:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
                         target_modules="all-linear", task_type="SEQ_CLS",
                         modules_to_save=["classifier"])
        m = get_peft_model(m, cfg)
        m.print_trainable_parameters()
    return m

def tune_bias(logits, labels, n_iter=3, grid=None):
    if grid is None: grid = np.round(np.arange(-2.5, 2.51, 0.05), 2)
    bias = np.zeros(NUM_CLASSES, dtype=np.float32)
    for _ in range(n_iter):
        for c in range(NUM_CLASSES):
            bc = bias.copy(); best_g, best_s = 0.0, -1
            for g in grid:
                bc[c] = g
                s = f1_score(labels, (logits + bc).argmax(1), average="macro")
                if s > best_s: best_s, best_g = s, g
            bias[c] = best_g
    return bias, f1_score(labels, (logits + bias).argmax(1), average="macro")
''')

# ---- TRAIN ONE FOLD ----
code('''# ===== 단일 fold 학습 + val_logits/모델 저장 =====
if FOLD is not None:
    import torch.nn as nn
    from transformers import TrainingArguments, Trainer
    class WeightedTrainer(Trainer):
        # sqrt 역빈도 class weight CE (train_derived_cw.py 와 동일)
        def __init__(self, *a, class_weights=None, **kw):
            super().__init__(*a, **kw); self.class_weights = class_weights
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels"); out = model(**inputs); logits = out.logits
            w = None if self.class_weights is None else self.class_weights.to(logits.device)
            loss = nn.CrossEntropyLoss(weight=w)(logits, labels)
            return (loss, out) if return_outputs else loss
    _is_all = (str(FOLD).lower() == "all")
    if _is_all:
        k = "all"; tr = list(range(N)); va = None   # train-on-all: 전체 학습, 홀드아웃 없음
    else:
        k = int(FOLD); tr, va = folds[k]
    _has_val = va is not None
    _ngpu = torch.cuda.device_count()
    print(f"===== TRAIN FOLD {k}  (train {len(tr)} / val {0 if va is None else len(va)}) =====")
    print(f"n_gpu={_ngpu} | per_device TRAIN_BS={TRAIN_BS} | global batch={TRAIN_BS*max(1,_ngpu)} "
          f"| {'DataParallel' if _ngpu>1 else 'single-GPU'} | DROP_META_FIELDS={DROP_META_FIELDS} "
          f"| ADD_DERIVED={ADD_DERIVED} | CLASS_WEIGHT={CLASS_WEIGHT} | OPENFILES={OPENFILES} "
          f"| TURN_REPR={TURN_REPR} | LORA={LORA}")
    class_weights = None
    if CLASS_WEIGHT:
        counts = np.bincount(y[tr], minlength=NUM_CLASSES).astype(np.float64)
        cw = len(tr) / (NUM_CLASSES * np.clip(counts, 1.0, None))
        cw = np.sqrt(cw); cw = cw / cw.mean()
        class_weights = torch.tensor(cw, dtype=torch.float32)
        print(f"class_weight(sqrt) min={cw.min():.2f} max={cw.max():.2f} "
              f"(max={ACTION_CLASSES[int(cw.argmax())]}, min={ACTION_CLASSES[int(cw.argmin())]})")
    model = new_model()
    args = TrainingArguments(
        output_dir=str(RESULTS_DIR / "trainer_tmp" / f"fold{k}"),
        num_train_epochs=1 if DEV_MODE else EPOCHS,
        learning_rate=LR,
        per_device_train_batch_size=TRAIN_BS, per_device_eval_batch_size=EVAL_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        weight_decay=WEIGHT_DECAY, warmup_ratio=WARMUP_RATIO,
        eval_strategy=("epoch" if _has_val else "no"), save_strategy=("epoch" if _has_val else "no"),
        load_best_model_at_end=_has_val, metric_for_best_model="macro_f1", greater_is_better=True,
        save_total_limit=1, logging_steps=50, report_to="none",
        fp16=(device.type == "cuda"), seed=TRAIN_SEED,
    )
    _extra = {"class_weights": class_weights} if CLASS_WEIGHT else {}
    _TrainerCls = WeightedTrainer if CLASS_WEIGHT else Trainer
    _cbs = []
    if _has_val and EARLY_STOP and int(EARLY_STOP) > 0:
        from transformers import EarlyStoppingCallback
        _cbs = [EarlyStoppingCallback(early_stopping_patience=int(EARLY_STOP))]
    trainer = _TrainerCls(model=model, args=args,
                      train_dataset=ActionDataset(tr),
                      eval_dataset=(ActionDataset(va) if _has_val else None),
                      data_collator=collator, compute_metrics=compute_metrics, callbacks=_cbs, **_extra)
    trainer.train()

    if _has_val:
        val_logits = trainer.predict(ActionDataset(va)).predictions
        fold_f1 = f1_score(y[va], val_logits.argmax(1), average="macro")
        print(f"fold{k} val macro-F1: {fold_f1:.4f}")
    else:
        val_logits = None; fold_f1 = None
        print(f"fold{k}: train-on-all (홀드아웃 없음, val 없음)")

    fd = MODEL_ROOT / f"{MODEL_PREFIX}-fold{k}"
    if globals().get("LORA", False):
        merged = trainer.model.merge_and_unload()   # LoRA 어댑터를 base에 병합 -> 표준 모델로 저장
        merged.save_pretrained(str(fd))
    else:
        trainer.save_model(str(fd))
    tokenizer.save_pretrained(str(fd))
    json.dump({"fold":k, "n_splits":N_SPLITS, "base_model":BASE_MODEL, "max_length":MAX_LENGTH,
               "max_history_events":MAX_HISTORY_EVENTS,
               "best_val_macro_f1":(None if fold_f1 is None else float(fold_f1)),
               "action_classes":ACTION_CLASSES, "split":"StratifiedGroupKFold(session)", "seed":SEED, "train_seed":TRAIN_SEED},
              open(fd/"training_meta.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)

    if _has_val:
        np.savez_compressed(OOF_DIR / f"fold{k}_val.npz",
                            va=np.asarray(va, dtype=np.int64),
                            logits=val_logits.astype("float32"),
                            y=y[va], fold=k, f1=float(fold_f1))
    print("saved model:", fd)
    del model, trainer
    if device.type == "cuda": torch.cuda.empty_cache()
else:
    print("FOLD is None -> 학습 건너뜀")
''')

# ---- FINALIZE: assemble OOF + bias ----
code('''# ===== [FINALIZE] 저장된 fold들 -> OOF 조립 + logit_bias 튜닝 =====
if RUN_FINALIZE:
    oof_logits = np.zeros((N, NUM_CLASSES), dtype=np.float32)
    covered    = np.zeros(N, dtype=bool)
    per_fold_f1 = {}
    loaded = []
    _fold_list = FOLDS_TO_FINALIZE if FOLDS_TO_FINALIZE else list(range(N_SPLITS))
    for k in _fold_list:
        fp = OOF_DIR / f"fold{k}_val.npz"
        if fp.exists():
            d = np.load(fp)
            va = d["va"]; oof_logits[va] = d["logits"]; covered[va] = True
            per_fold_f1[k] = float(d["f1"]); loaded.append(k)
    print("loaded folds:", loaded, "| covered:", int(covered.sum()), "/", N)
    assert loaded, "fold_val npz 가 없음 — 학습이 먼저 돌았는지 확인"

    np.savez_compressed(OOF_DIR / "oof_logits.npz",
                        logits=oof_logits, ids=np.array(ids), y=y, covered=covered)

    cov  = covered
    y_c  = y[cov]
    oof_c = oof_logits[cov]
    base_f1 = f1_score(y_c, oof_c.argmax(1), average="macro")
    bias, tuned_f1 = tune_bias(oof_c, y_c)
    print(f"OOF Macro-F1  base={base_f1:.4f}  tuned={tuned_f1:.4f}  (커버 {cov.sum()})")

    json.dump({"n_splits":N_SPLITS, "folds_used":loaded,
               "base_macro_f1":float(base_f1), "tuned_macro_f1":float(tuned_f1),
               "per_fold_val_macro_f1":per_fold_f1, "action_classes":ACTION_CLASSES,
               "bias":{ACTION_CLASSES[i]: float(bias[i]) for i in range(NUM_CLASSES)}},
              open(OOF_DIR / "logit_bias.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)
    y_pred = (oof_c + bias).argmax(1)
    print("saved:", OOF_DIR / "oof_logits.npz", "|", OOF_DIR / "logit_bias.json")
else:
    print("RUN_FINALIZE=False -> OOF/분석/앱상블 건너뜀")
''')

# ---- analysis 1: classwise F1 ----
md("## 분석 ① 클래스별 F1 + 혼동행렬 (OOF 기반)")
code('''if RUN_FINALIZE:
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    p,r,f1c,sup = precision_recall_fscore_support(y_c, y_pred, labels=range(NUM_CLASSES), zero_division=0)
    lines = [f"{'class':18s} {'prec':>6s} {'rec':>6s} {'f1':>6s} {'support':>8s}", "-"*50]
    for i in np.argsort(f1c):
        flag = "  <-- 약함" if f1c[i] < f1c.mean() else ""
        lines.append(f"{ACTION_CLASSES[i]:18s} {p[i]:6.3f} {r[i]:6.3f} {f1c[i]:6.3f} {sup[i]:8d}{flag}")
    lines += ["-"*50, f"{'MACRO':18s} {p.mean():6.3f} {r.mean():6.3f} {f1c.mean():6.3f} {sup.sum():8d}"]
    report = "\\n".join(lines); print(report)
    (ANALYSIS_DIR / "classwise_f1.txt").write_text(report, encoding="utf-8")
''')

# ---- analysis 1b: confusion matrix + savefig ----
code('''if RUN_FINALIZE:
    cm = confusion_matrix(y_c, y_pred, labels=range(NUM_CLASSES))
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    pairs = sorted([(cm[i,j], cmn[i,j], ACTION_CLASSES[i], ACTION_CLASSES[j])
                    for i in range(NUM_CLASSES) for j in range(NUM_CLASSES) if i!=j and cm[i,j]>0], reverse=True)
    plines = ["가장 많이 헷갈리는 쌍 (정답 -> 예측 : 건수, 행비율)"]
    for cnt,frac,ti,tj in pairs[:15]:
        plines.append(f"  {ti:16s} -> {tj:16s} : {cnt:4d} ({frac:5.1%})")
    ptxt = "\\n".join(plines); print(ptxt)
    (ANALYSIS_DIR / "confusion_pairs.txt").write_text(ptxt, encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(figsize=(9,8)); im=ax.imshow(cmn,cmap="Blues",vmin=0,vmax=1)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(ACTION_CLASSES,rotation=90,fontsize=8); ax.set_yticklabels(ACTION_CLASSES,fontsize=8)
    ax.set_xlabel("pred"); ax.set_ylabel("true"); ax.set_title("OOF confusion (row-normalized)")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if cmn[i,j]>0.03: ax.text(j,i,f"{cmn[i,j]:.2f}",ha="center",va="center",fontsize=7,
                                      color="white" if cmn[i,j]>0.5 else "black")
    fig.colorbar(im); plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "confusion_matrix.png", dpi=130)
    print("saved:", ANALYSIS_DIR / "confusion_matrix.png")
    plt.show()
''')

# ---- analysis 2: lang / length ----
md("## 분석 ② 언어별 / 길이별 성능")
code('''if RUN_FINALIZE:
    langs = np.array([ (samples[i].get("session_meta") or {}).get("language_pref","?") for i in range(N) ])[cov]
    al = ["언어별 Macro-F1"]
    for lg in ["ko","en","mixed"]:
        m = (langs==lg)
        if m.sum(): al.append(f"  {lg:6s} n={m.sum():6d}  f1={f1_score(y_c[m], y_pred[m], average='macro'):.4f}")
    tok_lens = np.array([len(tokenizer(texts[i], truncation=False)['input_ids']) for i in np.where(cov)[0]])
    correct = (y_c==y_pred)
    al.append(f"\\n512 초과(잘림) 비율: {(tok_lens>MAX_LENGTH).mean():.1%}")
    al.append(f"{'구간':>12s} {'n':>6s} {'acc':>7s}")
    for lo,hi in [(0,256),(256,512),(512,1024),(1024,10**9)]:
        m=(tok_lens>=lo)&(tok_lens<hi)
        if m.sum(): al.append(f"{f'{lo}-{hi}':>12s} {m.sum():6d} {correct[m].mean():7.1%}")
    atxt="\\n".join(al); print(atxt)
    (ANALYSIS_DIR / "lang_length.txt").write_text(atxt, encoding="utf-8")
''')

# ---- analysis 3: misclassification + label noise ----
md("## 분석 ③ 오분류 정독 + 라벨 노이즈")
code('''if RUN_FINALIZE:
    cov_idx = np.where(cov)[0]
    mlines=[]
    def show_mis(cls, n=6):
        ci=label2id[cls]
        rows=[k for k in range(len(cov_idx)) if y_c[k]==ci and y_pred[k]!=ci]
        mlines.append(f"[{cls}] 오분류 {len(rows)}건 중 {min(n,len(rows))}")
        for k in rows[:n]:
            s=samples[cov_idx[k]]
            mlines.append(f"  정답={cls} 예측={id2label[y_pred[k]]} | CUR: {(s.get('current_prompt') or '')[:90]}")
    for c in ["list_directory","read_file","grep_search"]:
        show_mis(c); mlines.append("")
    from collections import defaultdict
    k2l=defaultdict(set)
    for i in range(N): k2l[texts[i]].add(int(y[i]))
    conf=[k for k,v in k2l.items() if len(v)>1]
    mlines.append(f"동일 맥락-다른라벨 그룹: {len(conf)}개 (Macro-F1 상한 제약)")
    mtxt="\\n".join(mlines); print(mtxt)
    (ANALYSIS_DIR / "misclassified.txt").write_text(mtxt, encoding="utf-8")
''')

# ---- ensemble -> submission ----
md("## 5-fold 앙상블 → test 예측 → submission")
code('''if RUN_FINALIZE:
    test_samples = load_jsonl(DATA_DIR/"test.jsonl")
    test_ids   = [s["id"] for s in test_samples]
    test_texts = [render_sample(s, MAX_HISTORY_EVENTS) for s in test_samples]

    def infer(model_dir, texts_):
        m = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True).to(device).eval()
        outs=[]
        with torch.no_grad():
            for i in range(0, len(texts_), EVAL_BS):
                b = tokenizer(texts_[i:i+EVAL_BS], truncation=True, max_length=MAX_LENGTH,
                              padding=True, return_tensors="pt").to(device)
                outs.append(m(**b).logits.float().cpu().numpy())
        del m
        if device.type=="cuda": torch.cuda.empty_cache()
        return np.concatenate(outs,0)

    use_dirs = sorted(MODEL_ROOT.glob(f"{MODEL_PREFIX}-fold*"))
    print("ensemble dirs:", [d.name for d in use_dirs])
    ens = np.mean([infer(d, test_texts) for d in use_dirs], axis=0)
    pred_ids = (ens + bias).argmax(1)
    pred_map = {tid: ACTION_CLASSES[p] for tid,p in zip(test_ids, pred_ids)}

    sub = list(csv.DictReader(open(DATA_DIR/"sample_submission.csv", encoding="utf-8")))
    fields = list(sub[0].keys())
    fallback = ACTION_CLASSES[int(np.bincount(y).argmax())]
    for row in sub:
        row["action"] = pred_map.get(row["id"], fallback)
    out_csv = SUB_DIR / "submission.csv"
    with open(out_csv,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(sub)
    print(f"submission 저장: {out_csv}  (앵상블 {len(use_dirs)} folds)")

    metrics = {"oof_base_macro_f1":float(base_f1), "oof_tuned_macro_f1":float(tuned_f1),
               "folds_used":loaded, "per_fold_val_macro_f1":per_fold_f1,
               "n_ensemble":len(use_dirs), "n_test":len(test_ids), "covered":int(cov.sum())}
    json.dump(metrics, open(RESULTS_DIR / "metrics.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
    print("metrics:", metrics)
''')

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name":"python3","display_name":"Python 3","language":"python"}
out = sys.argv[1] if len(sys.argv)>1 else "train_and_analyze_4gpu.ipynb"
nbf.write(nb, out)
print("wrote", out, "cells=", len(cells))
