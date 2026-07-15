"""E40 — language-aware character-level noise for free-text fields (KO jamo + EN ASCII).

Self-contained in the E40 folder; imported by run_e40.py (training) and the demo. Operates on
raw sample dicts, noising ONLY `current_prompt` + `USER:` message content — structure, action
names, args, verdicts, metadata, and labels are never touched.

Korean is handled at the JAMO (자모) level — decompose each Hangul syllable via the standard
formula (0xAC00 + 초성×588 + 중성×28 + 종성; 19×21×28, dependency-free), perturb ONE component,
recompose → a realistic single-keystroke typo (word stays recognizable). English/ASCII letters get
the classic char ops (swap/delete/insert/QWERTY-substitute). Digits, punctuation, code symbols,
and whitespace are left alone.
"""
import numpy as np

# ---- Hangul jamo tables ----
CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"                 # 19 initials
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"           # 21 medials
JONG = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")  # 28 finals (idx0 = none)
_CHO_I = {c: i for i, c in enumerate(CHO)}
_JUNG_I = {c: i for i, c in enumerate(JUNG)}

# realistic confusion tables (dubeolsik-adjacent / phonetic neighbors)
_CHO_SIM = {"ㄱ": "ㄲㅋ", "ㄷ": "ㄸㅌ", "ㅂ": "ㅃㅍ", "ㅅ": "ㅆ", "ㅈ": "ㅉㅊ",
            "ㅁ": "ㄴ", "ㄴ": "ㅁㄹ", "ㄹ": "ㄴ", "ㅇ": "ㅎ", "ㅎ": "ㅇ"}
_JUNG_SIM = {"ㅏ": "ㅓㅑ", "ㅓ": "ㅏㅕ", "ㅗ": "ㅜㅛ", "ㅜ": "ㅗㅠ", "ㅐ": "ㅔㅒ",
             "ㅔ": "ㅐㅖ", "ㅡ": "ㅜㅣ", "ㅣ": "ㅡ", "ㅑ": "ㅏ", "ㅕ": "ㅓ"}

# compact QWERTY neighbours for ASCII substitution
_QWERTY = {"q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh",
           "u": "yij", "i": "uok", "o": "ipl", "p": "ol", "a": "qsz", "s": "awdz",
           "d": "sefc", "f": "drgv", "g": "fthb", "h": "gyjn", "j": "hukm", "k": "jil",
           "l": "kop", "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn",
           "n": "bhjm", "m": "njk"}


def _is_hangul(c):
    return "가" <= c <= "힣"


def _perturb_hangul(c, rng):
    """Change ONE jamo of a composed syllable (initial / medial / final)."""
    s = ord(c) - 0xAC00
    ci, ji, ki = s // 588, (s % 588) // 28, s % 28
    r = rng.integers(0, 3)
    if r == 0:                                   # initial consonant
        alt = _CHO_SIM.get(CHO[ci])
        if alt:
            ci = _CHO_I[alt[rng.integers(0, len(alt))]]
    elif r == 1:                                 # medial vowel
        alt = _JUNG_SIM.get(JUNG[ji])
        if alt:
            ji = _JUNG_I[alt[rng.integers(0, len(alt))]]
    else:                                        # final consonant (batchim): drop/add/alter
        ki = 0 if ki else int(rng.integers(1, 28))
    return chr(0xAC00 + ci * 588 + ji * 28 + ki)


def noise_text(s, rng, p):
    """Char-level noise on ONE free-text string. KO→jamo perturb; ASCII letter→char op;
    everything else untouched. `rng` is a np.random.Generator (fresh draw = fresh noise)."""
    if not s or p <= 0:
        return s
    out = []
    for c in s:
        if _is_hangul(c):
            out.append(_perturb_hangul(c, rng) if rng.random() < p else c)
        elif c.isascii() and c.isalpha() and rng.random() < p:
            op = rng.integers(0, 4)
            if op == 0:                          # delete
                continue
            elif op == 1:                        # insert a random ascii letter after
                out.append(c)
                out.append(chr(rng.integers(97, 123)))
            elif op == 2 and out:                # swap with previous emitted char
                prev = out.pop()
                out.append(c)
                out.append(prev)
            else:                                # QWERTY-adjacent substitute (case-preserving)
                nb = _QWERTY.get(c.lower())
                sub = nb[rng.integers(0, len(nb))] if nb else c.lower()
                out.append(sub.upper() if c.isupper() else sub)
        else:
            out.append(c)                        # digits / punct / symbols / space
    return "".join(out)


def noise_sample_freetext(sample, rng, p):
    """Return a copy of `sample` with ONLY current_prompt + USER-event content char-noised.
    No deepcopy: reuses non-user events verbatim; copies only what it mutates."""
    s2 = dict(sample)
    s2["current_prompt"] = noise_text(sample.get("current_prompt") or "", rng, p)
    new_hist = []
    for t in sample.get("history", []):
        if t.get("role") == "user" and t.get("content"):
            t2 = dict(t)
            t2["content"] = noise_text(t["content"], rng, p)
            new_hist.append(t2)
        else:
            new_hist.append(t)
    s2["history"] = new_hist
    return s2
