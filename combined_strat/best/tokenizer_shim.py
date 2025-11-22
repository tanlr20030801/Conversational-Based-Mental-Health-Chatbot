# tokenizer_shim.py
import re
try:
    import jieba
    HAS_JIEBA = True
except Exception:
    HAS_JIEBA = False

EN_STOP = set()
CN_STOP = set()

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
TOKEN_RE_EN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
FALLBACK_WORD_RE = re.compile(r"\w+")
NUM_PUNC_RE = re.compile(r"^[\W_]+$")

def mixed_tokenize(text):
    text = str(text).strip()
    toks = []
    en = [w.lower() for w in TOKEN_RE_EN.findall(text)]
    en = [w for w in en if w not in EN_STOP]
    toks.extend(en)
    if CJK_RE.search(text):
        if HAS_JIEBA:
            cn = [w.strip() for w in jieba.cut(text, cut_all=False) if w.strip()]
        else:
            cn = [ch for ch in text if CJK_RE.match(ch)]
        cn = [w for w in cn if w not in CN_STOP and not NUM_PUNC_RE.match(w)]
        toks.extend(cn)
    if not toks:
        toks = [t for t in FALLBACK_WORD_RE.findall(text.lower()) if t not in EN_STOP]
    return toks
