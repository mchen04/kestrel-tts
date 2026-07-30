"""FastG2P — fast English G2P reproducing misaki.en.G2P(trf=False, british=False, fallback=None).

Strategy:
  - misaki's cost is spaCy tok2vec+tagger (~150 ms / 15k chars). Its tokenizer is
    rule-based; we replicate it exactly with a whitespace-chunk cache (spaCy
    tokenization is context-free per whitespace-delimited chunk; verified).
  - Tags only matter for a small set of tag-sensitive words (POS-keyed lexicon
    dicts, misaki special cases, NNP decisions for cap/upper OOV words, punct,
    CD). We ship a compact tag model (unigram + contextual bigram overrides)
    trained to match en_core_web_sm on ~19M chars of book text.
  - All downstream logic (retokenize, lexicon, stress, numbers, currency,
    context) is inherited unchanged from misaki.en.G2P, so phoneme logic is
    bit-identical whenever tokens+tags match.
  - A memoized fast path caches per-chunk phoneme contributions so chapter-scale
    calls avoid the Python token machinery for already-seen simple chunks.
"""
import gzip
import json
import re
from dataclasses import replace
from pathlib import Path

from misaki import en as men
from misaki.en import TokenContext
from misaki.token import MToken

DATA_DIR = Path(__file__).parent / "data"

_WS_GAP = re.compile(r"\s+")
_DIGIT_RE = re.compile(r"\d")
_ORDINAL_RE = re.compile(r"\d+(?:st|nd|rd|th)$")
_CURRENCY_CHARS = frozenset("$£€")
_INHERIT = object()  # future_vowel passes through from the right

# words whose phonemes depend on TokenContext regardless of what the corpus scan found
_CTX_SAFETY = {
    "a", "A", "am", "Am", "AM", "an", "An", "AN", "I", "by", "By", "BY",
    "to", "To", "TO", "in", "In", "IN", "the", "The", "THE",
    "used", "Used", "USED", "vs", "vs.", "Vs", "Vs.", "versus",
}


class _LTok:
    """Duck-typed lightweight MToken for cached chunks."""
    __slots__ = ("text", "tag", "whitespace", "phonemes", "start_ts", "end_ts", "_")

    def __init__(self, text, tag, whitespace, phonemes):
        self.text = text
        self.tag = tag
        self.whitespace = whitespace
        self.phonemes = phonemes
        self.start_ts = None
        self.end_ts = None
        self._ = None

# tags spaCy sm assigns (near-)deterministically by surface form
PUNCT_TAG_BY_TEXT = {
    ",": ",", ".": ".", "!": ".", "?": ".", "...": ":", "…": ":",
    ";": ":", ":": ":", "-": ":", "–": ":", "—": ":",
    "(": "-LRB-", ")": "-RRB-", "[": "-LRB-", "]": "-RRB-",
    "{": "-LRB-", "}": "-RRB-",
    '"': "''", "”": "''", "“": "``", "``": "``", "''": "''",
    "$": "$", "£": "$", "€": "$", "#": "$", "%": "NN",
    "&": "CC", "+": "SYM", "=": "SYM", "/": "SYM", "*": "SYM",
    "‘": "``", "’": "''",
}


class _Tagger:
    """Match en_core_web_sm tag_ closely on the tokens where it matters."""

    def __init__(self):
        with gzip.open(DATA_DIR / "fastg2p_tags.json.gz", "rt") as f:
            d = json.load(f)
        self.uni = d["unigram"]   # token text -> most frequent tag
        self.tri = d["tri"]       # "prev\tword\tnext" overrides
        self.bnext = d["bnext"]   # "word\tnext" overrides
        self.bprev = d["bprev"]   # "prev\tword" overrides
        self.patch = d.get("patch", {})  # "p2\tp1\tw\tn1\tn2" exact-context overrides

    def tag(self, texts, ws):
        uni, tri, bnext, bprev = self.uni, self.tri, self.bnext, self.bprev
        patch = self.patch
        tags = []
        n = len(texts)
        for i, w in enumerate(texts):
            if w.isspace():
                tags.append("_SP")
                continue
            if patch:
                k5 = "\t".join([
                    texts[i - 2] if i > 1 else "<S>", texts[i - 1] if i > 0 else "<S>", w,
                    texts[i + 1] if i + 1 < n else "</S>", texts[i + 2] if i + 2 < n else "</S>"])
                t = patch.get(k5)
                if t is not None:
                    tags.append(t)
                    continue
            t = None
            if w in uni:
                p = texts[i - 1] if i > 0 else "<S>"
                nx = texts[i + 1] if i + 1 < n else "</S>"
                t = tri.get(p + "\t" + w + "\t" + nx)
                if t is None:
                    t = bnext.get(w + "\t" + nx)
                if t is None:
                    t = bprev.get(p + "\t" + w)
                if t is None and w not in ('"', "'"):
                    t = uni[w]
                if t is not None:
                    tags.append(t)
                    continue
            if w == '"' or w == "'":
                # positional fallback for straight quotes unseen in context tables
                if w == "'" and i > 0 and ws[i - 1] == "" and texts[i - 1][-1:].lower() == "s":
                    tags.append("POS")
                    continue
                opening = i == 0 or ws[i - 1] == " " or texts[i - 1].isspace() or texts[i - 1] in "([{—–-“‘"
                tags.append("``" if opening else "''")
                continue
            if w in ("-", "–", "—"):
                if w == "-" and (i > 0 and ws[i - 1] == "" and not texts[i - 1].isspace()
                                 and i + 1 < n and ws[i] == "" and not texts[i + 1].isspace()):
                    t = "HYPH"  # intra-word hyphen
                else:
                    t = ":"
            if t is None:
                t = PUNCT_TAG_BY_TEXT.get(w)
            if t is None:
                if _DIGIT_RE.search(w):
                    t = "JJ" if _ORDINAL_RE.match(w) else "CD"
                elif w[:1].isupper():
                    t = "NNP"
                elif not any(c.isalpha() for c in w):
                    t = "NFP"
                else:
                    t = "NN"
            tags.append(t)
        return tags


class FastG2P(men.G2P):
    def __init__(self, unk="❓", fallback=None):
        # NOTE: deliberately do NOT call super().__init__ (it loads spaCy models)
        self.version = None
        self.british = False
        self.lexicon = men.Lexicon(False)
        self.fallback = fallback if fallback else None
        self.unk = unk
        self.nlp = None  # lazy blank tokenizer, only for cache misses
        self.tagger = _Tagger()
        self._chunk_cache = {}
        try:
            with gzip.open(DATA_DIR / "fastg2p_chunks.json.gz", "rt") as f:
                self._chunk_cache = {k: tuple(v) for k, v in json.load(f).items()}
        except FileNotFoundError:
            pass
        self._memo = {}        # chunk fill memo, see __call__
        self._info_cache = {}  # chunk -> [tokens, hard_slow, needs_nbr, needs_ctx]
        d = self.tagger
        self._patch_words = {k.split("\t")[2] for k in d.patch}
        with gzip.open(DATA_DIR / "fastg2p_tags.json.gz", "rt") as f:
            self._ctxsens = set(json.load(f).get("ctx_sensitive", ())) | _CTX_SAFETY

    # ---- tokenization ----

    def _split_chunk(self, chunk):
        toks = self._chunk_cache.get(chunk)
        if toks is None:
            if len(chunk) == 1 or chunk.isalpha():
                toks = (chunk,)
            else:
                if self.nlp is None:
                    import spacy
                    self.nlp = spacy.blank("en")
                toks = tuple(t.text for t in self.nlp.tokenizer(chunk))
            self._chunk_cache[chunk] = toks
        return toks

    def _tokenize_texts(self, text):
        """Reproduce spaCy tokenization: (texts, whitespaces)."""
        texts, ws = [], []
        pos = 0
        n = len(text)
        split = self._split_chunk
        for m in _WS_GAP.finditer(text):
            if m.start() > pos:
                toks = split(text[pos:m.start()])
                texts.extend(toks)
                ws.extend([""] * len(toks))
            elif pos == 0 and m.start() == 0:
                pass
            gap = m.group()
            if gap == " ":
                if ws:
                    ws[-1] = " "
                else:
                    texts.append(" ")
                    ws.append("")
            else:
                if gap[0] == " " and ws:
                    ws[-1] = " "
                    gap = gap[1:]
                texts.append(gap)
                ws.append("")
            pos = m.end()
        if pos < n:
            toks = split(text[pos:])
            texts.extend(toks)
            ws.extend([""] * len(toks))
        return texts, ws

    def tokenize(self, text, tokens, features):
        texts, whitespaces = self._tokenize_texts(text)
        tags = self.tagger.tag(texts, whitespaces)
        mutable_tokens = [MToken(
            text=t, tag=g, whitespace=w,
            _=MToken.Underscore(is_head=True, num_flags="", prespace=False),
        ) for t, g, w in zip(texts, tags, whitespaces)]
        if not features:
            return mutable_tokens
        import numpy as np
        import spacy.training
        align = spacy.training.Alignment.from_strings(tokens, texts)
        for k, v in features.items():
            assert isinstance(v, str) or isinstance(v, int) or v in (0.5, -0.5), (k, v)
            for i, j in enumerate(np.where(align.y2x.data == k)[0]):
                if j >= len(mutable_tokens):
                    continue
                if not isinstance(v, str):
                    mutable_tokens[j]._.stress = v
                elif v.startswith("/"):
                    mutable_tokens[j]._.is_head = i == 0
                    mutable_tokens[j].phonemes = v.lstrip("/") if i == 0 else ""
                    mutable_tokens[j]._.rating = 5
                elif v.startswith("#"):
                    mutable_tokens[j]._.num_flags = v.lstrip("#")
        return mutable_tokens

    # ---- fast path ----

    def _process_tokens(self, words, ctx):
        """misaki G2P.__call__ core after tokenize (no features => fold_left is identity)."""
        words = men.G2P.retokenize(words)
        for i, w in reversed(list(enumerate(words))):
            if not isinstance(w, list):
                if w.phonemes is None:
                    w.phonemes, w.rating = self.lexicon(replace(w, _=w._), ctx)
                if w.phonemes is None and self.fallback is not None:
                    w.phonemes, w.rating = self.fallback(replace(w, _=w._))
                ctx = men.G2P.token_context(ctx, w.phonemes, w)
                continue
            left, right = 0, len(w)
            should_fallback = False
            while left < right:
                if any(tk._.alias is not None or tk.phonemes is not None for tk in w[left:right]):
                    tk = None
                else:
                    tk = men.merge_tokens(w[left:right])
                ps, rating = (None, None) if tk is None else self.lexicon(tk, ctx)
                if ps is not None:
                    w[left].phonemes = ps
                    w[left]._.rating = rating
                    for x in w[left + 1:right]:
                        x.phonemes = ''
                        x.rating = rating
                    ctx = men.G2P.token_context(ctx, ps, tk)
                    right = left
                    left = 0
                elif left + 1 < right:
                    left += 1
                else:
                    right -= 1
                    tk = w[right]
                    if tk.phonemes is None:
                        if all(c in men.SUBTOKEN_JUNKS for c in tk.text):
                            tk.phonemes = ''
                            tk._.rating = 3
                        elif self.fallback is not None:
                            should_fallback = True
                            break
                    left = 0
            if should_fallback:
                tk = men.merge_tokens(w)
                w[0].phonemes, w[0]._.rating = self.fallback(tk)
                for j in range(1, len(w)):
                    w[j].phonemes = ''
                    w[j]._.rating = w[0]._.rating
            else:
                men.G2P.resolve_tokens(w)
        tokens = [men.merge_tokens(tk, unk=self.unk) if isinstance(tk, list) else tk for tk in words]
        for tk in tokens:
            if tk.phonemes:
                tk.phonemes = tk.phonemes.replace('ɾ', 'T').replace('ʔ', 't')
        return tokens, ctx

    def _chunk_info(self, chunk):
        info = self._info_cache.get(chunk)
        if info is None:
            toks = self._split_chunk(chunk)
            hard_slow = any(
                any(c.isdigit() or c.isnumeric() for c in t) for t in toks
            ) or not _CURRENCY_CHARS.isdisjoint(chunk)
            uni = self.tagger.uni
            pw = self._patch_words
            needs_nbr = any(t in uni or t in pw or t == '"' or t == "'" for t in toks)
            needs_ctx = True if any(t in self._ctxsens for t in toks) else None  # None: unknown yet
            info = [toks, hard_slow, needs_nbr, needs_ctx]
            self._info_cache[chunk] = info
        return info

    def _mtokens_for(self, toks, tags, trailing=" "):
        n = len(toks)
        return [MToken(
            text=t, tag=g, whitespace=(trailing if j == n - 1 else ""),
            _=MToken.Underscore(is_head=True, num_flags="", prespace=False),
        ) for j, (t, g) in enumerate(zip(toks, tags))]

    def _tags_windowed(self, toks, prevs, nexts):
        ext = list(prevs) + list(toks) + list(nexts)
        ws = [" "] * len(prevs) + [""] * (len(toks) - 1) + [" "] + [""] * len(nexts)
        a = len(prevs)
        return self.tagger.tag(ext, ws)[a:a + len(toks)]

    @staticmethod
    def _parts_of(final_tokens):
        return tuple((tk.text, tk.tag, tk.phonemes, tk.whitespace) for tk in final_tokens)

    def _fill(self, info, chunk, prevs, nexts, fv, ft):
        """Compute (parts, vowel_out_or_INHERIT, future_to_out) for a chunk; manage memo flags."""
        toks = info[0]
        tags = self._tags_windowed(toks, prevs, nexts)

        def run(fv_, ft_):
            mts = self._mtokens_for(toks, tags)
            final, ctx_out = self._process_tokens(mts, TokenContext(future_vowel=fv_, future_to=ft_))
            return self._parts_of(final), ctx_out

        if info[3] is True:
            parts, ctx_out = run(fv, ft)
            return parts, ctx_out.future_vowel, ctx_out.future_to
        # unknown or ctx-free: probe three contexts once
        probes = [(None, False), (True, False), (False, True)]
        outs = [run(*p) for p in probes]
        if all(o[0] == outs[0][0] for o in outs[1:]):
            vs = tuple(o[1].future_vowel for o in outs)
            if vs[0] == vs[1] == vs[2]:
                info[3] = False
                return outs[0][0], vs[0], outs[0][1].future_to
            if vs == (None, True, False):
                info[3] = False
                return outs[0][0], _INHERIT, outs[0][1].future_to
        # genuinely context-dependent
        info[3] = True
        parts, ctx_out = run(fv, ft)
        return parts, ctx_out.future_vowel, ctx_out.future_to

    def __call__(self, text, preprocess=True):
        if preprocess is not True:
            return super().__call__(text, preprocess)
        if "[" in text and men.LINK_REGEX.search(text):
            return super().__call__(text)
        text = text.lstrip()
        if not text:
            return "", []

        # ---- split into whitespace chunks / space tokens ----
        items = []  # [kind 'c'|'s', text, trailing_ws]
        pos = 0
        for m in _WS_GAP.finditer(text):
            if m.start() > pos:
                items.append(["c", text[pos:m.start()], ""])
            gap = m.group()
            if gap == " ":
                if items:
                    items[-1][2] = " "
            else:
                if gap[0] == " " and items:
                    items[-1][2] = " "
                    gap = gap[1:]
                items.append(["s", gap, ""])
            pos = m.end()
        if pos < len(text):
            items.append(["c", text[pos:], ""])

        n = len(items)
        infos = [None] * n
        slow = [False] * n
        for i, (k, txt, _tw) in enumerate(items):
            if k == "s":
                slow[i] = True
            else:
                infos[i] = self._chunk_info(txt)
                if infos[i][1]:
                    slow[i] = True
        for i in range(n):
            if items[i][0] == "s":  # space tokens glue neighbors into one word-list
                if i > 0:
                    slow[i - 1] = True
                if i + 1 < n:
                    slow[i + 1] = True

        # neighbor token texts per item (for tag windows / memo keys), O(n) precompute
        item_toks = [infos[i][0] if items[i][0] == "c" else (items[i][1],) for i in range(n)]
        prevs_of = [()] * (n + 1)  # last two token texts before item i
        acc = ()
        for i in range(n):
            prevs_of[i] = acc
            tk = item_toks[i]
            acc = (tk[-2], tk[-1]) if len(tk) >= 2 else ((acc[-1], tk[-1]) if acc else (tk[-1],))
        prevs_of[n] = acc
        nexts_of = [()] * (n + 1)  # first two token texts from item i onward
        acc = ()
        for i in range(n - 1, -1, -1):
            tk = item_toks[i]
            acc = (tk[0], tk[1]) if len(tk) >= 2 else ((tk[0], acc[0]) if acc else (tk[0],))
            nexts_of[i] = acc

        def last2(i):
            return prevs_of[i + 1] if i >= 0 else ()

        def first2(i):
            return nexts_of[i] if i < n else ()

        # ---- assemble right to left ----
        memo = self._memo
        rev_tokens = []
        rev_ps = []
        unk = self.unk
        cfv, cft = None, False  # ctx.future_vowel, ctx.future_to
        i = n - 1
        while i >= 0:
            if not slow[i]:
                info = infos[i]
                chunk = items[i][1]
                trailing = items[i][2]
                needs_nbr = info[2]
                if needs_nbr:
                    prevs = last2(i - 1)
                    nexts = first2(i + 1)
                    keybase = (chunk, prevs, nexts)
                else:
                    prevs = nexts = ()
                    keybase = chunk
                needs_ctx = info[3]
                if needs_ctx:
                    key = (keybase, cfv, cft)
                else:
                    key = keybase
                entry = memo.get(key)
                if entry is None:
                    fv, ft = cfv, cft
                    parts, v_out, ft_out = self._fill(info, chunk, prevs, nexts, fv, ft)
                    entry = (parts, v_out, ft_out)
                    # info[3] may have been resolved inside _fill; re-derive key
                    key = (keybase, fv, ft) if info[3] else keybase
                    memo[key] = entry
                parts, v_out, ft_out = entry
                for j in range(len(parts) - 1, -1, -1):
                    t, tg, ps, w = parts[j]
                    if j == len(parts) - 1:
                        w = trailing
                    rev_tokens.append(_LTok(t, tg, w, ps))
                    rev_ps.append((unk if ps is None else ps) + w)
                if v_out is not _INHERIT:
                    cfv = v_out
                cft = ft_out
                i -= 1
            else:
                j = i
                while j >= 0 and slow[j]:
                    j -= 1
                rkey = (tuple((items[k2][0], items[k2][1], items[k2][2]) for k2 in range(j + 1, i + 1)),
                        last2(j), first2(i + 1), cfv, cft)
                entry = memo.get(rkey)
                if entry is None:
                    mts = []
                    for k2 in range(j + 1, i + 1):
                        kind, txt, tw = items[k2]
                        toks = item_toks[k2]
                        tags = self._tags_windowed(toks, last2(k2 - 1), first2(k2 + 1)) if kind == "c" \
                            else ["_SP"]
                        m = len(toks)
                        for j2, (t, g) in enumerate(zip(toks, tags)):
                            mts.append(MToken(
                                text=t, tag=g, whitespace=(tw if j2 == m - 1 else ""),
                                _=MToken.Underscore(is_head=True, num_flags="", prespace=False)))
                    final, ctx_out = self._process_tokens(mts, TokenContext(future_vowel=cfv, future_to=cft))
                    entry = (self._parts_of(final), ctx_out.future_vowel, ctx_out.future_to)
                    memo[rkey] = entry
                parts, v_out, ft_out = entry
                for j2 in range(len(parts) - 1, -1, -1):
                    t, tg, ps, w = parts[j2]
                    rev_tokens.append(_LTok(t, tg, w, ps))
                    rev_ps.append((unk if ps is None else ps) + w)
                cfv, cft = v_out, ft_out
                i = j

        rev_tokens.reverse()
        rev_ps.reverse()
        return "".join(rev_ps), rev_tokens

    # ---- engine.chunk support ----

    @staticmethod
    def _tokens_to_ps(tokens):
        return "".join(t.phonemes + (" " if t.whitespace else "") for t in tokens).strip()

    @staticmethod
    def _waterfall_last(tokens, next_count, waterfall=("!.?…", ":;", ",—"), bumps=(")", "”")):
        for w in waterfall:
            z = next((i for i, t in reversed(list(enumerate(tokens))) if t.phonemes in set(w)), None)
            if z is None:
                continue
            z += 1
            if z < len(tokens) and tokens[z].phonemes in bumps:
                z += 1
            if next_count - len(FastG2P._tokens_to_ps(tokens[:z])) <= 510:
                return z
        return len(tokens)

    def en_tokenize(self, tokens):
        """Identical grouping to mlx_audio KokoroPipeline.en_tokenize (510-phoneme chunks)."""
        tks = []
        pcount = 0
        for t in tokens:
            t.phonemes = "" if t.phonemes is None else t.phonemes.replace("ɾ", "T")
            next_ps = t.phonemes + (" " if t.whitespace else "")
            next_pcount = pcount + len(next_ps.rstrip())
            if next_pcount > 510:
                z = FastG2P._waterfall_last(tks, next_pcount)
                text = "".join(tk.text + tk.whitespace for tk in tks[:z]).strip()
                ps = FastG2P._tokens_to_ps(tks[:z])
                yield text, ps, tks[:z]
                tks = tks[z:]
                pcount = len(FastG2P._tokens_to_ps(tks))
                if not tks:
                    next_ps = next_ps.lstrip()
            tks.append(t)
            pcount += len(next_ps)
        if tks:
            text = "".join(tk.text + tk.whitespace for tk in tks).strip()
            yield text, FastG2P._tokens_to_ps(tks), tks

    def chunk(self, text, split_pattern=r"\n+"):
        """(graphemes, phonemes, tokens) chunks, mirroring fastkoko.engine.chunk."""
        pieces = [p.strip() for p in re.split(split_pattern, text.strip())] if split_pattern else [text.strip()]
        for piece in pieces:
            if not piece:
                continue
            _, tokens = self(piece)
            for gs, ps, tks in self.en_tokenize(tokens):
                if ps:
                    yield gs, ps[:510], tks
