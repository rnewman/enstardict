"""Label base letters and marks from OCR consensus, then compose exact characters.

OCR reads base letters and above-marks well but strips below-marks, so the
below-marks come from a hand-read table (there are only three kinds).
"""
import pickle, json, collections, unicodedata, sys, numpy as np

words, groups = pickle.load(open('pipe.pkl', 'rb'))
D = pickle.load(open('decomp.pkl', 'rb'))
meta = json.load(open('glyphs.json'))

# class -> base cluster, and class -> list of mark clusters
base_cluster, mark_cluster = {}, collections.defaultdict(list)
for ci, g in enumerate(D['bclusters']):
    for gi in g['members']: base_cluster[gi] = ci
mark_pos = {}
for gi, mk in D['marks_of'].items():
    for j, (w, m, f) in enumerate(mk): mark_pos[(gi, j)] = f
for tag, key in (('a', 'mabove'), ('b', 'mbelow')):
    for ci, g in enumerate(D[key]):
        for (gi, j, where) in g['members']:
            mark_cluster[gi].append(('%s%d' % (tag, ci), mark_pos.get((gi, j), 0.5)))
for gi in mark_cluster: mark_cluster[gi].sort()

cls = {}
for gi, g in enumerate(groups):
    for key, i in g['members']: cls[(key, i)] = gi

NOISE = __import__('re').compile(r'\\[a-zA-Z]+\{?|[*_`${}]|\\')
def clean(s):
    s = NOISE.sub('', unicodedata.normalize('NFC', s))
    for a, b in (('\u2019', "'"), ('\u2018', "'"), ('\u2013', '-'), ('\u2014', '-')):
        s = s.replace(a, b)
    return s.strip()

def ocr_sources():
    out = collections.defaultdict(list)
    for fn in ('jina_all.json', 'dots_all.json'):
        try: d = json.load(open(fn))
        except FileNotFoundError: continue
        for k, v in d.items():
            v = clean(v)
            if v: out[k].append(v)
    return out
OCR = ocr_sources()

WIDTHS = [g['arr'].shape[1] for g in groups]
UNIT = float(np.median([w for w in WIDTHS if w > 4]) or 13.0)
def expect(c):
    """How many characters a shape this wide probably holds."""
    return max(1, min(4, int(round(groups[c]['arr'].shape[1] / UNIT))))

def pieces_for(labels):
    """Assign a substring of each OCR reading to each glyph of each word."""
    got = collections.defaultdict(collections.Counter)
    for k, gs in words.items():
        seqc = [cls[(k, i)] for i in range(len(gs))]
        for s in OCR.get(k, []):
            if len(s) == len(seqc):
                for c, ch in zip(seqc, s): got[c][ch] += 1
                continue
            if not labels or len(s) > 4*len(seqc): continue
            n, m = len(seqc), len(s); INF = float('inf')
            dp = [[INF]*(m+1) for _ in range(n+1)]; bk = [[None]*(m+1) for _ in range(n+1)]
            dp[0][0] = 0
            for i in range(n):
                lab = labels.get(seqc[i])
                for j in range(m+1):
                    if dp[i][j] == INF: continue
                    for w in range(1, 5):
                        if j+w > m: break
                        pc = s[j:j+w]
                        if lab is None:
                            cost = 0.5 + 0.35*abs(w - expect(seqc[i]))
                        else:
                            cost = 0.0 if pc == lab else 1.0+abs(len(lab)-w)*0.5
                        if dp[i][j]+cost < dp[i+1][j+w]:
                            dp[i+1][j+w] = dp[i][j]+cost; bk[i+1][j+w] = (j, pc)
            if dp[n][m] == INF or dp[n][m] > 1.5*n: continue
            i, j = n, m
            while i > 0:
                pj, pc = bk[i][j]; got[seqc[i-1]][pc] += 1; i, j = i-1, pj
    return got

# --- round 1: class-level consensus straight from OCR --------------------
labels = None
for _ in range(3):
    got = pieces_for(labels)
    labels = {c: ctr.most_common(1)[0][0] for c, ctr in got.items()}
conf = {c: (ctr.most_common(1)[0][1]/sum(ctr.values()), sum(ctr.values())) for c, ctr in got.items()}

# --- rounds 2/3: alternate between base letters and marks ---------------
COMB = {'dotbelow':'\u0323', 'underline':'\u0331', 'commabelow':'\u0326',
        'macron':'\u0304', 'dot':'\u0307', 'caron':'\u030c', 'breve':'\u0306',
        'acute':'\u0301', 'grave':'\u0300', 'ring':'\u030a', 'circumflex':'\u0302',
        'diaeresis':'\u0308', 'ogonek':'\u0328', 'tilde':'\u0303', 'cedilla':'\u0327'}
hand = {}
try:
    for line in open('marks_below.tsv', encoding='utf-8'):
        line = line.rstrip('\n')
        if not line.strip() or line.startswith('#'): continue
        p = line.split('\t')
        if len(p) > 1 and not p[1].startswith('?'):
            hand[p[0]] = COMB.get(p[1], p[1])
except FileNotFoundError: pass

def vote_bases(mlabel):
    bv = collections.defaultdict(collections.Counter)
    for c, ctr in got.items():
        bc = base_cluster.get(c)
        if bc is None: continue
        mks = [m for m, _f in mark_cluster.get(c, [])]
        if not mks:
            for piece, n in ctr.items(): bv[bc][piece] += n
            continue
        labs = [mlabel.get(m) for m in mks]
        if any(l is None for l in labs): continue
        for piece, n in ctr.items():
            p = piece
            for l in labs:
                if l.startswith('PRE:') and p.startswith(l[4:]): p = p[len(l)-4:]
                elif l.startswith('POST:') and p.endswith(l[5:]): p = p[:-(len(l)-5)]
            nf = unicodedata.normalize('NFD', p)
            plain = ''.join(ch for ch in nf if not unicodedata.combining(ch))
            bv[bc][plain] += n
    return {b: ctr.most_common(1)[0][0] for b, ctr in bv.items() if ctr}

def vote_marks(blabel):
    mv = collections.defaultdict(collections.Counter)
    for c, ctr in got.items():
        mks = [m for m, _f in mark_cluster.get(c, [])]
        if len(mks) != 1: continue
        b = blabel.get(base_cluster.get(c))
        if not b: continue
        for piece, n in ctr.items():
            if piece == b: mv[mks[0]]['INHERENT'] += n; continue
            nf = unicodedata.normalize('NFD', piece)
            comb = ''.join(ch for ch in nf if unicodedata.combining(ch))
            plain = ''.join(ch for ch in nf if not unicodedata.combining(ch))
            if plain == b and len(comb) == 1: mv[mks[0]][comb] += n
            elif piece.endswith(b) and len(piece) > len(b): mv[mks[0]]['PRE:'+piece[:-len(b)]] += n
            elif piece.startswith(b) and len(piece) > len(b): mv[mks[0]]['POST:'+piece[len(b):]] += n
    out = {m: ctr.most_common(1)[0][0] for m, ctr in mv.items() if ctr}
    out.update(hand)                      # hand-read below marks always win
    return out

blabel, mlabel = vote_bases(hand), dict(hand)
for _ in range(4):
    mlabel = vote_marks(blabel)
    blabel = vote_bases(mlabel)

def compose(c):
    b = blabel.get(base_cluster.get(c))
    if b is None: return None
    pre = post = ''
    # A mark belongs to the character it sits over -- merged shapes such as
    # "eg" can carry a macron on either letter.
    per = [''] * len(b)
    for m, f in mark_cluster.get(c, []):
        l = mlabel.get(m)
        if l is None: return None
        if l == 'INHERENT': continue
        elif l.startswith('PRE:'): pre += l[4:]
        elif l.startswith('POST:'): post += l[5:]
        else: per[min(len(b)-1, int(f*len(b)))] += l
    return pre + unicodedata.normalize('NFC', ''.join(ch+per[i] for i, ch in enumerate(b))) + post

final = {}
for c in range(len(groups)):
    v = compose(c)
    if v is not None: final[c] = v
from pipeline import load_overrides
final.update(load_overrides(groups))
json.dump({'final': {str(k): v for k, v in final.items()},
           'blabel': {str(k): v for k, v in blabel.items()},
           'mlabel': mlabel,
           'ocr': {str(k): v for k, v in labels.items()},
           'conf': {str(k): conf[k] for k in labels},
           'base_cluster': {str(k): v for k, v in base_cluster.items()},
           'mark_cluster': {str(k): [m for m, _f in v] for k, v in mark_cluster.items()}},
          open('votes.json', 'w'), ensure_ascii=False)
print('OCR readings per word: %.2f' % (sum(len(v) for v in OCR.values())/max(1,len(OCR))))
print('base clusters labelled : %d/%d' % (len(blabel), len(D['bclusters'])))
print('mark clusters labelled : %d/%d' % (len(mlabel), len(D['mabove'])+len(D['mbelow'])))
print('shape classes composed : %d/%d' % (len(final), len(groups)))
print('mark labels:', {m: mlabel[m] for m in sorted(mlabel)[:24]})
