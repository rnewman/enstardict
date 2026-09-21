"""Segment every glyph image into characters and cluster identical shapes."""
import json, numpy as np, collections, pickle, hashlib, sys
from PIL import Image
from scipy import ndimage
SLOPE = 0.30

def load(p): return np.array(Image.open(p).convert('L')) < 170

def deslant(mask, slope=SLOPE):
    h, w = mask.shape
    pad = int(abs(slope)*h)+2
    out = np.zeros((h, w+2*pad), bool)
    for y in range(h):
        dx = pad - int(round(slope*(h-1-y)))
        out[y, dx:dx+w] = mask[y]
    return out

def glyphs(mask, slope=SLOPE):
    """Components of the ORIGINAL bitmap, but marks are assigned to bases using
    de-slanted x-coordinates -- in italic the dot of an i sits over the NEXT letter."""
    lab, n = ndimage.label(mask, np.ones((3,3), bool))
    if n == 0: return []
    H = mask.shape[0]
    cs = []
    for i in range(1, n+1):
        ys, xs = np.where(lab == i)
        dx = xs - np.round(slope*(H-1-ys)).astype(int)   # upright coordinates
        cs.append(dict(x0=xs.min(), x1=xs.max(), y0=ys.min(), y1=ys.max(),
                       d0=dx.min(), d1=dx.max(), cx=dx.mean(),
                       h=ys.max()-ys.min()+1, m=(lab == i)))
    bases = [c for c in cs if c['h'] >= 7]
    small = [c for c in cs if c['h'] < 7]
    if not bases: bases, small = cs, []
    # Decide mark-vs-character against the letter the blob actually sits on:
    # a macron over a short letter still sits below the top of a tall one.
    def owner(mk):
        best, bx = None, -1e9
        for b in bases:
            ov = (min(mk['d1'], b['d1']) - max(mk['d0'], b['d0'])) / max(1, mk['d1']-mk['d0'])
            if ov > bx: bx, best = ov, b
        return best
    pairs = []
    for mk in small:
        b = owner(mk)
        if b is not None and (mk['y1'] < b['y0'] + 2 or mk['y0'] > b['y1'] - 2):
            pairs.append((mk, b))
        else:
            bases.append(mk)           # a hyphen, a stray dot: its own glyph
    for mk, best in pairs:
        best['m'] |= mk['m']
        best['x0'] = min(best['x0'], mk['x0']); best['x1'] = max(best['x1'], mk['x1'])
        best['y0'] = min(best['y0'], mk['y0']); best['y1'] = max(best['y1'], mk['y1'])
    bases.sort(key=lambda b: b['cx'])
    return [dict(sub=b['m'][b['y0']:b['y1']+1, b['x0']:b['x1']+1],
                 x0=int(b['x0']), x1=int(b['x1']), y0=int(b['y0']), y1=int(b['y1'])) for b in bases]

def class_hash(arr):
    """Stable id for a shape class, so an override cannot silently mis-apply."""
    return hashlib.sha1(arr.tobytes() + repr(arr.shape).encode()).hexdigest()[:12]

def load_overrides(groups, path='overrides.tsv'):
    """class index TAB text [TAB expected shape hash]; hand-read corrections."""
    out = {}
    try: fh = open(path, encoding='utf-8')
    except FileNotFoundError: return out
    for line in fh:
        line = line.rstrip('\n')
        if not line.strip() or line.lstrip().startswith('#'): continue
        p = line.split('\t')
        gi = int(p[0]); text = (p[1] if len(p) > 1 else '').replace('\\s', ' ')
        want = p[2].strip() if len(p) > 2 else ''
        if want and (gi >= len(groups) or class_hash(groups[gi]['arr']) != want):
            print('overrides: class %d no longer matches %s -- skipped' % (gi, want), file=sys.stderr)
            continue
        out[gi] = text
    return out

def close(a, b, tol=0.97):
    if abs(a.shape[0]-b.shape[0]) > 1 or abs(a.shape[1]-b.shape[1]) > 1: return False
    H, W = max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1])
    pa = np.zeros((H+2, W+2), bool); pa[1:1+a.shape[0], 1:1+a.shape[1]] = a
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            pb = np.zeros((H+2, W+2), bool); pb[1+dy:1+dy+b.shape[0], 1+dx:1+dx+b.shape[1]] = b
            u = (pa | pb).sum()
            if u and (pa & pb).sum()/u >= tol: return True
    return False

if __name__ == '__main__':
    meta = json.load(open('glyphs.json'))
    words = {}
    exact = collections.defaultdict(list)
    for key, m in meta.items():
        gs = glyphs(load(m['file']))
        words[key] = gs
        top = min([g['y0'] for g in gs], default=0); bot = max([g['y1'] for g in gs], default=0)
        for i, g in enumerate(gs):
            band = (round((g['y0']-top)/4.0), round((bot-g['y1'])/4.0))
            exact[(g['sub'].shape, band, g['sub'].tobytes())].append((key, i))
    reps = sorted(({'arr': np.frombuffer(k[2], bool).reshape(k[0]), 'band': k[1],
                    'n': len(v), 'members': v} for k, v in exact.items()), key=lambda r: -r['n'])
    groups, bysize = [], collections.defaultdict(list)
    for r in reps:
        h, w = r['arr'].shape; hit = None
        for dh in (-1,0,1):
            for dw in (-1,0,1):
                for gi in bysize.get((h+dh, w+dw, r['band']), []):
                    if close(groups[gi]['arr'], r['arr']): hit = gi; break
                if hit is not None: break
            if hit is not None: break
        if hit is None:
            bysize[(h, w, r['band'])].append(len(groups)); groups.append(r)
        else:
            groups[hit]['n'] += r['n']; groups[hit]['members'] += r['members']
    groups.sort(key=lambda g: -g['n'])
    tot = sum(g['n'] for g in groups)
    print('images %d  glyph instances %d  shape classes %d' % (len(meta), tot, len(groups)))
    cum = np.cumsum([g['n'] for g in groups])
    for p in (0.90, 0.95, 0.99, 1.0):
        print('  %.0f%% of instances: %d classes' % (100*p, int(np.searchsorted(cum, p*tot))+1))
    pickle.dump((words, groups), open('pipe.pkl','wb'))
