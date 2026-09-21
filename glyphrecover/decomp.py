"""Split each shape class into base letter + diacritical marks, and cluster each
separately.  OCR reads base letters reliably; it is only the marks it strips,
and there are very few distinct marks."""
import pickle, numpy as np, collections, json
from scipy import ndimage
from pipeline import close
words, groups = pickle.load(open('pipe.pkl','rb'))

def parts(arr):
    lab, n = ndimage.label(arr, np.ones((3,3), bool))
    cs = []
    for i in range(1, n+1):
        ys, xs = np.where(lab == i)
        cs.append(dict(y0=ys.min(), y1=ys.max(), x0=xs.min(), x1=xs.max(),
                       h=ys.max()-ys.min()+1, m=(lab == i)))
    if not cs: return None, []
    bases = [c for c in cs if c['h'] >= 7] or cs
    marks = [c for c in cs if c['h'] < 7 and c not in bases]
    by0 = min(b['y0'] for b in bases); by1 = max(b['y1'] for b in bases)
    base = np.zeros_like(arr)
    for b in bases: base |= b['m']
    ys, xs = np.where(base)
    base = base[ys.min():ys.max()+1, xs.min():xs.max()+1]
    bx0 = min(b['x0'] for b in bases); bx1 = max(b['x1'] for b in bases)
    out = []
    for mk in marks:
        sub = mk['m'][mk['y0']:mk['y1']+1, mk['x0']:mk['x1']+1]
        where = 'above' if mk['y1'] < (by0+by1)/2 else 'below'
        xf = ((mk['x0']+mk['x1'])/2.0 - bx0) / max(1.0, bx1-bx0)
        out.append((where, sub, min(0.999, max(0.0, xf))))
    return base, out

def cluster(items, tol=0.95):
    """items: list of (key, bitmap) -> list of groups."""
    reps = []
    for key, arr in items:
        hit = None
        for g in reps:
            if abs(g['arr'].shape[0]-arr.shape[0]) <= 1 and abs(g['arr'].shape[1]-arr.shape[1]) <= 1 \
               and close(g['arr'], arr, tol):
                hit = g; break
        if hit is None: reps.append(dict(arr=arr, members=[key], n=1))
        else: hit['members'].append(key); hit['n'] += 1
    reps.sort(key=lambda g: -g['n'])
    return reps

base_of, marks_of = {}, {}
for gi, g in enumerate(groups):
    b, mk = parts(g['arr'])
    if b is None: continue
    base_of[gi] = b; marks_of[gi] = mk

bclusters = cluster([(gi, b) for gi, b in base_of.items()])
mitems = [((gi, j, w), m) for gi, mk in marks_of.items() for j, (w, m, _f) in enumerate(mk)]
# cluster marks by shape, keeping above/below separate
mabove = cluster([(k, m) for k, m in mitems if k[2] == 'above'], tol=0.90)
mbelow = cluster([(k, m) for k, m in mitems if k[2] == 'below'], tol=0.90)
print('shape classes: %d' % len(groups))
print('base-letter clusters: %d  (weighted by class use: top %s)'
      % (len(bclusters), [g['n'] for g in bclusters[:12]]))
print('mark clusters: above %d, below %d' % (len(mabove), len(mbelow)))
print('  above sizes:', [g['n'] for g in mabove[:14]])
print('  below sizes:', [g['n'] for g in mbelow[:14]])
nmarks = collections.Counter(len(v) for v in marks_of.values())
print('marks per class:', sorted(nmarks.items()))
pickle.dump(dict(base_of=base_of, marks_of=marks_of, bclusters=bclusters,
                 mabove=mabove, mbelow=mbelow), open('decomp.pkl','wb'))
