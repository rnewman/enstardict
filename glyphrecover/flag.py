"""Find shape classes whose composed label is probably missing a diacritic.

If two base clusters vote the same letter but one is visibly taller or wider,
the taller one is carrying a mark fused to the letter that OCR silently drops.
"""
import pickle, json, collections
words, groups = pickle.load(open('pipe.pkl','rb'))
D = pickle.load(open('decomp.pkl','rb'))
V = json.load(open('votes.json'))
final = {int(k): v for k, v in V['final'].items()}
blabel = V['blabel']
bsize = {}
for ci, g in enumerate(D['bclusters']):
    bsize[str(ci)] = g['arr'].shape          # (h, w)
by = collections.defaultdict(list)
for ci, lab in blabel.items(): by[lab].append(ci)
sus = []
for lab, cis in by.items():
    if len(cis) < 2: continue
    hs = [(bsize[c][0], c) for c in cis if bsize[c][0] >= 7]
    if len(hs) < 2: continue
    lo = min(h for h, _ in hs)
    for h, c in hs:
        if h >= lo + 3:                      # clearly taller than its twins
            sus.append((lab, c, h, lo))
bc = {int(k): v for k, v in V['base_cluster'].items()}
n_cls = collections.Counter()
for gi, ci in bc.items(): n_cls[str(ci)] += groups[gi]['n']
sus.sort(key=lambda s: -n_cls[s[1]])
print('base clusters that look like a letter + a fused mark OCR dropped:')
for lab, c, h, lo in sus[:25]:
    ex = [gi for gi, ci in bc.items() if str(ci) == c]
    print('   label %-4r cluster %-4s h=%d (twin h=%d)  x%d  classes=%s'
          % (lab, c, h, lo, n_cls[c], ex[:4]))
print('total flagged clusters:', len(sus))
missing = [gi for gi in range(len(groups)) if gi not in final]
print('classes with no composition at all:', len(missing),
      '(%d instances)' % sum(groups[g]['n'] for g in missing))
open('flagged.txt','w').write('\n'.join(str(gi) for gi in
    sorted(set([e for lab,c,h,lo in sus for e in [gi for gi,ci in bc.items() if str(ci)==c]] + missing),
           key=lambda g: -groups[g]['n'])))
