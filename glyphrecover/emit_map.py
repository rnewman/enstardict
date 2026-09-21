"""Reconstruct every glyph image and append it to the dictionary's glyph map."""
import pickle, json, numpy as np, collections, unicodedata, sys
words, groups = pickle.load(open('pipe.pkl','rb'))
meta = json.load(open('glyphs.json'))
V = json.load(open('votes.json'))
final = {int(k): v for k, v in V['final'].items()}
from pipeline import load_overrides
final.update(load_overrides(groups))
cls = {}
for gi, g in enumerate(groups):
    for key, i in g['members']: cls[(key, i)] = gi
out, blocked = {}, collections.Counter()
for key, gs in words.items():
    if not gs: continue
    med = np.median([g['x1']-g['x0']+1 for g in gs])
    parts, ok = [], True
    for i, g in enumerate(gs):
        c = cls[(key, i)]
        if c not in final: blocked[c] += 1; ok = False; break
        if i and (g['x0'] - gs[i-1]['x1'])/max(1.0, med) > 0.5: parts.append(' ')
        parts.append(final[c])
    if ok:
        s = unicodedata.normalize('NFC', ''.join(parts)).strip()
        if s: out[key] = s
# Cross-check: OCR reads the letters reliably even when it drops the marks,
# so a reconstruction shorter than what both models agree on has lost a
# letter -- usually a merged shape labelled with only part of its content.
import unicodedata as _ud, re as _re
_NOISE = _re.compile(r'\\[a-zA-Z]+\{?|[*_`${}]|\\')
def _clean(s):
    return _NOISE.sub('', _ud.normalize('NFC', s)).strip()
_ocr = {}
for _fn in ('jina_all.json', 'dots_all.json'):
    try: _d = json.load(open(_fn))
    except FileNotFoundError: continue
    for _k, _v in _d.items(): _ocr.setdefault(_k, []).append(_clean(_v))
short = []
for k, v in out.items():
    rs = [r for r in _ocr.get(k, []) if r]
    if len(rs) == 2 and len(rs[0]) == len(rs[1]) and len(v) < len(rs[0]):
        short.append((len(rs[0]) - len(v), k, v, rs[0]))
short.sort(reverse=True)
if short:
    print('\nSHORT: %d reconstructions lost a letter both OCR models saw:' % len(short))
    for d, k, v, r in short[:15]:
        print('   %-6s -%d  ours=%-18r ocr=%r' % (k, d, v, r))

refs = sum(meta[k]['uses'] for k in out)
tot = sum(m['uses'] for m in meta.values())
print('reconstructed %d/%d images (%.1f%%), %d/%d references (%.1f%%)'
      % (len(out), len(meta), 100*len(out)/len(meta), refs, tot, 100*refs/tot))
print('blocked by %d unlabelled classes; worst: %s' % (len(blocked), blocked.most_common(6)))
json.dump(out, open('recon.json','w'), ensure_ascii=False, indent=0)
with open('glyphmap_add.tsv','w', encoding='utf-8') as f:
    f.write('\n# --- etymology and usage-note art, recovered by OCR consensus ---\n')
    for k in sorted(out, key=lambda x: int(x)):
        v = out[k].replace('\t',' ')
        f.write('%s\t%s\n' % (k, v))
print('wrote glyphmap_add.tsv')
print('\nsample:')
for k in sorted(out, key=lambda x: -meta[x]['uses'])[:18]:
    print('   %-6s x%-4d %r' % (k, meta[k]['uses'], out[k]))
