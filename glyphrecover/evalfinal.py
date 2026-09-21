import pickle, json, unicodedata
words, groups = pickle.load(open('pipe.pkl','rb'))
V=json.load(open('votes.json'))
final={int(k):v for k,v in V['final'].items()}; ocr={int(k):v for k,v in V['ocr'].items()}
truth={}
for line in open('eval120.tsv', encoding='utf-8'):
    p=line.rstrip('\n').split('\t')
    if len(p)>1: truth[int(p[0])]=p[1]
N=lambda s: unicodedata.normalize('NFC', s)
ok=bad=miss=0; errs=[]; ocrok=0
for gi,lab in truth.items():
    o=ocr.get(gi)
    if o is not None and N(o)==N(lab): ocrok+=1
    g=final.get(gi)
    if g is None: miss+=1; errs.append((gi,lab,'<none>',groups[gi]['n'])); continue
    if N(g)==N(lab): ok+=1
    else: bad+=1; errs.append((gi,lab,g,groups[gi]['n']))
tot=len(truth)
print('eval on %d hand-read classes (%.0f%% of all glyph instances)'
      % (tot, 100*sum(groups[g]['n'] for g in truth)/sum(g['n'] for g in groups)))
print('  composed correct : %d (%.1f%%)' % (ok, 100*ok/tot))
print('  composed wrong   : %d   not composed: %d' % (bad, miss))
print('  raw OCR consensus: %d (%.1f%%)' % (ocrok, 100*ocrok/tot))
if errs:
    print('  errors:')
    for e in sorted(errs, key=lambda x:-x[3]): print('    class %-5d hand=%-6r got=%-8r x%d' % e)
