"""Render a review sheet: each flagged class shown large, next to whole words
that contain it (the glyph boxed), so it can be judged in context."""
import pickle, json, sys, numpy as np
from PIL import Image, ImageDraw, ImageFont
words, groups = pickle.load(open('pipe.pkl','rb'))
meta = json.load(open('glyphs.json'))
idxs = [int(x) for x in sys.argv[1].split(',')] if sys.argv[1] != '-' else None
out = sys.argv[2]
if idxs is None:
    idxs = [int(x) for x in open('flagged.txt').read().split()]
GS, WS = 7, 4
cw, rowh = 1700, 118
sheet = Image.new('RGB', (cw, rowh*len(idxs)), (255,255,255))
dr = ImageDraw.Draw(sheet)
fb = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf', 22)
fs = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 15)
ctxs = []
for r, gi in enumerate(idxs):
    g = groups[gi]
    y = r*rowh
    dr.line([(0, y), (cw, y)], fill=(215,215,215))
    dr.text((6, y+8), str(gi), fill=(200,0,0), font=fb)
    dr.text((6, y+38), 'x%d' % g['n'], fill=(110,110,180), font=fs)
    im = Image.fromarray((~g['arr']*255).astype(np.uint8)).convert('RGB')
    s = min(GS, (rowh-16)/max(1, im.height))
    im = im.resize((max(1,int(im.width*s)), max(1,int(im.height*s))), Image.LANCZOS)
    sheet.paste(im, (62, y+(rowh-im.height)//2))
    x = 210
    seen = []
    for key, pos in g['members'][:3]:
        if key in seen: continue
        seen.append(key)
        w = Image.open(meta[key]['file']).convert('RGB')
        w = w.resize((w.width*WS, w.height*WS), Image.LANCZOS)
        if x + w.width + 30 > cw: break
        sheet.paste(w, (x, y+(rowh-w.height)//2))
        gl = words[key][pos]
        dr.rectangle([x+gl['x0']*WS-2, y+(rowh-w.height)//2+gl['y0']*WS-2,
                      x+gl['x1']*WS+2, y+(rowh-w.height)//2+gl['y1']*WS+2],
                     outline=(220,0,0))
        x += w.width + 40
    ctxs.append((gi, g['n'], [(k, meta[k]['ctx']) for k in seen[:2]]))
sheet.save(out)
print('wrote', out, sheet.size)
for gi, n, cs in ctxs:
    print('--- class %d (x%d)' % (gi, n))
    for k, c in cs: print('      [%s] %s' % (k, c[:160]))
