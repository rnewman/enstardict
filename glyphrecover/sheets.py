import pickle, numpy as np, sys
from PIL import Image, ImageDraw, ImageFont
words, groups = pickle.load(open('pipe.pkl','rb'))
PER, COLS, S = 120, 10, 5
cw, ch = 150, 132
f = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf', 19)
fs = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 13)
n = 0
for page in range((len(groups)+PER-1)//PER):
    sel = groups[page*PER:(page+1)*PER]
    rows = (len(sel)+COLS-1)//COLS
    sh = Image.new('RGB', (COLS*cw, rows*ch), (255,255,255))
    dr = ImageDraw.Draw(sh)
    for k, g in enumerate(sel):
        gi = page*PER + k
        x, y = (k % COLS)*cw, (k//COLS)*ch
        a = g['arr']
        im = Image.fromarray((~a*255).astype(np.uint8)).convert('RGB')
        sc = min(S, (cw-12)/max(1,im.width), (ch-40)/max(1,im.height))
        im = im.resize((max(1,int(im.width*sc)), max(1,int(im.height*sc))), Image.LANCZOS)
        dr.rectangle([x+2, y+2, x+cw-4, y+ch-4], outline=(224,224,224))
        dr.text((x+6, y+4), str(gi), fill=(200,0,0), font=f)
        dr.text((x+cw-34, y+6), str(g['n']), fill=(120,120,180), font=fs)
        sh.paste(im, (x+(cw-im.width)//2, y+30+(ch-36-im.height)//2))
    sh.save('g%02d.png' % page)
    n += 1
print('pages', n, 'classes', len(groups))
