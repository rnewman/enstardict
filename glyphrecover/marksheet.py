import pickle, numpy as np, sys
from PIL import Image, ImageDraw, ImageFont
words, groups = pickle.load(open('pipe.pkl','rb'))
D = pickle.load(open('decomp.pkl','rb'))
which = sys.argv[1]           # above | below
cl = D['mabove'] if which=='above' else D['mbelow']
COLS, cw, ch = 8, 210, 150
rows=(len(cl)+COLS-1)//COLS
sh=Image.new('RGB',(COLS*cw, rows*ch),(255,255,255)); dr=ImageDraw.Draw(sh)
fb=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf', 20)
fs=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 14)
for k,g in enumerate(cl):
    x,y=(k%COLS)*cw,(k//COLS)*ch
    dr.rectangle([x+2,y+2,x+cw-4,y+ch-4], outline=(220,220,220))
    dr.text((x+6,y+4), '%s%d' % (which[0], k), fill=(200,0,0), font=fb)
    dr.text((x+cw-40,y+6), 'x%d'%g['n'], fill=(110,110,180), font=fs)
    m=Image.fromarray((~g['arr']*255).astype(np.uint8)).convert('RGB')
    s=min(9,(cw-20)/max(1,m.width), 46/max(1,m.height))
    m=m.resize((max(1,int(m.width*s)),max(1,int(m.height*s))), Image.LANCZOS)
    sh.paste(m,(x+10,y+30))
    gi=g['members'][0][0]
    ex=Image.fromarray((~groups[gi]['arr']*255).astype(np.uint8)).convert('RGB')
    s=min(4,(cw-20)/max(1,ex.width), 58/max(1,ex.height))
    ex=ex.resize((max(1,int(ex.width*s)),max(1,int(ex.height*s))), Image.LANCZOS)
    sh.paste(ex,(x+10,y+ch-8-ex.height))
sh.save('marks_%s.png'%which); print('marks_%s.png'%which, sh.size, len(cl),'clusters')
