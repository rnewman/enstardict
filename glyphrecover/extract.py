"""Enumerate every inline image still unmapped, keyed the way the glyph map is
(by the tag's recindex), with its hi-res artwork, usage count and a context sample."""
import struct, re, json, os, collections, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOBI = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, 'american.mobi')
GLYPHS = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, 'american.glyphs')
from calibre.ebooks.mobi.reader.index import (parse_indx_header, parse_tagx_section,
                                              parse_index_record, get_tag_section_start)
import importlib.util
spec = importlib.util.spec_from_file_location('m2s', os.path.join(REPO, 'mobidict2stardict.py'))
m2s = importlib.util.module_from_spec(spec); spec.loader.exec_module(m2s)

mobi = m2s.Mobi(MOBI)
text = mobi.text()
entries, _ = mobi.index(mobi.orth_index)
known, _runs = m2s.read_glyph_map(GLYPHS)

IMG = re.compile(rb'<img\b[^>]*>')
count = collections.Counter(); hires = {}; ctx = {}
for label, tm in entries:
    if 1 not in tm or 2 not in tm: continue
    off, ln = tm[1][0], tm[2][0]
    body = text[off:off+ln]
    for m in IMG.finditer(body):
        ids = {k: int(v) for k, v in m2s.RE_RECIDX.findall(m.group(0))}
        key = ids.get(b'recindex') or ids.get(b'hirecindex') or ids.get(b'lorecindex')
        if key is None or key in known: continue
        count[key] += 1
        hires.setdefault(key, ids.get(b'hirecindex') or key)
        if key not in ctx:
            a = max(0, m.start()-170); b = min(len(body), m.end()+90)
            s = body[a:b].decode('utf-8','replace')
            s = re.sub(r'<img[^>]*\brecindex="0*(\d+)"[^>]*>', lambda x: '{%s}'%x.group(1).lstrip('0'), s)
            ctx[key] = ' '.join(re.sub(r'<[^>]+>',' ',s).split())
os.makedirs('glyphs', exist_ok=True)
meta = {}
for key, n in count.most_common():
    hi = hires[key]
    data = mobi.image(hi)
    ext = 'gif' if data[:3]==b'GIF' else 'png' if data[:4]==b'\x89PNG' else 'jpg' if data[:2]==b'\xff\xd8' else 'bin'
    fn = 'glyphs/%05d.%s' % (key, ext)
    open(fn,'wb').write(data)
    w,h = struct.unpack_from('<HH', data, 6) if ext=='gif' else (0,0)
    meta[key] = dict(file=fn, hi=hi, uses=n, w=w, h=h, ctx=ctx[key][:220])
json.dump(meta, open('glyphs.json','w'), indent=0)
print('unmapped distinct images:', len(meta), 'total refs:', sum(count.values()))
print('already mapped:', len(known))
