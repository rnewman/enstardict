#!/usr/bin/env python3
"""Convert a Kindle/Mobipocket dictionary (.mobi) straight to StarDict.

No HTML export step: the PalmDOC text records and the orthographic (ORTH)
index are read directly, using calibre's battle-tested MOBI index parser for
the fiddly INDX/TAGX/ORDT decoding.  Inflections and variants recorded in the
index become a StarDict .syn file.

Kindle dictionaries often set their pronunciation respelling and section
headings as one tiny image per character.  A glyph map (--glyph-map, see
american.glyphs) turns those back into text; whatever is left over is
extracted to a res/ directory and referenced as an image.  Use --dump-glyphs
to bootstrap a map for a dictionary that does not have one yet.

Run it with a normal python3 -- it re-execs itself under calibre's interpreter
(calibre-debug) automatically, since that is where the parser lives.

    ./mobidict2stardict.py american.mobi -o out/

Output is StarDict 3.0.0 with 32-bit index offsets and an uncompressed .idx.
"""

import os
import sys

# --- calibre bootstrap -----------------------------------------------------
try:
    from calibre.ebooks.mobi.huffcdic import HuffReader
    from calibre.ebooks.mobi.reader.index import (CNCX, get_tag_section_start,
                                                  parse_index_record,
                                                  parse_indx_header,
                                                  parse_tagx_section)
    from calibre.ebooks.mobi.reader.mobi6 import decompress_doc
    from calibre.ebooks.mobi.utils import get_trailing_data
except ImportError:  # not running under calibre's python yet
    import shutil
    _CANDIDATES = ('/Applications/calibre.app/Contents/MacOS/calibre-debug',
                   '/usr/bin/calibre-debug', '/usr/local/bin/calibre-debug',
                   '/opt/homebrew/bin/calibre-debug')
    _exe = shutil.which('calibre-debug') or next(
        (p for p in _CANDIDATES if os.path.exists(p)), None)
    if not _exe:
        sys.exit('This tool needs calibre installed (for its MOBI parser).\n'
                 'Install calibre, or put calibre-debug on your PATH.')
    os.execv(_exe, [_exe, '-e', os.path.abspath(__file__), '--'] + sys.argv[1:])

import argparse
import bisect
import re
import struct
import time
import zlib
from collections import OrderedDict

# --- MOBI ------------------------------------------------------------------

PALMDOC, HUFFCDIC = 2, 17480


class Mobi:
    """Enough of a MOBI reader to get at the raw text, the indexes and images."""

    def __init__(self, path):
        raw = open(path, 'rb').read()
        if raw[60:68] != b'BOOKMOBI':
            raise SystemExit('%s: not a MOBI file' % path)
        n = struct.unpack_from('>H', raw, 76)[0]
        offs = [struct.unpack_from('>I', raw, 78 + i * 8)[0] for i in range(n)]
        offs.append(len(raw))
        self.sec = [raw[offs[i]:offs[i + 1]] for i in range(n)]
        r0 = self.sec[0]
        if r0[16:20] != b'MOBI':
            raise SystemExit('%s: no MOBI header (Palm DOC only?)' % path)
        u32 = lambda o: struct.unpack_from('>I', r0, o)[0]
        self.compression = struct.unpack_from('>H', r0, 0)[0]
        self.encryption = struct.unpack_from('>H', r0, 12)[0]
        self.ntext = struct.unpack_from('>H', r0, 8)[0]
        self.hdrlen = u32(0x14)
        self.codec = {1252: 'cp1252', 65001: 'utf-8'}.get(u32(0x1C), 'utf-8')
        self.orth_index = u32(0x28)
        self.first_image = u32(0x6C)
        self.huff_off, self.huff_num = u32(0x70), u32(0x74)
        self.extra_flags = (struct.unpack_from('>H', r0, 0xF2)[0]
                            if self.hdrlen + 16 >= 0xF4 else 0)
        off, ln = u32(0x54), u32(0x58)
        self.fullname = r0[off:off + ln].decode(self.codec, 'replace')
        self.exth = self._exth(r0)
        if self.encryption:
            raise SystemExit('%s is DRM-encrypted; cannot convert.' % path)

    def _exth(self, r0):
        out = {}
        p = 16 + self.hdrlen
        if r0[p:p + 4] != b'EXTH':
            return out
        count = struct.unpack_from('>I', r0, p + 8)[0]
        p += 12
        for _ in range(count):
            t, ln = struct.unpack_from('>II', r0, p)
            out.setdefault(t, r0[p + 8:p + ln])
            p += ln
        return out

    def meta(self, tag, default=''):
        v = self.exth.get(tag)
        return v.decode(self.codec, 'replace').strip() if v else default

    def text(self):
        """The decompressed book text, with per-record trailing data removed."""
        if self.compression == HUFFCDIC:
            huffs = [self.sec[self.huff_off + i] for i in range(self.huff_num)]
            unpack = HuffReader(huffs).unpack
        elif self.compression == PALMDOC:
            unpack = decompress_doc
        elif self.compression == 1:
            unpack = lambda d: d
        else:
            raise SystemExit('unsupported compression %d' % self.compression)
        out = bytearray()
        for i in range(1, self.ntext + 1):
            _, rec = get_trailing_data(self.sec[i], self.extra_flags)
            out += unpack(bytes(rec))
        return bytes(out)

    def image(self, recindex):
        """Image record by 1-based recindex, as used by <img recindex=...>."""
        i = self.first_image + recindex - 1
        return self.sec[i] if 0 <= i < len(self.sec) else b''

    def index(self, idx):
        """Parse an INDX chain -> list of (label, {tag: [values]}), in order."""
        data = self.sec[idx]
        h = parse_indx_header(data)
        ordt_map = self._ordt(data, h)
        start = get_tag_section_start(data, h)
        control_bytes, tags = parse_tagx_section(data[start:])
        entries = _Ordered()
        for i in range(h['count']):
            parse_index_record(entries, self.sec[idx + 1 + i], control_bytes,
                               tags, self.codec, ordt_map)
        cncx = CNCX([self.sec[idx + h['count'] + 1 + i]
                     for i in range(h['ncncx'])], self.codec)
        return entries.list, cncx

    @staticmethod
    def _ordt(data, h):
        """Build the ORDT byte->character map (calibre truncates these to 8 bit)."""
        for off, width in ((h['ordt2'], 2), (h['ordt1'], 1)):
            if not off:
                continue
            base = off + 4 if data[off:off + 4] == b'ORDT' else off
            n = h['oentries']
            if width == 2:
                vals = struct.unpack_from('>%dH' % n, data, base)
            else:
                vals = struct.unpack_from('>%dB' % n, data, base)
            return ''.join(map(chr, vals))
        return ''


class _Ordered(dict):
    """parse_index_record() writes into a dict; keep duplicates and order."""

    def __init__(self):
        dict.__init__(self)
        self.list = []

    def __setitem__(self, key, value):
        self.list.append((key, value))


# --- MOBI markup -> StarDict HTML -----------------------------------------

CP1252 = {n: bytes([n]).decode('cp1252', 'replace').encode('utf-8')
          for n in range(128, 160)}
RE_RECIDX = re.compile(rb'\b(hirecindex|lorecindex|recindex)="?(\d+)"?')
RE_ALIGN = re.compile(rb'\balign="(\w+)"')
RE_WIDTH = re.compile(rb'\bwidth="(-?\d+)"')
RE_IMGTAG = re.compile(rb'<img\b[^>]*>')
RE_MASTER = re.compile(
    rb'(?:<img\b[^>]*>\s*)+'                # a run of inline glyphs
    rb'|<a\s*>\s*(?:<mbp:nu\s*/?>\s*)?</a>'  # empty anchors MOBI litters
    rb'|<a\s+filepos=(\d+)[^>]*>'           # cross-reference
    rb'|</?mbp:[A-Za-z]+\s*/?>'             # mbp:nu, mbp:pagebreak, frameset
    rb'|<div\b[^>]*>'                       # width= indentation
    rb'|&#(1(?:2[89]|[345][0-9]));'         # cp1252 numeric entities
)
ESCAPE = ((b'&', b'&amp;'), (b'"', b'&quot;'), (b'<', b'&lt;'), (b'>', b'&gt;'))
# Glyph output is fenced with \x01 so the whitespace the MOBI puts between
# adjacent images -- including across the <u>/<font> wrappers it uses to
# style them -- can be dropped once those images have become text.
FENCE = b'\x01'
RE_JOIN = re.compile(rb'\x01\s*((?:<[^>]+>\s*)*)\x01')
RE_TAG = re.compile(rb'<[^>]+>')
RE_EMPTY = re.compile(rb'<(b|i|u|sub|sup)></\1>|<font\b[^>]*></font>')
# Kindle butts the grey grammar label straight onto the bold inflected form
# with no space of its own -- "pl.mice", "pastran", "past part.run" -- 5,058
# times, so its renderer must be separating them at the tag boundary.  Only
# grammar labels ever take this grey-italic-then-bold shape (example sentences
# are blue, and word splits like "televi"+"sion" have no closing </font>), so
# the repair is exact.  Labels that already end in a space are left alone.
RE_LABEL = re.compile(rb'(<font color="#555555"><i>[^<]{1,40}(?<!\s)</i></font>)(<b\b)')


def read_glyph_map(path):
    """Parse a glyph map: <recindex>[,<recindex>...] TAB <replacement>."""
    single, runs = {}, {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            parts = line.split('\t')
            keys = tuple(int(k) for k in parts[0].split(',') if k.strip())
            if not keys:
                continue
            value = (parts[1] if len(parts) > 1 else '')
            value = value.replace('\\s', ' ').encode('utf-8')
            (single if len(keys) == 1 else runs)[keys if len(keys) > 1 else keys[0]] = value
    return single, runs


class Cleaner:
    """Rewrites one MOBI entry body into self-contained HTML."""

    def __init__(self, mobi, resolve, images='hi', scale=1.0, glyphs=None):
        self.mobi, self.resolve, self.images, self.scale = mobi, resolve, images, scale
        self.glyph, self.runs = glyphs or ({}, {})
        self.maxrun = max([len(k) for k in self.runs] + [1])
        self.used = {}          # recindex -> filename
        self.nglyph = self.nimg = 0
        self._dims = {}

    def __call__(self, body):
        body = RE_LABEL.sub(rb'\1 \2', body.replace(FENCE, b''))
        out = RE_MASTER.sub(self._dispatch, body)
        out = RE_JOIN.sub(lambda m: b''.join(RE_TAG.findall(m.group(1))), out)
        out = out.replace(FENCE, b'')
        return RE_EMPTY.sub(b'', out).strip()

    def _dispatch(self, m):
        tag = m.group(0)
        head = tag[:2]
        if head == b'<i':
            return self._imgs(tag)
        if head == b'<a':
            return self._link(m.group(1)) if m.group(1) else b''
        if head == b'<d':
            return self._div(tag)
        if head == b'&#':
            return CP1252[int(m.group(2))]
        return b''              # mbp:* open or close

    def _imgs(self, run):
        """Turn a run of <img> tags into text and/or image references.

        Images are handled a run at a time because a pronunciation is one
        image per character: they have to be joined without the whitespace
        that separates the tags, and some map to text only as a pair.
        """
        tags = RE_IMGTAG.findall(run)
        ids = [{k: int(v) for k, v in RE_RECIDX.findall(t)} for t in tags]
        keys = [i.get(b'recindex') or i.get(b'hirecindex') or i.get(b'lorecindex')
                for i in ids]
        out, i, n = [], 0, len(tags)
        while i < n:
            for size in range(min(self.maxrun, n - i), 1, -1):
                value = self.runs.get(tuple(keys[i:i + size]))
                if value is not None:
                    out.append(value)
                    self.nglyph += size
                    i += size
                    break
            else:
                value = self.glyph.get(keys[i])
                if value is None:
                    out.append(self._img(tags[i], ids[i]))
                    self.nimg += 1
                else:
                    out.append(value)
                    self.nglyph += 1
                i += 1
        return (FENCE + b''.join(out) + FENCE
                + (b' ' if run[-1:].isspace() else b''))

    def _div(self, tag):
        out = b'<div'
        align = RE_ALIGN.search(tag)
        if align:
            out += b' align="' + align.group(1) + b'"'
        width = RE_WIDTH.search(tag)
        if width and int(width.group(1)) < 0:   # MOBI hanging indent
            out += b' style="margin-left:1.3em"'
        return out + b'>'

    def _link(self, filepos):
        word = self.resolve(int(filepos))
        if not word:
            return b'<a>'
        for a, b in ESCAPE:
            word = word.replace(a, b)
        return b'<a href="bword://' + word + b'">'

    def _dim(self, recindex):
        if recindex not in self._dims:
            data = self.mobi.image(recindex)
            if data[:3] == b'GIF':
                self._dims[recindex] = struct.unpack_from('<HH', data, 6)
            elif data[:8] == b'\x89PNG\r\n\x1a\n':
                self._dims[recindex] = struct.unpack_from('>II', data, 16)
            else:
                self._dims[recindex] = None
        return self._dims[recindex]

    def _name(self, recindex):
        name = self.used.get(recindex)
        if name is None:
            data = self.mobi.image(recindex)
            ext = ('gif' if data[:3] == b'GIF' else
                   'png' if data[:4] == b'\x89PNG' else
                   'jpg' if data[:2] == b'\xff\xd8' else 'bin')
            name = self.used[recindex] = b'%05d.%s' % (recindex, ext.encode())
        return name

    def _img(self, tag, ids):
        if self.images == 'none':
            return b''
        lo = ids.get(b'lorecindex') or ids.get(b'recindex')
        hi = ids.get(b'hirecindex') or ids.get(b'recindex')
        pick = hi if self.images == 'hi' else lo
        if not pick:
            return b''
        out = b'<img src="' + self._name(pick) + b'"'
        # Size hi-res art to the metrics the low-res art was drawn for, so
        # inline glyphs keep their intended size but stay crisp.
        if self.images == 'hi' and lo and lo != hi:
            dim = self._dim(lo)
            if dim:
                out += b' width="%d" height="%d"' % (
                    max(1, round(dim[0] * self.scale)),
                    max(1, round(dim[1] * self.scale)))
        align = RE_ALIGN.search(tag)
        if align:
            out += b' align="' + align.group(1) + b'"'
        return out + b'>'


# --- StarDict --------------------------------------------------------------

# StarDict orders words by g_ascii_strcasecmp(), falling back to strcmp().
ASCII_LOWER = bytes.maketrans(bytes(range(65, 91)), bytes(range(97, 123)))


def sortkey(word):
    return (word.translate(ASCII_LOWER), word)


def dictzip(src, dst, chunk=58315, level=9):
    """Write src (bytes) as a dictzip (.dz) file: gzip + random-access index."""
    chunks = [src[i:i + chunk] for i in range(0, len(src), chunk)] or [b'']
    if len(chunks) > 32760:
        raise SystemExit('too large to dictzip; use --no-dictzip')
    co = zlib.compressobj(level, zlib.DEFLATED, -zlib.MAX_WBITS)
    body, sizes = [], []
    for i, part in enumerate(chunks):
        blob = co.compress(part)
        blob += co.flush(zlib.Z_FINISH if i == len(chunks) - 1
                         else zlib.Z_FULL_FLUSH)
        body.append(blob)
        sizes.append(len(blob))
    extra = b'RA' + struct.pack('<HHHH', 6 + 2 * len(sizes), 1, chunk, len(sizes))
    extra += b''.join(struct.pack('<H', s) for s in sizes)
    name = os.path.basename(dst)[:-3].encode('utf-8', 'replace') + b'\0'
    with open(dst, 'wb') as f:
        f.write(b'\x1f\x8b\x08\x0c' + struct.pack('<IBB', int(time.time()), 2, 3))
        f.write(struct.pack('<H', len(extra)) + extra + name)
        for blob in body:
            f.write(blob)
        f.write(struct.pack('<II', zlib.crc32(src) & 0xFFFFFFFF, len(src) & 0xFFFFFFFF))


def write_ifo(path, fields):
    with open(path, 'w', encoding='utf-8') as f:
        f.write("StarDict's dict ifo file\nversion=3.0.0\n")
        for k, v in fields.items():
            if v not in (None, '', 0):
                f.write('%s=%s\n' % (k, str(v).replace('\n', ' ')))


# --- conversion ------------------------------------------------------------

def convert(args):
    log = lambda *a: print(*a, file=sys.stderr, flush=True)
    t0 = time.time()

    mobi = Mobi(args.mobi)
    if not mobi.orth_index or mobi.orth_index == 0xFFFFFFFF:
        raise SystemExit('%s has no ORTH index -- it is not a MOBI dictionary.\n'
                         'KF8/.azw3 dictionaries are not supported.' % args.mobi)
    log('reading %s (%s)' % (os.path.basename(args.mobi), mobi.fullname))
    text = mobi.text()
    log('  text: %.1f MB in %d records' % (len(text) / 1e6, mobi.ntext))

    raw_entries, _ = mobi.index(mobi.orth_index)
    log('  ORTH index: %d entries' % len(raw_entries))

    if args.dump_glyphs:
        return dump_glyphs(mobi, text, raw_entries, args.dump_glyphs, log)

    glyph_map = args.glyph_map
    if glyph_map is None:
        beside = os.path.splitext(args.mobi)[0] + '.glyphs'
        glyph_map = beside if os.path.exists(beside) else None
    glyphs = read_glyph_map(glyph_map) if glyph_map else None
    if glyphs:
        log('  glyph map: %s (%d glyphs, %d runs)'
            % (glyph_map, len(glyphs[0]), len(glyphs[1])))

    # Tags 1/2 are the entry's position and length in the text; an entry with
    # neither is an inflection or variant, and tag 22/25 is the ordinal of the
    # entry it belongs to.
    entries, aliases = [], []
    for ordinal, (label, tagmap) in enumerate(raw_entries):
        word = label.strip().encode('utf-8')
        if not word or b'\0' in word or len(word) > 255:
            continue
        if 1 in tagmap and 2 in tagmap:
            entries.append((word, tagmap[1][0], tagmap[2][0], ordinal))
        else:
            target = tagmap.get(22) or tagmap.get(25)
            if target:
                aliases.append((word, target[0]))
    if not entries:
        raise SystemExit('no entries found in the ORTH index')

    if args.front_matter:
        entries += front_matter(text, entries)

    # Resolve a filepos (which may point into the middle of an entry) to the
    # headword of the entry containing it.
    starts = sorted((off, word) for word, off, _, _ in entries)
    offsets = [s[0] for s in starts]
    spans = {off: off + ln for _, off, ln, _ in entries}

    def resolve(pos):
        i = bisect.bisect_right(offsets, pos) - 1
        if i < 0:
            return None
        off, word = starts[i]
        return word if pos < spans[off] else None

    entries.sort(key=lambda e: (sortkey(e[0]), e[3]))

    name = args.name or safe_name(mobi.meta(503) or mobi.fullname or 'dictionary')
    os.makedirs(args.out, exist_ok=True)
    base = os.path.join(args.out, name)
    clean = Cleaner(mobi, resolve, args.images, args.img_scale, glyphs)

    # Entries that share a headword (homographs) become one article.
    idx = bytearray()
    body = bytearray()
    ord2word, words, n = {}, 0, len(entries)
    i = 0
    while i < n:
        word = entries[i][0]
        j = i
        start = len(body)
        while j < n and entries[j][0] == word:
            _, off, ln, ordinal = entries[j]
            body += clean(text[off:off + ln])
            ord2word[ordinal] = words
            j += 1
        size = len(body) - start
        if size:
            idx += word + b'\0' + struct.pack('>II', start, size)
            words += 1
        else:
            del body[start:]
        i = j
        if words % 20000 == 0 and j < n:
            log('  %d/%d entries...' % (j, n))
    log('  %d articles (%d headwords merged), %.1f MB'
        % (words, n - words, len(body) / 1e6))
    if clean.nglyph:
        log('  %d images replaced with text, %d kept as images (%.1f%%)'
            % (clean.nglyph, clean.nimg,
               100.0 * clean.nglyph / (clean.nglyph + clean.nimg)))

    if len(body) > 0xFFFFFFFF:
        raise SystemExit('dictionary exceeds 4 GB; 32-bit offsets cannot index it')

    # Inflections/variants -> .syn, pointing at the article's index position.
    syn, seen = [], set()
    for word, target in aliases:
        hop, at = 0, target
        while at not in ord2word and hop < 4:       # alias chained to an alias
            nxt = raw_entries[at][1] if at < len(raw_entries) else None
            nxt = (nxt.get(22) or nxt.get(25)) if nxt else None
            if not nxt:
                break
            at, hop = nxt[0], hop + 1
        pos = ord2word.get(at)
        if pos is None or (word, pos) in seen:
            continue
        seen.add((word, pos))
        syn.append((word, pos))
    syn.sort(key=lambda s: sortkey(s[0]))

    with open(base + '.idx', 'wb') as f:
        f.write(idx)
    if args.dictzip:
        dictzip(bytes(body), base + '.dict.dz')
    else:
        with open(base + '.dict', 'wb') as f:
            f.write(body)
    if syn:
        with open(base + '.syn', 'wb') as f:
            f.write(b''.join(w + b'\0' + struct.pack('>I', p) for w, p in syn))

    nres = 0
    if clean.used:
        res = os.path.join(args.out, 'res')
        os.makedirs(res, exist_ok=True)
        for recindex, fname in clean.used.items():
            with open(os.path.join(res, fname.decode()), 'wb') as f:
                f.write(mobi.image(recindex))
        nres = len(clean.used)

    desc = mobi.meta(103) or ''
    if nres:
        desc += ('<br>' if desc else '') + \
            'Inline images are in the res/ directory next to this dictionary.'
    write_ifo(base + '.ifo', OrderedDict((
        ('bookname', mobi.meta(503) or mobi.fullname or name),
        ('wordcount', words),
        ('synwordcount', len(syn)),
        ('idxfilesize', len(idx)),
        ('sametypesequence', 'h'),
        ('author', mobi.meta(100)),
        ('description', desc),
        ('date', time.strftime('%Y.%m.%d')),
    )))

    log('wrote %s.{ifo,idx,%s%s} %s in %.0fs'
        % (base, 'dict.dz' if args.dictzip else 'dict', ',syn' if syn else '',
           '+ %d images' % nres if nres else '', time.time() - t0))
    log('  %d words, %d synonyms, idx %d bytes' % (words, len(syn), len(idx)))


def dump_glyphs(mobi, text, raw_entries, out, log, top=250):
    """Write the most-used inline images plus a starter glyph map."""
    import collections
    used = collections.Counter()
    for _, tagmap in raw_entries:
        if 1 in tagmap and 2 in tagmap:
            off, ln = tagmap[1][0], tagmap[2][0]
            used.update(int(i) for i in
                        re.findall(rb'\brecindex="(\d+)"', text[off:off + ln]))
    os.makedirs(out, exist_ok=True)
    total = sum(used.values()) or 1
    seen = 0
    with open(os.path.join(out, 'glyphs.tsv'), 'w', encoding='utf-8') as f:
        f.write('# Starter glyph map for %s.  Look at the images in this\n'
                '# directory, then uncomment a line and fill in the text it\n'
                '# should become.  A line with no replacement drops the image.\n'
                '# Format: <recindex>[,<recindex>...] TAB <replacement>\n'
                % os.path.basename(mobi.fullname or 'this dictionary'))
        for recindex, count in used.most_common(top):
            data = mobi.image(recindex)
            ext = ('gif' if data[:3] == b'GIF' else
                   'png' if data[:4] == b'\x89PNG' else
                   'jpg' if data[:2] == b'\xff\xd8' else 'bin')
            name = '%05d.%s' % (recindex, ext)
            with open(os.path.join(out, name), 'wb') as img:
                img.write(data)
            seen += count
            f.write('# %d\t\t%s, %d uses, %.2f%% cumulative\n'
                    % (recindex, name, count, 100.0 * seen / total))
    log('wrote %d images and glyphs.tsv to %s/ (%d distinct images in all)'
        % (min(top, len(used)), out, len(used)))


def front_matter(text, entries):
    """Pick up the guide references (pronunciation key etc.) around the A-Z."""
    lo = min(off for _, off, _, _ in entries)
    hi = max(off + ln for _, off, ln, _ in entries)
    refs = []
    for m in re.finditer(rb'<reference\b[^>]*>', text[:8192]):
        title = re.search(rb'title="([^"]*)"', m.group(0))
        pos = re.search(rb'filepos=0*(\d+)', m.group(0))
        if title and pos and title.group(1).strip():
            refs.append((int(pos.group(1)), title.group(1).strip()))
    bounds = sorted({p for p, _ in refs} | {lo, hi, len(text)})
    out = []
    for pos, title in refs:
        if lo <= pos < hi:          # a link into the dictionary itself
            continue
        end = bounds[bisect.bisect_right(bounds, pos)]
        if end > pos:
            out.append((title, pos, end - pos, -1 - len(out)))
    return out


def safe_name(s):
    s = re.sub(r'[^\w.\- ]+', '', s).strip() or 'dictionary'
    return re.sub(r'\s+', '_', s)


def main():
    p = argparse.ArgumentParser(
        description='Convert a MOBI dictionary directly to StarDict.')
    p.add_argument('mobi')
    p.add_argument('-o', '--out', default='.', metavar='DIR',
                   help='output directory (default: .)')
    p.add_argument('-n', '--name', help='base name for the output files')
    p.add_argument('-g', '--glyph-map', metavar='FILE',
                   help='image-to-text table (default: <mobi>.glyphs if present)')
    p.add_argument('--dump-glyphs', metavar='DIR',
                   help='write the most-used inline images and a starter '
                        'glyph map to DIR, then stop')
    p.add_argument('--images', choices=('hi', 'lo', 'none'), default='hi',
                   help='which inline images to extract (default: hi)')
    p.add_argument('--img-scale', type=float, default=1.0, metavar='F',
                   help='scale factor for inline image dimensions')
    p.add_argument('--no-dictzip', dest='dictzip', action='store_false',
                   help='write a plain .dict instead of .dict.dz')
    p.add_argument('--no-front-matter', dest='front_matter',
                   action='store_false',
                   help='skip the pronunciation/abbreviation keys')
    convert(p.parse_args())


if __name__ == '__main__':
    main()
