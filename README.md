# dictionaries

Tools for turning Kindle/Mobipocket dictionaries into StarDict, and for
recovering the text of the inline artwork they are set in.

Built against *The New Oxford American Dictionary* (`american.mobi`), but the
converter is general; the glyph map is per-dictionary.

## mobidict2stardict.py

Converts a `.mobi` dictionary straight to StarDict. No HTML export step: it
reads the PDB record table, decompresses the text records, and walks the
orthographic (ORTH) index, using calibre's MOBI index parser for the fiddly
INDX/TAGX/ORDT decoding. Run it with a normal `python3` — it re-execs itself
under calibre's interpreter automatically, since that is where the parser
lives.

```sh
./mobidict2stardict.py american.mobi -o noad-stardict/
```

| option | |
|---|---|
| `-o, --out DIR` | output directory |
| `-n, --name NAME` | base name for the output files |
| `-g, --glyph-map FILE` | image-to-text table; defaults to `<mobi>.glyphs` if present |
| `--dump-glyphs DIR` | write the most-used inline images and a starter glyph map, then stop |
| `--images hi\|lo\|none` | which inline art to extract for whatever is left unmapped |
| `--img-scale F` | scale factor for inline image dimensions |
| `--no-dictzip` | write a plain `.dict` instead of `.dict.dz` |
| `--no-front-matter` | skip the pronunciation and abbreviation keys |

Output is StarDict 3.0.0: 32-bit index offsets (no `idxoffsetbits=64`), an
uncompressed `.idx`, and a `.dict.dz` written by an in-process dictzip
implementation, so the `dictzip` binary is not needed. Inflections and variants
recorded in the index become a `.syn` file, `filepos` cross-references become
`bword://` links, and anything still set as an image is extracted to `res/`
beside the dictionary, where GoldenDict looks for it.

One thing worth knowing: calibre's `read_index()` truncates ORDT2 character
tables to 8 bits, which turns every space in a headword into `?` — `-- a pop`
comes out as `-?-?a?pop`. `Mobi._ordt` rebuilds that table as UTF-16BE and
drives calibre's record parser directly.

On NOAD: 86,080 articles, 43,352 synonyms, ~13 MB `.dict.dz`, about 5 seconds.

## american.glyphs

NOAD-for-Kindle sets its pronunciation respelling, its section headings and its
sense markers as **one tiny GIF per character** — 1.42M image references.
`american.glyphs` maps them back to text, so `|ˈzīˌmərjē|` is real, searchable,
selectable characters rather than 11 images:

```
<recindex>[,<recindex>...] TAB <replacement> [TAB <comment>]
```

A comma-separated list matches a run of consecutive images, which is how the
digraphs work (`97,86` → `o͞o`, the macron spanning both letters of *boot*). A
line with no replacement drops the image — that is how the 1px spacer rules
disappear. `\s` in a replacement means a literal space.

The common ~100 glyphs were derived by decoding known pronunciations until
every transcription matched NOAD (*lamb*, *father*, *though*, *ship*,
*measure*, *loch*, *sing*, *boot*, *book*, *water*, *machine*). The rest — the
etymology transliterations — come from `glyphrecover/`.

For a dictionary with no map yet, `--dump-glyphs DIR` writes the most-used
images alongside a commented starter table to fill in.

## glyphrecover/

Recovers the ~4,600 remaining images: Greek, Latin, Old English, Sanskrit,
Arabic and Hebrew etymons set as whole-word art.

Off-the-shelf OCR reads the letters but silently drops the diacritics that
carry the information. Scored on 24 hand-read words: Tesseract 5 with
`script/Latin` manages 56% on isolated glyphs; jina-ocr-v1 gets 19/24 and
dots.ocr 18/24 — and both fail on the *same four*, `ḥ`, `ṣ`, `ǣ`, `ł`. At 25%
word error over 4,574 words that is ~1,100 wrong etymons, so neither is usable
on its own.

Instead the corpus is solved rather than read. It is a single font at a single
size, so the images are segmented into glyphs and clustered by shape — 4,574
words collapse to ~1,300 shapes, which decompose further into ~690 base letters
and ~87 marks. Each shape appears in many words, so *voting* both models'
readings across every word a shape occurs in cancels the random slips. The
systematic ones are then detectable: if two base clusters vote the same letter
but one is visibly taller, the taller one is carrying a mark fused to the
letter that OCR dropped. Only what OCR is structurally blind to gets read by
eye — the below-marks, of which there are three.

See `glyphrecover/README.md` for the details and the measurements.

### Running it

The stages are a build, not a sequence to type. Each is a file rule with real
prerequisites, so re-running only redoes what is stale.

```sh
cd glyphrecover
make venvs     # one-off: the two OCR environments, then fetch the models
make ocr       # run both models over the extracted art (~35 min on an M-series Mac)
make           # label the glyphs, emit the map, rebuild the dictionary
make report    # accuracy against the hand-read classes, and what is left
make review    # contact sheets for anything the pipeline could not settle
```

| target | from | does |
|---|---|---|
| `glyphs.json` | `extract.py`, the `.mobi`, the existing map | pull out every still-unmapped image (needs calibre) |
| `pipe.pkl` | `pipeline.py` | segment into glyphs, cluster identical shapes |
| `decomp.pkl` | `decomp.py` | split each shape into base letter + marks |
| `jina_all.json` | `runall.py jina` | OCR the art, 24 words to a synthetic page |
| `dots_all.json` | `runall.py dots` | the same with the second model |
| `votes.json` | `compose.py`, `marks_below.tsv`, `overrides.tsv` | vote the readings into labels, compose characters |
| `glyphmap_add.tsv` | `emit_map.py` | reconstruct every image as `recindex TAB text` |
| `american_full.glyphs` | | base map + additions |
| `dict` | `mobidict2stardict.py` | rebuild against the completed map |

The two OCR models keep separate virtualenvs — jina-ocr-v1 wants current
transformers, dots.ocr wants 4.51.3 and a directory name without a period —
and that is encoded in the rules rather than remembered.

`overrides.tsv` (`class TAB text`, same format as the glyph map) is picked up
automatically when present, so corrections made after looking at `make review`
re-enter the build instead of being applied by hand.

Variables: `MOBI`, `GLYPHS`, `OUT`, `PY`, `CALIBRE`, `JINA_PY`, `DOTS_PY`.

## Requirements

- calibre, for its MOBI parser (`mobidict2stardict.py` finds `calibre-debug` on
  `PATH` or in `/Applications`)
- Python 3 with numpy, scipy and Pillow, for `glyphrecover/`
- torch with MPS or CUDA, plus the two OCR models, only for `make ocr`

Nothing here needs the `dictzip` binary, `sdcv`, or GoldenDict, though a
StarDict reader is handy for looking at the result.
