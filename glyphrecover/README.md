# glyphrecover

Recovers the text of the inline images in a Kindle dictionary.

NOAD-for-Kindle sets its pronunciation respelling, its section headings and all
of its etymology transliterations as **one tiny GIF per character** — 1.42M
image references in `american.mobi`. The common ~100 of those are mapped by
hand in `../american.glyphs`. This directory recovers the remaining ~4,600,
which are Greek, Latin, Old English, Sanskrit, Arabic and Hebrew etymons set as
whole-word images.

## Why not just run OCR

Off-the-shelf OCR reads the letters but silently drops the diacritics that
carry all the information. Scored on 24 hand-read words:

| | exact |
|---|---|
| Tesseract 5 (`script/Latin`, isolated glyphs, best of 12 configs) | 56% |
| jina-ocr-v1, one image per call | 5/24 |
| jina-ocr-v1, words laid out as a document page | 18/24 |
| jina-ocr-v1, + a prompt naming under-dots, `ł` and `ǣ` | **19/24** |
| dots.ocr (`dots-studio/dots.ocr`) | **18/24** |

Both VLMs fail on the *same* four: `ḥ`, `ṣ`, `ǣ`, `ł`. Prompt wording and image
scale move the needle by a point; framing the crops as a document page is what
took jina from 5/24 to 18/24 (its processor tiles at 640/1024, so an upscaled
lone crop gets stretched to fit an aspect-ratio grid). Neither is good enough
alone: 25% word error over 4,574 words is ~1,100 wrong etymons.

## How this works instead

The corpus is a single font at a single size, so the images can be solved
exactly rather than read:

1. **`pipeline.py`** — segment each word image into glyphs (connected
   components) and cluster identical bitmaps. 4,574 images collapse to ~1,300
   shape classes. Marks are assigned to the letter they sit on using
   *de-slanted* coordinates (slope 0.30): in italic, the dot of an `i` sits
   over the *next* letter, and a macron over a short letter still sits below
   the top of a tall one in the same word.
2. **`decomp.py`** — split each class into base letter + diacritical marks and
   cluster those separately. ~690 base shapes, ~87 marks.
3. **`runall.py`** — run jina-ocr-v1 and dots.ocr over all the word images, 24
   to a synthetic page.
4. **`compose.py`** — label base clusters and mark clusters by *voting* the OCR
   readings across every word a class appears in, alternating between the two
   until both are stable, then compose `base + mark` into a character. Random
   OCR slips cancel out; the systematic ones don't, which is the point of the
   next step.
5. **`flag.py`** — the systematic failures are detectable: if two base clusters
   vote the same letter but one is visibly taller, the taller one is carrying a
   mark fused to the letter that OCR dropped. Those, plus the below-marks OCR
   cannot see at all, are the only things read by eye (`marks_below.tsv` — three
   kinds: dot-below, cedilla, underline).
6. **`emit_map.py`** — reconstruct every word and append `recindex<TAB>text`
   lines to the glyph map.

`review.py`, `sheets.py` and `marksheet.py` render contact sheets for the
by-eye steps; `evalfinal.py` scores composed labels against `eval120.tsv`, 120
hand-read classes covering 87% of all glyph instances.

## Running it

Needs calibre (for the MOBI parser, as `../mobidict2stardict.py` does) and, for
the OCR stage, two venvs — jina-ocr-v1 wants current transformers, dots.ocr
wants 4.51.3 and a directory name without a period.

```sh
calibre-debug -e extract.py -- ../american.mobi ../american.glyphs  # -> glyphs/, glyphs.json
python pipeline.py && python decomp.py                              # -> pipe.pkl, decomp.pkl
python runall.py jina    # in a venv with current transformers
python runall.py dots    # in a venv with transformers==4.51.3
python compose.py && python flag.py && python emit_map.py           # -> glyphmap_add.tsv
cat ../american.glyphs glyphmap_add.tsv > american_full.glyphs
```
