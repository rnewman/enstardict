# Recover the text of a Kindle dictionary's inline art and rebuild the
# StarDict output with it.  See README.md for what each stage does.
#
#   make venvs     one-off: build the two OCR environments
#   make ocr       run both OCR models over the extracted art (slow, ~35 min)
#   make           label the glyphs, emit the map, rebuild the dictionary
#   make report    accuracy against the hand-read classes, plus what is left
#   make review    contact sheets for anything the pipeline could not settle
#
# Stages are ordinary file rules, so re-running only redoes what is stale.

MOBI    ?= ../american.mobi
GLYPHS  ?= base.glyphs      # the hand-made core map; the full one is generated
OUT     ?= ../noad-stardict
CALIBRE ?= $(firstword $(wildcard /Applications/calibre.app/Contents/MacOS/calibre-debug) \
                       $(shell command -v calibre-debug 2>/dev/null))
PY      ?= python3
JINA_PY ?= ./ocrenv/bin/python
DOTS_PY ?= ./ocrenv2/bin/python

# Hand-read input: the below-marks OCR is blind to, plus any manual overrides.
HAND := marks_below.tsv $(wildcard overrides.tsv)

.PHONY: all ocr report review venvs clean distclean
all: dict

# 1. pull every still-unmapped image out of the .mobi (needs calibre's parser)
glyphs.json: extract.py $(MOBI) $(GLYPHS)
	$(CALIBRE) -e extract.py -- $(MOBI) $(GLYPHS)

# 2. segment into glyphs and cluster identical shapes
pipe.pkl: pipeline.py glyphs.json
	$(PY) pipeline.py

# 3. split each shape into base letter + marks, cluster those separately
decomp.pkl: decomp.py pipe.pkl
	$(PY) decomp.py

# 4. OCR the art with two independent models, 24 words to a synthetic page.
#    Separate venvs: jina wants current transformers, dots.ocr wants 4.51.3.
jina_all.json: runall.py glyphs.json
	$(JINA_PY) runall.py jina
dots_all.json: runall.py glyphs.json
	$(DOTS_PY) runall.py dots
ocr: jina_all.json dots_all.json

# 5. vote the OCR readings into per-cluster labels and compose characters
votes.json: compose.py pipe.pkl decomp.pkl jina_all.json dots_all.json $(HAND)
	$(PY) compose.py

# 6. reconstruct every image and append it to the glyph map
glyphmap_add.tsv: emit_map.py votes.json $(wildcard overrides.tsv)
	$(PY) emit_map.py

# The shipped map is base + recovered, written where the converter finds it
# automatically (it looks for <mobi>.glyphs beside the .mobi).
../american.glyphs: $(GLYPHS) glyphmap_add.tsv
	cat $(GLYPHS) glyphmap_add.tsv > $@

# 7. rebuild the dictionary against the completed map
.PHONY: dict
dict: ../american.glyphs $(MOBI) ../mobidict2stardict.py
	$(PY) ../mobidict2stardict.py $(MOBI) -o $(OUT)

report: votes.json
	@$(PY) evalfinal.py
	@echo
	@$(PY) flag.py

# Contact sheets for the by-eye steps.  flag.py writes flagged.txt; review.py
# reads it when given "-" as its class list.
review: votes.json
	$(PY) flag.py >/dev/null
	$(PY) review.py - review.png
	$(PY) marksheet.py below
	$(PY) marksheet.py above

venvs:
	$(PY) -m venv --system-site-packages ocrenv
	./ocrenv/bin/pip -q install transformers accelerate 'huggingface_hub[hf_transfer]'
	$(PY) -m venv --system-site-packages ocrenv2
	./ocrenv2/bin/pip -q install 'transformers==4.51.3' accelerate qwen-vl-utils
	@echo 'Now fetch the models, e.g.:'
	@echo '  ./ocrenv/bin/python -c "from huggingface_hub import snapshot_download as d; d(\"jinaai/jina-ocr-v1\", local_dir=\"jina-ocr-v1\")"'
	@echo '  ./ocrenv/bin/python -c "from huggingface_hub import snapshot_download as d; d(\"dots-studio/dots.ocr\", local_dir=\"dots.ocr\")" && ln -sfn dots.ocr DotsOCR'

clean:
	rm -f pipe.pkl decomp.pkl votes.json recon.json glyphmap_add.tsv \
	      flagged.txt *.png
distclean: clean
	rm -rf glyphs glyphs.json jina_all.json dots_all.json
