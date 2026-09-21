"""OCR every unmapped glyph image, a page of 24 words at a time."""
import json, torch, time, os, sys
from PIL import Image
MODEL = sys.argv[1]                       # 'jina' | 'dots'
OUT = {'jina':'jina_all.json','dots':'dots_all.json'}[MODEL]
TARGETED = ("Pay specific attention to under-dot diacritics (ḥ, ṣ, ṭ, ḍ, ṇ, ẓ, ḳ, ṛ) and to the "
            "characters ł and ǣ, which appear in this text.")
JINA_PROMPT = ("Transcribe the provided document image into a clean Markdown format, preserving "
               "the natural reading order. " + TARGETED)
meta = json.load(open('glyphs.json'))
keys = sorted(meta, key=lambda k: int(k))
dev = torch.device('mps')
from transformers import AutoModelForCausalLM, AutoProcessor
if MODEL == 'jina':
    proc = AutoProcessor.from_pretrained('jina-ocr-v1', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained('jina-ocr-v1', dtype=torch.bfloat16,
                                                 trust_remote_code=True).to(dev).eval()
else:
    proc = AutoProcessor.from_pretrained('DotsOCR', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained('DotsOCR', torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True,
                                                 attn_implementation='sdpa').to(dev).eval()
N, MARGIN, W, MULT = 24, 60, 1024, 1.5
def build(chunk):
    rows = [Image.open(meta[k]['file']).convert('L') for k in chunk]
    rows = [r.resize((max(1, round(r.width*MULT)), max(1, round(r.height*MULT))),
                     Image.Resampling.LANCZOS) for r in rows]
    lead = int(max(r.height for r in rows) * 1.9)
    pg = Image.new('L', (W, MARGIN*2 + lead*len(rows)), 255); y = MARGIN
    for im in rows: pg.paste(im, (MARGIN, y)); y += lead
    return pg.convert('RGB')
def read(img):
    if MODEL == 'jina':
        inp = proc.prepare_ocr_inputs(img, prompt=JINA_PROMPT, device=dev)
        with torch.no_grad():
            o = model.generate(**inp, max_new_tokens=420, do_sample=False, no_repeat_ngram_size=0)
        return proc.decode_ocr(o, inp['input_ids'])
    msgs = [{"role":"user","content":[{"type":"image","image":img},
                                      {"type":"text","text":"Extract the text content from this image. "+TARGETED}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=[img], return_tensors="pt").to(dev)
    inp.pop("mm_token_type_ids", None)
    with torch.no_grad():
        o = model.generate(**inp, max_new_tokens=500, do_sample=False, no_repeat_ngram_size=0)
    return proc.decode(o[0][inp['input_ids'].shape[1]:], skip_special_tokens=True)
out = json.load(open(OUT)) if os.path.exists(OUT) else {}
chunks = [keys[i:i+N] for i in range(0, len(keys), N)]
t0 = time.time()
for ci, ch in enumerate(chunks):
    if all(k in out for k in ch): continue
    lines = [l.strip() for l in read(build(ch)).splitlines() if l.strip()]
    for i, k in enumerate(ch): out[k] = lines[i] if i < len(lines) else ''
    if ci % 20 == 0:
        json.dump(out, open(OUT, 'w'))
        el = time.time()-t0
        print('%s page %d/%d  eta %.0f min' % (MODEL, ci+1, len(chunks), (el/(ci+1))*(len(chunks)-ci-1)/60), flush=True)
json.dump(out, open(OUT, 'w'))
print('%s done: %d words in %.1f min' % (MODEL, len(out), (time.time()-t0)/60))
