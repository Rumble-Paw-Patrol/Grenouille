# ruff: noqa: E501
"""Estime, police réelle à l'appui, si chaque zone de texte du pptx tient dans son cadre."""
import sys

from PIL import ImageFont
from pptx import Presentation
from pptx.util import Emu

FONTS = {("Calibri", False): "calibri.ttf", ("Calibri", True): "calibrib.ttf", ("Cambria", True): "cambriab.ttf", ("Cambria", False): "cambria.ttc", ("Consolas", False): "consola.ttf", ("Consolas", True): "consolab.ttf"}
cache = {}
def font(name, bold, size):
    k = (name, bold, size)
    if k not in cache:
        cache[k] = ImageFont.truetype("C:/Windows/Fonts/" + FONTS.get((name, bold), "calibri.ttf"), int(size * 4))
    return cache[k]
prs = Presentation(sys.argv[1])
for n, s in enumerate(prs.slides, 1):
    for sh in s.shapes:
        if not sh.has_text_frame or not sh.text_frame.text.strip() or not sh.text_frame.word_wrap:
            continue
        W = Emu(sh.width).inches * 72 * 4
        h = 0
        for p in sh.text_frame.paragraphs:
            runs = [r for r in p.runs if r.text]
            if not runs:
                continue
            size = max(r.font.size.pt for r in runs)
            words, line, nl = [], 0, 1
            for r in runs:
                f = font(r.font.name, bool(r.font.bold), r.font.size.pt)
                for w in r.text.split(" "):
                    wl = f.getlength(w + " ")
                    if line + wl > W and line > 0:
                        nl += 1
                        line = wl
                    else:
                        line += wl
            h += nl * size * 1.2 + (p.space_after.pt if p.space_after else 0)
        h_in = h / 72
        bas = Emu(sh.top).inches + h_in
        if h_in > Emu(sh.height).inches + 0.05 or bas > 7.35:
            print(f"diapo {n:2d} {sh.name:10s} besoin {h_in:.2f} in, cadre {Emu(sh.height).inches:.2f}, bas {bas:.2f} : {sh.text_frame.text[:60]!r}")
