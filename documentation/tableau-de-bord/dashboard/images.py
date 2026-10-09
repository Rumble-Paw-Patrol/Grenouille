"""Écrit page/bandeau.css (spectrogramme et planche des 24 images de la grenouille 8-bit) et
page/poste.css (capture du poste d'annotation), images en data: URL comme le veut le dashboard.

    python documentation/tableau-de-bord/dashboard/images.py
"""
import base64, io, json, re
from pathlib import Path
from PIL import Image
ICI = Path(__file__).resolve().parents[1]
PAGE = Path(__file__).resolve().parent / "page"
G = json.loads((ICI / "grenouille.json").read_text())
W, H = G["largeur"], G["hauteur"]
cles = sorted(G["images"])
def rgba(c):
    c = c.lstrip("#")
    if len(c) == 6: c += "ff"
    return tuple(int(c[i:i+2], 16) for i in (0, 2, 4, 6))
planche = Image.new("RGBA", (W * len(cles), H), (0, 0, 0, 0))
px = planche.load()
for i, k in enumerate(cles):
    for y, rang in enumerate(G["images"][k]):
        x = 0
        for c, n in re.findall(r"(\D)(\d+)", rang):
            n = int(n)
            if c != ".":
                col = rgba(G["palette"][c])
                for dx in range(n): px[i * W + x + dx, y] = col
            x += n
b = io.BytesIO(); planche.save(b, "PNG", optimize=True); sprite = base64.b64encode(b.getvalue()).decode()
spectro = base64.b64encode((ICI / "spectrogramme.jpg").read_bytes()).decode()
im = Image.open(ICI / "poste-annotation.png").convert("RGB"); im = im.resize((1200, 750), Image.LANCZOS)
b = io.BytesIO(); im.save(b, "JPEG", quality=70, optimize=True); poste = base64.b64encode(b.getvalue()).decode()
(PAGE / "bandeau.css").write_text(
    f".gb-spectro-img{{background-image:url(data:image/jpeg;base64,{spectro})}}\n"
    f".gb-sprite{{background-image:url(data:image/png;base64,{sprite})}}\n")
(PAGE / "poste.css").write_text(f".gb-poste-img{{background-image:url(data:image/jpeg;base64,{poste})}}\n")
print(f"bandeau.css, poste.css : {len(sprite) + len(spectro)} et {len(poste)} octets d'images")
