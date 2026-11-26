# ruff: noqa: E501
"""Version navigateur d'un pptx de la prez n° 7 : un seul fichier HTML, sans PowerPoint ni réseau.

    uv run --with python-pptx --with pillow python documentation/prez/scripts/presentation_7/exporter_html.py [entrée.pptx] [sortie.html] [--video]

Par défaut : `documentation/prez/presentation_Leonard_PNA.pptx` → `….html`. Lit les formes du pptx
(rectangles, ellipses, textes, images, GIF, flèches, tracés libres, son) et rejoue ses transitions :
la Morphose (les formes de même nom « !!… » glissent, grandissent et changent de couleur ; les autres
apparaissent ou s'effacent en fondu) et la poussée latérale. Images et son sont intégrés (data URI) :
le fichier se copie seul sur une clé USB et s'ouvre dans Edge, Chrome ou Firefox.

Clavier : → espace Entrée clic (suivante), ← retour arrière clic droit (précédente), Début / Fin,
un numéro puis Entrée (aller à la diapo), F (plein écran), N (notes de l'orateur). Un clic sur le
spectrogramme de la diapo 3 lance le son. `?print` : toutes les diapos, une par page (impression en
PDF) ; les GIF y montrent leur dernière image. `--video` : ajoute les images des GIF une à une, pour
que exporter_video.py pilote le temps.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
from pathlib import Path

from lxml import etree
from PIL import Image
from pptx import Presentation

DEPOT = Path(__file__).resolve().parents[4]
args = [a for a in sys.argv[1:] if not a.startswith("--")]
VIDEO = "--video" in sys.argv
ENTREE = Path(args[0]) if args else DEPOT / "documentation" / "prez" / "presentation_Leonard_PNA.pptx"
SORTIE = Path(args[1]) if len(args) > 1 else ENTREE.with_suffix(".html")
NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "p14": "http://schemas.microsoft.com/office/powerpoint/2010/main",
    "p159": "http://schemas.microsoft.com/office/powerpoint/2015/09/main",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
}
EMU_PX = 914400 / 100  # 100 px par pouce : la scène fait 1 333 × 750 px
R = "{" + NS["r"] + "}"
ASSETS: dict[str, str] = {}
GIFS: dict[str, dict] = {}


def q(el, chemin):
    return el.find(chemin, NS)


def asset(blob: bytes, ctype: str) -> str:
    h = hashlib.sha1(blob).hexdigest()[:12]
    if h not in ASSETS:
        ASSETS[h] = f"data:{ctype};base64,{base64.b64encode(blob).decode()}"
        if ctype == "image/gif":
            im = Image.open(io.BytesIO(blob))
            images, durees = [], []
            for k in range(getattr(im, "n_frames", 1)):
                im.seek(k)
                durees.append(im.info.get("duration", 100))
                if VIDEO or k == im.n_frames - 1:
                    b = io.BytesIO()
                    im.convert("RGB").save(b, "PNG", optimize=True)
                    images.append(f"data:image/png;base64,{base64.b64encode(b.getvalue()).decode()}")
            GIFS[h] = {"fin": images[-1], "images": images if VIDEO else [], "durees": durees}
    return h


def couleur(el):
    """srgbClr d'un solidFill, None si noFill ou absent."""
    if el is None or q(el, "a:noFill") is not None:
        return None
    c = q(el, "a:solidFill/a:srgbClr")
    return c.get("val") if c is not None else None


def trait(sppr):
    ln = q(sppr, "a:ln") if sppr is not None else None
    if ln is None or q(ln, "a:noFill") is not None:
        return None
    c = q(ln, "a:solidFill/a:srgbClr")
    if c is None:
        return None
    d = q(ln, "a:prstDash")
    return {
        "c": c.get("val"), "w": round(int(ln.get("w", "9525")) / 12700 * 100 / 72, 2), "d": d is not None and d.get("val") != "solid",
        "fin": q(ln, "a:tailEnd") is not None and q(ln, "a:tailEnd").get("type", "none") != "none",
        "deb": q(ln, "a:headEnd") is not None and q(ln, "a:headEnd").get("type", "none") != "none",
    }


def texte(txbody):
    if txbody is None:
        return None
    bp = q(txbody, "a:bodyPr")
    ps = []
    for p in txbody.findall("a:p", NS):
        ppr = q(p, "a:pPr")
        sa = q(p, "a:pPr/a:spcAft/a:spcPts")
        runs = []
        for r in p.findall("a:r", NS):
            rpr = q(r, "a:rPr")
            lat = q(rpr, "a:latin") if rpr is not None else None
            runs.append({
                "t": q(r, "a:t").text or "", "f": lat.get("typeface") if lat is not None else "Calibri",
                "s": round(int(rpr.get("sz", "1800")) / 100 * 100 / 72, 2) if rpr is not None else 25,
                "b": rpr is not None and rpr.get("b") == "1", "i": rpr is not None and rpr.get("i") == "1",
                "c": couleur(rpr) or "000000",
            })
        if runs:
            ps.append({"al": (ppr.get("algn") if ppr is not None else None) or "l", "sa": round(int(sa.get("val")) / 100 * 100 / 72, 2) if sa is not None else 0, "rs": runs})
    if not ps:
        return None
    return {"a": (bp.get("anchor") if bp is not None else None) or "t", "wrap": bp is None or bp.get("wrap") != "none", "ps": ps}


def cadre(sppr):
    x = q(sppr, "a:xfrm")
    off, ext = q(x, "a:off"), q(x, "a:ext")
    return {
        "x": round(int(off.get("x")) / EMU_PX, 2), "y": round(int(off.get("y")) / EMU_PX, 2),
        "w": round(int(ext.get("cx")) / EMU_PX, 2), "h": round(int(ext.get("cy")) / EMU_PX, 2),
        "r": round(int(x.get("rot", "0")) / 60000, 2), "fx": x.get("flipH") == "1", "fy": x.get("flipV") == "1",
    }


def forme(el, part):
    tag = etree.QName(el).localname
    nom = q(el, ".//p:cNvPr").get("name")
    sppr = q(el, "p:spPr")
    if tag == "grpSp":
        return [f for e in el if etree.QName(e).localname in ("sp", "pic", "cxnSp", "grpSp") for f in forme(e, part)]
    d = {"n": nom, **cadre(sppr)}
    if tag == "cxnSp":
        d.update(k="line", l=trait(sppr))
        return [d]
    if tag == "pic":
        blip = q(el, "p:blipFill/a:blip")
        img = part.related_part(blip.get(R + "embed"))
        d.update(k="pic", src=asset(img.blob, img.content_type))
        media = q(el, "p:nvPicPr/p:nvPr/a:videoFile")
        if media is None:
            media = q(el, "p:nvPicPr/p:nvPr/a:audioFile")
        if media is not None:
            son = part.rels[media.get(R + "link")].target_part
            d.update(au=asset(son.blob, son.content_type))
        return [d]
    geo = q(sppr, "a:prstGeom")
    cust = q(sppr, "a:custGeom")
    d.update(k="box", f=couleur(sppr), l=trait(sppr), t=texte(q(el, "p:txBody")))
    if cust is not None:
        path = q(cust, "a:pathLst/a:path")
        pw, ph = int(path.get("w")), int(path.get("h"))
        cmds = []
        for c in path:
            n = etree.QName(c).localname
            pt = q(c, "a:pt")
            if n in ("moveTo", "lnTo"):
                cmds.append(f"{'M' if n == 'moveTo' else 'L'}{int(pt.get('x')) / pw * d['w']:.2f} {int(pt.get('y')) / ph * d['h']:.2f}")
            elif n == "close":
                cmds.append("Z")
        d.update(k="path", p=" ".join(cmds))
        return [d]
    prst = geo.get("prst") if geo is not None else "rect"
    gd = q(geo, "a:avLst/a:gd") if geo is not None else None
    adj = int(gd.get("fmla").split()[1]) / 100000 if gd is not None else 0.16667
    d["g"] = {"ellipse": "ellipse", "roundRect": "round", "snip1Rect": "snip"}.get(prst, "rect")
    if d["g"] in ("round", "snip"):
        d["rad"] = round(adj * min(d["w"], d["h"]), 2)
    return [d]


def transition(sld):
    tr = sld.find("mc:AlternateContent/mc:Choice/p:transition", NS)
    if tr is None:
        tr = q(sld, "p:transition")
    if tr is None:
        return None
    if q(tr, "p159:morph") is not None:
        return {"type": "morph", "dur": int(tr.get("{" + NS["p14"] + "}dur", "1000"))}
    if q(tr, "p:push") is not None:
        return {"type": "push", "dur": 1000}
    return {"type": "fade", "dur": 700}


prs = Presentation(ENTREE)
diapos = []
for s in prs.slides:
    sld = s._element
    fond = q(sld, "p:cSld/p:bg/p:bgPr")
    formes = [f for el in q(sld, "p:cSld/p:spTree") if etree.QName(el).localname in ("sp", "pic", "cxnSp", "grpSp") for f in forme(el, s.part)]
    note = s.notes_slide.notes_text_frame.text if s.has_notes_slide else ""
    diapos.append({"bg": couleur(fond) or "FFFFFF", "tr": transition(sld), "notes": note, "f": formes})
D = {"w": prs.slide_width / EMU_PX, "h": prs.slide_height / EMU_PX, "assets": ASSETS, "gifs": GIFS, "diapos": diapos}
modele = (Path(__file__).with_name("lecteur.html")).read_text(encoding="utf-8")
html = modele.replace("/*DONNEES*/null", json.dumps(D, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
html = html.replace("<title>Présentation</title>", f"<title>{ENTREE.stem.replace('_', ' ')}</title>")
SORTIE.write_text(html, encoding="utf-8")
print("écrit", SORTIE, f"{SORTIE.stat().st_size / 1e6:.1f} Mo", len(diapos), "diapos,", sum(len(d["f"]) for d in diapos), "formes,", len(ASSETS), "médias")
