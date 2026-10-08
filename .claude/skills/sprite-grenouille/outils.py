"""Outils du skill sprite-grenouille (lancer avec « uv run python <ce fichier> … »).

    grille PHOTO SORTIE.png [x0,y0,x1,y1] [pas]
        Grille numérotée sur la photo de profil (une case = une unité du repère, 48 de large)
        pour y relever la silhouette : museau, œil, dos, cuisse, pattes.
    apercu SPRITE.json SORTIE.png CLE[,CLE…] [zoom] [PHOTO]
        Images du sprite côte à côte, la photo de référence à droite si donnée.
    zoom SPRITE.json SORTIE.png CLE x0,y0,x1,y1
        Une zone agrandie, pixel par pixel, pour juger un détail (épaule, œil, doigts).
    planche SPRITE.json SORTIE.html NOM "TEXTE"
        La planche de relecture (planche.html) avec le sprite intégré.
    controle SPRITE.json
        Groupes de pixels qui ne tiennent pas au reste (contour exclu, voisins par un côté),
        image par image : un doigt, un orteil ou un talon détaché apparaît ici. Seul le bras
        de l'autre côté, cerné partout, peut légitimement faire un groupe à part.

Les photos de référence restent dans le bac à sable : jamais dans le dépôt ni dans une page.
"""

import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ICI = Path(__file__).resolve().parent


def rangees(sprite, cle):
    return [
        "".join(c * int(n) for c, n in re.findall(r"(\D)(\d+)", r)) for r in sprite["images"][cle]
    ]


def grille(photo, sortie, cadre=None, pas=None):
    im = Image.open(photo).convert("RGB")
    if cadre:
        im = im.crop(tuple(int(v) for v in cadre.split(",")))
    pas = float(pas) if pas else im.width / 48
    d = ImageDraw.Draw(im)
    for k in range(int(im.width / pas) + 1):
        x = k * pas
        d.line([(x, 0), (x, im.height)], fill=(80, 200, 255) if k % 5 == 0 else (40, 90, 110))
        if k % 5 == 0:
            d.text((x + 2, 2), str(k), fill=(255, 255, 0))
    for k in range(int(im.height / pas) + 1):
        y = k * pas
        d.line([(0, y), (im.width, y)], fill=(80, 200, 255) if k % 5 == 0 else (40, 90, 110))
        if k % 5 == 0:
            d.text((2, y + 2), str(k), fill=(255, 255, 0))
    im.resize((im.width * 2, im.height * 2)).save(sortie)


def apercu(fichier, sortie, cles, zoom="5", photo=None):
    s, z, imgs = json.load(open(fichier)), int(zoom), []
    for cle in cles.split(","):
        g = rangees(s, cle)
        im = Image.new("RGB", (len(g[0]) * z, len(g) * z), "#0A0F0E")
        for y, r in enumerate(g):
            for x, c in enumerate(r):
                if c != ".":
                    im.paste(Image.new("RGB", (z, z), s["palette"][c]), (x * z, y * z))
        imgs.append(im)
    if photo:
        ref = Image.open(photo).convert("RGB")
        imgs.append(ref.resize((imgs[0].width, round(imgs[0].width * ref.height / ref.width))))
    tot = Image.new("RGB", (sum(i.width + 10 for i in imgs), max(i.height for i in imgs)), "#000")
    x = 0
    for i in imgs:
        tot.paste(i, (x, 0))
        x += i.width + 10
    tot.save(sortie)


def zoom(fichier, sortie, cle, cadre):
    s = json.load(open(fichier))
    g, (x0, y0, x1, y1), z = rangees(s, cle), map(int, cadre.split(",")), 12
    im = Image.new("RGB", ((x1 - x0) * z, (y1 - y0) * z), "#0A0F0E")
    for y in range(y0, y1):
        for x in range(x0, x1):
            if g[y][x] != ".":
                im.paste(
                    Image.new("RGB", (z - 1, z - 1), s["palette"][g[y][x]]),
                    ((x - x0) * z, (y - y0) * z),
                )
    im.save(sortie)


def planche(fichier, sortie, nom, texte):
    page = (ICI / "planche.html").read_text(encoding="utf-8")
    page = page.replace("/*NOM*/", nom).replace("/*TEXTE*/", texte)
    page = page.replace("/*SPRITE*/null", Path(fichier).read_text(encoding="utf-8"), 1)
    Path(sortie).write_text(page, encoding="utf-8")


def controle(fichier):
    s = json.load(open(fichier))
    for cle in sorted(s["images"]):
        g = rangees(s, cle)
        plein = {(j, i) for j, r in enumerate(g) for i, c in enumerate(r) if c not in ".K"}
        groupes, vus = [], set()
        for depart in sorted(plein):
            if depart in vus:
                continue
            pile, groupe = [depart], []
            vus.add(depart)
            while pile:
                j, i = pile.pop()
                groupe.append((j, i))
                for v in ((j + 1, i), (j - 1, i), (j, i + 1), (j, i - 1)):
                    if v in plein and v not in vus:
                        vus.add(v)
                        pile.append(v)
            groupes.append(groupe)
        groupes.sort(key=len, reverse=True)
        autres = ", ".join(f"{len(g)} px en {min(g)}" for g in groupes[1:]) or "aucun"
        print(f"{cle} : corps {len(groupes[0])} px ; autres groupes (rangée, colonne) : {autres}")


if __name__ == "__main__":
    {"grille": grille, "apercu": apercu, "zoom": zoom, "planche": planche, "controle": controle}[
        sys.argv[1]
    ](*sys.argv[2:])
