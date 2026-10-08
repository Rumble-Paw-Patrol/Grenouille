"""Outils du skill sprite-grenouille (lancer avec « python3 <ce fichier> … »).

Un SPRITE est le JSON d'une grenouille (palette, largeur, hauteur, images codées par plages).
Une COMPOSITION en réunit plusieurs sur une même toile (voir « composer ») ; superposer et
planche acceptent l'un ou l'autre.

    grille PHOTO SORTIE.png [x0,y0,x1,y1] [pas]
        Grille numérotée sur la photo (une case = une unité du repère, 48 de large par défaut,
        ou « pas » pixels de photo par unité) pour y relever la silhouette, en unités.
    superposer FICHIER.json PHOTO x0,y0,x1,y1 SORTIE.png [zoom] [CLE]
        La photo recadrée (x0,y0,x1,y1 : la zone de la photo qui correspond exactement à la
        toile du sprite) ramenée à la taille du sprite, puis trois panneaux : photo, sprite,
        et le contour du sprite tracé sur la photo. Sert à chaque étape de forme.
    pipette PHOTO x0,y0,x1,y1 [x0,y0,x1,y1 …]
        Couleur médiane de chaque zone de la photo (prendre des zones bien éclairées, sans
        reflet ni ombre) : les couleurs à plat de l'étape 2.
    apercu SPRITE.json SORTIE.png CLE[,CLE…] [zoom] [PHOTO]
        Images du sprite côte à côte, la photo de référence à droite si donnée.
    zoom SPRITE.json SORTIE.png CLE x0,y0,x1,y1
        Une zone agrandie, pixel par pixel, pour juger un détail (épaule, œil, doigts).
    controle SPRITE.json
        Groupes de pixels qui ne tiennent pas au reste (contour K et k exclu, voisins par un côté),
        image par image. Les grands groupes sont des formes cernées tout autour (membre de
        l'autre côté, membre proche cerné) ; les petits (moins de 12 pixels) sont des miettes :
        un doigt, un orteil ou un talon détaché, à corriger.
    composer SORTIE.json LARGEURxHAUTEUR SPRITE.json@x,y[,m] [SPRITE.json@x,y[,m] …]
        Une composition : chaque sprite posé en (x, y) sur la toile, « m » pour le retourner
        (tête à droite). Le miroir retourne aussi la lumière : pour une scène d'après photo,
        dessiner plutôt la grenouille déjà tournée (MIROIR dans modele.py).
    planche FICHIER.json SORTIE.html NOM "TEXTE" [ETAPE]
        La planche de relecture (planche.html) : en grand avec une grille de pixels numérotée,
        aux tailles d'affichage, et, si le sprite est animé, chaque position. ETAPE (texte
        libre, ex. « Étape 1 · silhouette ») s'affiche en tête de page.

Les photos de référence restent dans le bac à sable (scratchpad) : jamais dans le dépôt ni dans
une page publiée. Les images produites par superposer contiennent la photo : à montrer dans la
conversation seulement.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ICI = Path(__file__).resolve().parent
FOND = (10, 15, 14)


def rangees(sprite, cle):
    return [
        "".join(c * int(n) for c, n in re.findall(r"(\D)(\d+)", r)) for r in sprite["images"][cle]
    ]


def plages(ligne):
    """« ...OOOK » → « .3O3K1 »."""
    return "".join(f"{m.group()[0]}{len(m.group())}" for m in re.finditer(r"(.)\1*", ligne))


def cle_fixe(sprite, cle=None):
    """La clé demandée si le sprite l'a, sinon la position de repos, sinon la première."""
    if cle and cle in sprite["images"]:
        return cle
    return "0001" if "0001" in sprite["images"] else next(iter(sprite["images"]))


def rgba(fichier, cle=None):
    """Sprite ou composition → tableau RGBA (hauteur, largeur, 4)."""
    d = json.load(open(fichier)) if isinstance(fichier, (str, Path)) else fichier
    if "grenouilles" not in d:
        d = {
            "largeur": d["largeur"],
            "hauteur": d["hauteur"],
            "grenouilles": [{"x": 0, "y": 0, "sprite": d}],
        }
    out = np.zeros((d["hauteur"], d["largeur"], 4), np.uint8)
    for g in d["grenouilles"]:
        s = g["sprite"]
        for j, r in enumerate(rangees(s, cle_fixe(s, cle))):
            for i, c in enumerate(r):
                y, x = g["y"] + j, g["x"] + i
                if c != "." and 0 <= y < out.shape[0] and 0 <= x < out.shape[1]:
                    h = s["palette"][c]
                    out[y, x] = [int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16), 255]
    return out


def agrandir(im, z):
    return im.resize((im.width * z, im.height * z), Image.NEAREST)


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


def superposer(fichier, photo, cadre, sortie, zoom="6", cle=None):
    a = rgba(fichier, cle)
    h, w = a.shape[:2]
    z = int(zoom)
    x0, y0, x1, y1 = (int(v) for v in cadre.split(","))
    if abs((x1 - x0) / (y1 - y0) - w / h) > 0.02 * w / h:
        print(f"attention : cadre {x1 - x0} × {y1 - y0}, toile {w} × {h} : proportions différentes")
    ref = Image.open(photo).convert("RGB").crop((x0, y0, x1, y1)).resize((w, h), Image.LANCZOS)
    sprite = Image.new("RGB", (w, h), FOND)
    sprite.paste(Image.fromarray(a[..., :3]), mask=Image.fromarray(a[..., 3]))
    # contour : pixels pleins dont un voisin (par un côté) est vide
    m = a[..., 3] > 0
    v = np.pad(m, 1)
    bord = m & ~(v[:-2, 1:-1] & v[2:, 1:-1] & v[1:-1, :-2] & v[1:-1, 2:])
    calque = agrandir(ref, z)
    d = ImageDraw.Draw(calque)
    for y, x in zip(*np.nonzero(bord), strict=False):
        d.rectangle([x * z + 1, y * z + 1, x * z + z - 2, y * z + z - 2], outline=(0, 255, 240))
    panneaux = [agrandir(ref, z), agrandir(sprite, z), calque]
    tot = Image.new("RGB", (sum(p.width for p in panneaux) + 20, h * z), "#000")
    for k, p in enumerate(panneaux):
        tot.paste(p, (k * (w * z + 10), 0))
    tot.save(sortie)


def pipette(photo, *zones):
    im = np.asarray(Image.open(photo).convert("RGB"))
    for z in zones:
        x0, y0, x1, y1 = (int(v) for v in z.split(","))
        r, g, b = np.median(im[y0:y1, x0:x1].reshape(-1, 3), axis=0).astype(int)
        print(f"{z} : #{r:02X}{g:02X}{b:02X}")


def apercu(fichier, sortie, cles, zoom="5", photo=None):
    s, z, imgs = json.load(open(fichier)), int(zoom), []
    for cle in cles.split(","):
        a = rgba(s, cle)
        im = Image.new("RGB", (a.shape[1], a.shape[0]), FOND)
        im.paste(Image.fromarray(a[..., :3]), mask=Image.fromarray(a[..., 3]))
        imgs.append(agrandir(im, z))
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
    im = Image.new("RGB", ((x1 - x0) * z, (y1 - y0) * z), FOND)
    for y in range(y0, y1):
        for x in range(x0, x1):
            if g[y][x] != ".":
                im.paste(
                    Image.new("RGB", (z - 1, z - 1), s["palette"][g[y][x]]),
                    ((x - x0) * z, (y - y0) * z),
                )
    im.save(sortie)


def controle(fichier):
    s = json.load(open(fichier))
    for cle in sorted(s["images"]):
        g = rangees(s, cle)
        plein = {(j, i) for j, r in enumerate(g) for i, c in enumerate(r) if c not in ".Kk"}
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
        formes = ", ".join(f"{len(g)} px" for g in groupes[1:] if len(g) >= 12) or "aucune"
        miettes = ", ".join(f"{len(g)} px en {min(g)}" for g in groupes if len(g) < 12) or "aucune"
        print(
            f"{cle} : corps {len(groupes[0])} px ; formes cernées à part : {formes} ; "
            f"miettes (rangée, colonne) : {miettes}"
        )


def composer(sortie, taille, *poses):
    largeur, hauteur = (int(v) for v in taille.lower().split("x"))
    grenouilles = []
    for pose in poses:
        chemin, ou = pose.rsplit("@", 1)
        x, y, *drapeaux = ou.split(",")
        s = json.load(open(chemin))
        if "m" in drapeaux:  # tête à droite : chaque rangée retournée
            s = dict(s, images={c: [plages(r[::-1]) for r in rangees(s, c)] for c in s["images"]})
        grenouilles.append({"nom": Path(chemin).stem, "x": int(x), "y": int(y), "sprite": s})
    composition = {"largeur": largeur, "hauteur": hauteur, "grenouilles": grenouilles}
    Path(sortie).write_text(json.dumps(composition, separators=(",", ":")), encoding="utf-8")


def planche(fichier, sortie, nom, texte, etape="Relecture"):
    page = (ICI / "planche.html").read_text(encoding="utf-8")
    page = page.replace("/*NOM*/", nom).replace("/*TEXTE*/", texte).replace("/*ETAPE*/", etape)
    page = page.replace("/*SPRITE*/null", Path(fichier).read_text(encoding="utf-8"), 1)
    Path(sortie).write_text(page, encoding="utf-8")


if __name__ == "__main__":
    {
        "grille": grille,
        "superposer": superposer,
        "pipette": pipette,
        "apercu": apercu,
        "zoom": zoom,
        "controle": controle,
        "composer": composer,
        "planche": planche,
    }[sys.argv[1]](*sys.argv[2:])
