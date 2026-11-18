"""Découpe un .pptx en paquets de diapos, pour travailler une présentation morceau par morceau.

Chaque paquet garde ses diapos telles quelles (retouches à la main, notes orateur, images) et
leur numéro d'origine : la diapo 6 reste la diapo 6 dans le paquet 2 (numéros écrits en dur sur
les diapos, et `firstSlideNum` pour le champ numéro des pages de notes). Bibliothèque standard
seulement ; le fichier source n'est que lu.

    python documentation/prez/scripts/decouper_pptx.py SOURCE.pptx DOSSIER 1-5 6-10 11-15 ...

Sortie : DOSSIER/<nom de SOURCE>_paquet_<k>_diapos_<a>-<b>.pptx. Pour réassembler, dans
PowerPoint : Accueil > Nouvelle diapositive > Réutiliser les diapositives, en gardant la mise en
forme source.
"""

from __future__ import annotations

import posixpath
import re
import sys
import zipfile
from pathlib import Path

REL_SLIDE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide"


def rels_de(z: zipfile.ZipFile, partie: str) -> dict[str, str]:
    """Cibles (chemins absolus dans le paquet) des relations d'une partie."""
    dossier, nom = posixpath.split(partie)
    chemin = f"{dossier}/_rels/{nom}.rels"
    if chemin not in z.namelist():
        return {}
    xml = z.read(chemin).decode("utf-8")
    out = {}
    for rid, cible, mode in re.findall(
        r'<Relationship [^>]*?Id="([^"]+)"[^>]*?Target="([^"]+)"(?:[^>]*?TargetMode="(\w+)")?',
        xml,
    ):
        if mode != "External":
            out[rid] = posixpath.normpath(posixpath.join(dossier, cible))
    return out


def paquet(source: Path, sortie: Path, garder: list[int]) -> None:
    with zipfile.ZipFile(source) as z:
        pres = z.read("ppt/presentation.xml").decode("utf-8")
        rels = rels_de(z, "ppt/presentation.xml")
        ids = re.findall(r'<p:sldId id="\d+" r:id="(rId\d+)"/>', pres)
        diapos = [rels[r] for r in ids]  # dans l'ordre de la présentation
        jetes_rid = [r for i, r in enumerate(ids, 1) if i not in garder]
        jetes = {rels[r] for r in jetes_rid}
        for r in jetes_rid:
            pres = re.sub(rf'<p:sldId id="\d+" r:id="{r}"/>', "", pres)
        premier = min(garder)
        pres = re.sub(r' firstSlideNum="\d+"', "", pres)
        pres = pres.replace("<p:presentation ", f'<p:presentation firstSlideNum="{premier}" ', 1)
        # Notes des diapos retirées ; puis toute partie qui n'est plus atteinte depuis la racine.
        for d in list(jetes):
            jetes |= {c for c in rels_de(z, d).values() if "/notesSlides/" in c}
        atteintes, pile = set(), ["ppt/presentation.xml"]
        while pile:
            p = pile.pop()
            if p in atteintes or p in jetes:
                continue
            atteintes.add(p)
            pile += rels_de(z, p).values()
        atteintes |= {n for n in z.namelist() if not n.startswith("ppt/")}
        sortie.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(sortie, "w", zipfile.ZIP_DEFLATED) as w:
            for n in z.namelist():
                dossier, nom = posixpath.split(n)
                if dossier.endswith("_rels"):
                    parent = posixpath.join(posixpath.dirname(dossier), nom[: -len(".rels")])
                    if parent.startswith("ppt") and parent not in atteintes:
                        continue
                elif n.startswith("ppt/") and n not in atteintes:
                    continue
                data = z.read(n)
                if n == "ppt/presentation.xml":
                    data = pres.encode("utf-8")
                elif n == "ppt/_rels/presentation.xml.rels":
                    xml = data.decode("utf-8")
                    for r in jetes_rid:
                        xml = re.sub(rf'<Relationship [^>]*?Id="{r}"[^>]*/>', "", xml)
                    data = xml.encode("utf-8")
                elif n == "[Content_Types].xml":
                    xml = data.decode("utf-8")
                    for o in re.findall(r'<Override PartName="/([^"]+)"[^>]*/>', xml):
                        if o.startswith("ppt/") and o not in atteintes:
                            xml = re.sub(rf'<Override PartName="/{re.escape(o)}"[^>]*/>', "", xml)
                    data = xml.encode("utf-8")
                elif n == "docProps/app.xml":
                    data = re.sub(rb"<Slides>\d+</Slides>",
                                  f"<Slides>{len(garder)}</Slides>".encode(), data)
                w.writestr(z.getinfo(n), data)
    print(f"{sortie.name} : diapos {premier}-{max(garder)} sur {len(diapos)}")


def main() -> None:
    source, dossier, *plages = sys.argv[1:]
    source, dossier = Path(source), Path(dossier)
    for k, plage in enumerate(plages, 1):
        a, b = (int(x) for x in plage.split("-"))
        nom = f"{source.stem}_paquet_{k}_diapos_{a:02d}-{b:02d}.pptx"
        paquet(source, dossier / nom, list(range(a, b + 1)))


if __name__ == "__main__":
    main()
