# ruff: noqa: E501
"""PDF et vidéo MP4 de la prez n° 7, tirés de sa version navigateur (exporter_html.py), sans PowerPoint.

    uv run --with python-pptx --with pillow python documentation/prez/scripts/presentation_7/exporter_html.py
    uv run --with python-pptx --with pillow python documentation/prez/scripts/presentation_7/exporter_html.py documentation/prez/presentation_Leonard_PNA.pptx rapports/presentation_7/video/lecteur_video.html --video
    uv run --with playwright --with imageio-ffmpeg --with pypdf python documentation/prez/scripts/presentation_7/exporter_video.py [pdf|mp4]

Pilote Microsoft Edge (déjà installé : Playwright ne télécharge aucun navigateur) :
- PDF : la page `?print`, une diapo par page de 13,33 × 7,5 pouces, textes vectoriels ; les GIF y
  montrent leur dernière image ;
- MP4 (1920 × 1080, 30 images/s, H.264) : chaque diapo est capturée image par image, transitions
  comprises (la Morphose et la poussée sont figées puis avancées de 1/30 s à chaque capture ; les GIF
  suivent la même horloge), puis tenue quelques secondes (TENUE). Le son de la diapo 3 est posé
  sur la bande audio pendant sa tenue.
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright
from pypdf import PdfReader, PdfWriter

DEPOT = Path(__file__).resolve().parents[4]
PREZ = DEPOT / "documentation" / "prez"
HTML = PREZ / "presentation_Leonard_PNA.html"
HTML_VIDEO = DEPOT / "rapports" / "presentation_7" / "video" / "lecteur_video.html"
SON = DEPOT / "rapports" / "presentation_7" / "extrait.wav"
PDF, MP4 = PREZ / "presentation_Leonard_PNA.pdf", PREZ / "presentation_Leonard_PNA.mp4"
IPS = 30
# tenue de chaque diapo après sa transition (ms), numéros à partir de 1 ; sinon selon la transition
TENUE = {1: 5000, 3: 11000, 8: 9000, 15: 11500, 17: 2500, 18: 2500, 19: 2500, 36: 6000}
TENUE_DEFAUT = {"push": 7000, "morph": 4000, None: 5000}
SEULEMENT = {a for a in sys.argv[1:]}  # « pdf » ou « mp4 » : l'un des deux seulement

with sync_playwright() as p:
    nav = p.chromium.launch(channel="msedge")
    if not SEULEMENT or "pdf" in SEULEMENT:
        # une diapo à la fois, puis on assemble : imprimées ensemble, Edge réduit toutes les pages
        # aux deux tiers (heuristique d'ajustement à la page), alors que chacune seule tient exactement
        page = nav.new_page()
        page.goto(HTML.as_uri() + "?print")
        page.evaluate("document.fonts.ready")
        nb = page.evaluate("document.querySelectorAll('.page').length")
        assemblage = PdfWriter()
        for k in range(nb):
            page.evaluate(f"document.querySelectorAll('.page').forEach((e, i) => e.style.display = i === {k} ? '' : 'none')")
            morceau = page.pdf(width="13.333in", height="7.5in", print_background=True, prefer_css_page_size=True)
            assemblage.add_page(PdfReader(io.BytesIO(morceau)).pages[0])
        assemblage.compress_identical_objects()
        with PDF.open("wb") as f:
            assemblage.write(f)
        page.close()
        print("écrit", PDF, f"{PDF.stat().st_size / 1e6:.1f} Mo,", nb, "pages")
    if not SEULEMENT or "mp4" in SEULEMENT:
        page = nav.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
        page.goto(HTML_VIDEO.as_uri() + "?video")
        page.evaluate("deck.prepare()")
        n = page.evaluate("deck.n")
        sortie = str(MP4.with_suffix(".sans_son.mp4"))
        ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(IPS), "-c:v", "png", "-i", "-",
                               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium", "-movflags", "+faststart", sortie], stdin=subprocess.PIPE)
        images, debut_son, captures = 0, None, 0
        for k in range(n):
            dur = page.evaluate(f"deck.demarre({k})")
            tr = page.evaluate(f"deck.transition({k})")
            tenue = TENUE.get(k + 1, TENUE_DEFAUT[tr["type"] if tr else None])
            gif = page.evaluate("deck.aGif()")
            if k + 1 == 3:
                debut_son = (images + round(dur * IPS / 1000)) / IPS
            derniere = None
            for f in range(round((dur + tenue) * IPS / 1000)):
                t = f * 1000 / IPS
                if derniere is None or t <= dur + 1000 / IPS or gif:
                    page.evaluate(f"deck.temps({t})")
                    derniere = page.screenshot(type="png")
                    captures += 1
                ff.stdin.write(derniere)
                images += 1
            page.evaluate("deck.termine()")
            print(f"diapo {k + 1} : transition {dur} ms, tenue {tenue} ms", flush=True)
        ff.stdin.close()
        ff.wait()
        page.close()
        # le son de la diapo 3, posé au début de sa tenue
        delai = round(debut_son * 1000)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", sortie, "-i", str(SON),
                        "-filter_complex", f"[1:a]adelay={delai}|{delai},apad[a]", "-map", "0:v", "-map", "[a]", "-shortest",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(MP4)], check=True)
        Path(sortie).unlink()
        print("écrit", MP4, f"{MP4.stat().st_size / 1e6:.1f} Mo, {images / IPS:.0f} s, {captures} captures")
    nav.close()
