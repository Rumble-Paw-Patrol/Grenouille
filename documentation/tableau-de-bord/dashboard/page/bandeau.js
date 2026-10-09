/* Bandeau (A. blanci en 8-bit sur son chant), chiffres marqués, navigation entre les pages. */

/* ---------- A. blanci en 8-bit ----------
   Planche de 24 images de 114 × 86 pixels (grenouille.py du dépôt), nommées gorge, flanc,
   clignement, tête : « 0101 ». Elle respire, sa gorge palpite, elle cligne, et de temps en
   temps elle baisse ou relève un peu la tête ; derrière elle, une mosaïque de tuiles ambrées
   qui scintillent. */
{
  const IMAGES = ['0000', '0001', '0002', '0010', '0011', '0012', '0100', '0101', '0102', '0110', '0111', '0112',
    '1000', '1001', '1002', '1010', '1011', '1012', '1100', '1101', '1102', '1110', '1111', '1112'];
  const bandeau = document.getElementById('bandeau');
  const sprite = bandeau.querySelector('.gb-sprite'), toile = bandeau.querySelector('.gb-lueur'), ctx = toile.getContext('2d');
  const LW = 114, LH = 86, DX = 24, DY = 8, TUILE = 2, COL = LW + 32, LIG = LH + DY, NC = COL / TUILE, NL = LIG / TUILE;
  const CX = (DX + LW * 0.54) / TUILE, CY = (DY + LH / 2) / TUILE, RX = (LW * 0.625) / TUILE, RY = (LH * 0.525) / TUILE;
  const calme = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const phases = Array.from({ length: NC * NL }, () => [Math.random() * 6.28, 0.6 + Math.random() * 1.6]);
  let s = 2, p = 2, image = '0001';

  function dimension() {
    const bw = bandeau.clientWidth;
    if (!bw) return;
    s = bw >= 1200 ? 3 : bw >= 900 ? 2.5 : bw >= 640 ? 2 : 1.5;
    const etroit = bw < 640;
    bandeau.classList.toggle('gb-etroit', etroit);
    const dpr = devicePixelRatio || 1;
    p = Math.max(1, Math.round(s * dpr));
    toile.width = COL * p;
    toile.height = LIG * p;
    Object.assign(toile.style, { width: COL * s + 'px', height: LIG * s + 'px', left: '0px', bottom: '0px' });
    Object.assign(sprite.style, {
      width: LW * s + 'px', height: LH * s + 'px', left: DX * s + 'px', bottom: '0px',
      backgroundSize: `${IMAGES.length * LW * s}px ${LH * s}px`,
    });
    const perchoir = bandeau.querySelector('.gb-perchoir');
    perchoir.style.right = etroit ? '' : 'clamp(0px, 2vw, 32px)';
    perchoir.style.width = etroit ? '' : COL * s + 'px';
    bandeau.style.setProperty('--gb-perchoir', etroit ? '0px' : (COL - DX / 2) * s + 'px');
    bandeau.style.setProperty('--gb-grenouille-h', (etroit ? LIG * s : LIG * s - 30) + 'px');
    bandeau.style.setProperty('--gb-grenouille-w', COL * s + 'px');
    montrer(image);
  }
  function montrer(cle) {
    image = cle;
    const i = Math.max(0, IMAGES.indexOf(cle));
    sprite.style.backgroundPosition = `${-i * LW * s}px 0px`;
  }
  function lueur(t) {
    ctx.clearRect(0, 0, toile.width, toile.height);
    const q = p * TUILE, jeu = Math.max(1, Math.round(q / 7));
    for (let y = 0; y < NL; y++) for (let x = 0; x < NC; x++) {
      const d = Math.hypot((x - CX) / RX, (y - CY) / RY);
      if (d >= 1) continue;
      const [ph, v] = phases[y * NC + x];
      const a = (1 - d) * (1 - d) * (calme ? 0.16 : 0.1 + 0.1 * Math.sin(t * v + ph));
      if (a < 0.015) continue;
      ctx.fillStyle = `rgba(245, 165, 58, ${a.toFixed(3)})`;
      ctx.fillRect(x * q + jeu, y * q + jeu, q - jeu, q - jeu);
    }
  }
  let cligne = -9, prochainCligne = 2.5, tete = 1, prochaineTete = 4 + Math.random() * 3;
  function animer(t) {
    const souffle = t % 3.2 < 1.3 ? 1 : 0;
    const gorge = t % 7 < 2.4 && t % 0.6 < 0.3 ? 1 : 0;
    if (t > prochainCligne) { cligne = t; prochainCligne = t + 3 + Math.random() * 4.5 + (Math.random() < 0.25 ? -2.8 : 0); }
    const yeux = t - cligne < 0.13 ? 1 : 0;
    if (t > prochaineTete) { tete = tete !== 1 ? 1 : Math.random() < 0.5 ? 0 : 2; prochaineTete = t + (tete === 1 ? 4 + Math.random() * 5 : 1.6 + Math.random() * 2.4); }
    montrer(`${gorge}${souffle}${yeux}${tete}`);
    lueur(t);
  }
  dimension();
  new ResizeObserver(() => { dimension(); lueur(performance.now() / 1000); }).observe(bandeau);
  if (calme) lueur(0);
  else {
    let dernier = -1;
    const boucle = (ms) => {
      const t = ms / 1000;
      if (t - dernier >= 1 / 12) { dernier = t; animer(t); } // 12 images par seconde, cadence de jeu 8-bit
      requestAnimationFrame(boucle);
    };
    requestAnimationFrame(boucle);
  }

  // Bande de fréquence d'A. blanci sur le spectrogramme (de 3 à 6 kHz, de bas en haut).
  const bande = document.getElementById('bande-blanci');
  dash.onData(() => {
    const rows = GB.rows('reperes');
    if (!rows) return;
    const v = (cle) => Number(rows.find((r) => r.cle === cle)?.valeur);
    const bas = v('blanci_bande_basse_khz'), haut = v('blanci_bande_haute_khz');
    if (Number.isNaN(bas) || Number.isNaN(haut)) return;
    bande.style.top = ((6 - haut) / 3) * 100 + '%';
    bande.style.height = ((haut - bas) / 3) * 100 + '%';
  });
}

/* ---------- chiffres marqués de toutes les pages ---------- */
dash.onData(GB.remplir);

/* ---------- navigation ---------- */
{
  const PAGES = ['vue', 'en-cours', 'donnees', 'chaine', 'annotation', 'encodage', 'anuraset', 'encodeurs', 'tetes', 'amorcage', 'rapports', 'biblio'];
  const ANURA = new Set(['anuraset', 'encodeurs', 'tetes', 'amorcage']);
  const onglets = [...document.querySelectorAll('.gb-onglet')], sous = document.getElementById('sous-onglets');
  const sousOnglets = [...sous.querySelectorAll('.gb-sous-onglet')];
  GB.aller = (page, lien = true) => {
    if (!PAGES.includes(page)) page = 'vue';
    for (const id of PAGES) document.getElementById(id).hidden = id !== page;
    onglets.forEach((b) => b.setAttribute('aria-selected', String(b.dataset.page === page || (b.dataset.page === 'anuraset' && ANURA.has(page)))));
    sous.hidden = !ANURA.has(page);
    sousOnglets.forEach((b) => (b.dataset.page === page ? b.setAttribute('aria-current', 'page') : b.removeAttribute('aria-current')));
    GB.cacher();
    if (lien) dash.setLink(page);
    requestAnimationFrame(() => (GB.dessins[page] || []).forEach((f) => f()));
  };
  [...onglets, ...sousOnglets].forEach((b) => b.addEventListener('click', () => GB.aller(b.dataset.page)));

  // Un lien vers une page, ou vers un élément d'une page, l'ouvre.
  const nom = decodeURIComponent(location.hash.slice(1));
  const cible = nom && document.getElementById(nom);
  const page = PAGES.includes(nom) ? nom : cible?.closest('.gb-page')?.id || 'vue';
  GB.aller(page, false);
  if (cible && !PAGES.includes(nom)) requestAnimationFrame(() => requestAnimationFrame(() => cible.scrollIntoView()));
}
