/* Outils communs aux scripts de la page, partagés par window.GB : formats français, création
   d'éléments, état des sources, remplissage des chiffres marqués, infobulle, pages. */
(() => {
  const GB = (window.GB = window.GB || {});
  const racine = document.getElementById('gb');
  const LOC = 'fr-FR';
  const vide = (x) => x == null || x === '' || Number.isNaN(+x);

  GB.fr = (x, n = 2) => (vide(x) ? '—' : (+x).toLocaleString(LOC, { minimumFractionDigits: n, maximumFractionDigits: n }));
  GB.entier = (x) => (vide(x) ? '—' : Math.round(+x).toLocaleString(LOC));
  GB.date = (s) => {
    if (!s) return '—';
    const [a, m, j] = String(s).slice(0, 10).split('-');
    return j ? `${j}/${m}/${a}` : String(s);
  };
  GB.vrai = (x) => x === true || x === 'True' || x === 'true';
  GB.num = (x) => (vide(x) ? null : +x);

  GB.el = (tag, attrs = {}, ...kids) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === 'class') e.className = v;
      else if (k === 'style') e.style.cssText = v;
      else if (k.startsWith('on') && typeof v === 'function') e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v === true ? '' : v);
    }
    for (const k of kids.flat(Infinity)) if (k != null && k !== false) e.append(k.nodeType ? k : document.createTextNode(String(k)));
    return e;
  };
  const NS = 'http://www.w3.org/2000/svg';
  GB.sv = (tag, attrs = {}, ...kids) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) if (v != null) e.setAttribute(k, v);
    for (const k of kids.flat(Infinity)) if (k != null && k !== false) e.append(k.nodeType ? k : document.createTextNode(String(k)));
    return e;
  };
  // Largeur d'un texte tel qu'il sera dessiné dans ce svg (déjà dans la page).
  GB.mesure = (svg, texte, cls) => {
    const t = GB.sv('text', { class: cls, x: 0, y: -9999 }, texte);
    svg.append(t);
    const w = t.getComputedTextLength();
    t.remove();
    return w;
  };

  /* ---------- sources ---------- */
  GB.rows = (id) => {
    const d = dash.data(id);
    return d.status === 'ok' && Array.isArray(d.data) ? d.data : null;
  };
  GB.statut = (...ids) => {
    let s = 'ok';
    for (const id of ids) {
      const st = dash.data(id).status;
      if (st === 'loading') s = 'loading';
      else if (st !== 'ok') return 'error';
    }
    return s;
  };
  GB.echec = () => GB.el('div', { class: 'gb-echec' }, GB.el('i'), 'Échec du chargement');
  // Prépare la boîte d'un graphique : vraie si ses sources sont là, sinon chargement ou échec.
  GB.pret = (box, ...ids) => {
    const s = GB.statut(...ids);
    if (s === 'ok') {
      box.classList.remove('dash-skeleton');
      return true;
    }
    box.replaceChildren(...(s === 'error' ? [GB.echec()] : []));
    box.classList.toggle('dash-skeleton', s === 'loading');
    return false;
  };
  // Remplit le corps d'un tableau tiré des lignes d'une source (fn rend les <tr>).
  GB.table = (table, ids, fn) => {
    const tb = table.tBodies[0];
    const s = GB.statut(...[ids].flat());
    if (s !== 'ok') {
      const n = table.tHead.rows[0].cells.length;
      const td = GB.el('td', { colspan: n, class: 'gb-table-etat' + (s === 'loading' ? ' dash-skeleton' : '') }, s === 'error' ? GB.echec() : null);
      tb.replaceChildren(GB.el('tr', {}, td));
      return;
    }
    tb.replaceChildren(...fn());
  };
  GB.td = (texte, cls) => GB.el('td', { class: cls }, texte);

  /* ---------- chiffres marqués dans le HTML (data-fmt) ---------- */
  const FORMATS = {
    entier: GB.entier,
    fr1: (x) => GB.fr(x, 1),
    fr2: (x) => GB.fr(x, 2),
    texte: (x) => (x == null || x === '' ? '—' : String(x)),
    date: GB.date,
    millions: (x) => GB.fr(+x / 1e6, 1),
  };
  const ligne = (rows, n) => {
    if (n.dataset.row != null) return rows[+n.dataset.row];
    if (!n.dataset.where) return rows[0];
    const conds = n.dataset.where.split('&').map((p) => p.split('=').map(decodeURIComponent));
    return rows.find((r) => conds.every(([k, v]) => String(r[k]) === v));
  };
  GB.remplir = () => {
    racine.querySelectorAll('[data-fmt][data-source][data-field]').forEach((n) => {
      const d = dash.data(n.dataset.source);
      if (d.status === 'ok') {
        const rows = Array.isArray(d.data) ? d.data : [d.data];
        const r = ligne(rows, n);
        n.textContent = r ? (FORMATS[n.dataset.fmt] || FORMATS.texte)(r[n.dataset.field]) : '—';
        n.classList.remove('dash-skeleton');
      } else {
        n.textContent = '—';
        n.classList.toggle('dash-skeleton', d.status === 'loading');
      }
    });
  };

  /* ---------- infobulle ---------- */
  const bulle = GB.el('div', { class: 'gb-bulle', hidden: true, role: 'tooltip' });
  racine.append(bulle);
  GB.bulle = (ev, contenu) => {
    bulle.replaceChildren(...[contenu].flat().filter((x) => x != null));
    bulle.hidden = false;
    const R = racine.getBoundingClientRect(), w = bulle.offsetWidth, h = bulle.offsetHeight;
    let x = ev.clientX - R.left + 14, y = ev.clientY - R.top + 14;
    if (x + w > R.width - 4) x = ev.clientX - R.left - w - 14;
    if (ev.clientY + 14 + h > innerHeight - 4) y = ev.clientY - R.top - h - 14;
    bulle.style.left = Math.max(4, Math.min(x, R.width - w - 4)) + 'px';
    bulle.style.top = Math.max(4, y) + 'px';
  };
  GB.cacher = () => { bulle.hidden = true; };
  GB.survol = (n, f) => {
    n.addEventListener('pointermove', (ev) => GB.bulle(ev, f()));
    n.addEventListener('pointerleave', GB.cacher);
  };
  // Contenu d'infobulle : un titre, puis des lignes [couleur ou null, texte].
  GB.contenu = (titre, lignes = []) => [
    GB.el('div', { class: 'gb-bulle-titre' }, titre),
    ...lignes.map(([c, t]) => GB.el('div', { class: 'gb-bulle-ligne' }, c ? GB.el('i', { style: `background:${c}` }) : null, GB.el('span', {}, t))),
  ];

  /* ---------- pages : un dessin ne se fait que sur la page affichée ---------- */
  GB.dessins = {};
  GB.visible = (page) => {
    const s = document.getElementById(page);
    return !!s && !s.hidden;
  };
  GB.dessin = (page, fn) => {
    (GB.dessins[page] = GB.dessins[page] || []).push(fn);
    dash.onData(() => { if (GB.visible(page)) fn(); });
  };

  /* ---------- libellés ---------- */
  GB.ESPECES = ['DENMIN', 'PITAZU', 'PHYCUV', 'LEPLAT', 'BOAFAB'];
  GB.coulEsp = (sp) => dash.colors[Math.max(0, GB.ESPECES.indexOf(sp)) % dash.colors.length];
  GB.TETES = {
    logistic: 'logistique', proto_probe: 'sonde à prototypes (jetons)', attentive: 'sonde attentive (jetons)',
    'logistic:max': 'logistique sur le max des jetons', simple_prototype: 'prototype simple', lda_shrunk: 'LDA rétrécie',
    'logistic+R37=glmm': 'logistique + biais par site (GLMM)', 'logistic+R37': 'logistique + biais par site', prototype: 'prototype différentiel',
    'knn:k=5': '5 plus proches voisins', dann: 'DANN (adversaire du site)', 'loss:focal': 'logistique, perte focale',
    'logistic+R13': 'logistique + poids par micro', 'logistic+R18=64': 'logistique + ACP à 64 dimensions', 'logistic+R19': 'logistique + centrage par point',
    'logistic+R19+R37': 'logistique + centrage par point + biais par site', 'logistic+R20': 'logistique + AdaBN par site',
    'logistic+R21': 'logistique + retrait des directions du site', classifieur_origine: "classifieur d'origine (sans entraînement)",
  };
  GB.tete = (h) => GB.TETES[h] || h;
  GB.nomEnc = (e) => String(e).replace(/^esp_aves2_/, 'aves2 ');
  GB.etat = (texte) => GB.el('span', { class: 'gb-etat', 'data-etat': texte }, texte);
  GB.seg = (options, valeur, f, label) => {
    const g = GB.el('div', { class: 'gb-seg', role: 'group', 'aria-label': label });
    for (const [v, nom] of options) g.append(GB.el('button', { type: 'button', 'aria-pressed': String(v === valeur), onclick: () => f(v) }, nom));
    return g;
  };
  GB.choix = (actif, nom, f, couleur, titre) =>
    GB.el('button', { type: 'button', class: 'gb-choix', 'aria-pressed': String(actif), title: titre, onclick: f }, couleur ? GB.el('i', { style: `background:${couleur}` }) : null, nom);
})();
