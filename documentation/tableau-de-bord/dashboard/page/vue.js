/* Vue d'ensemble, En développement, Données, Chaîne de traitement, Annotation, Temps d'encodage. */

/* ---------- calculs ---------- */
dash.calc('corpus', {
  title: "Corpus de l'ONF",
  description: "Totaux de l'inventaire (heures, enregistrements, Go, sites, les relevés 2026 rattachés à leur site) et enregistrements écartés par le contrôle qualité.",
  inputs: ['inventaire', 'controle_qualite'],
  fn: (inventaire, controle_qualite) => {
    const somme = (k) => inventaire.reduce((a, r) => a + Number(r[k]), 0);
    return [{
      heures: somme('heures'),
      enregistrements: somme('enregistrements'),
      go: somme('go'),
      sites: new Set(inventaire.map((r) => r.lieu)).size,
      ecartes: controle_qualite.filter((r) => r.decision === 'écarté').reduce((a, r) => a + Number(r.enregistrements), 0),
    }];
  },
});

dash.calc('annotation_avancement', {
  title: 'Avancement du premier lot',
  description: "Enregistrements annotés (et positifs) par jeu, face aux cibles du plan d'annotation v1.",
  inputs: ['annotation_sites', 'reperes'],
  fn: (annotation_sites, reperes) => {
    const cible = (cle) => Number(reperes.find((r) => r.cle === cle).valeur);
    const somme = (jeu, k) => annotation_sites.filter((r) => !jeu || r.jeu === jeu).reduce((a, r) => a + Number(r[k]), 0);
    return [
      { jeu: 'entraînement', annotes: somme('entraînement', 'annotes'), positifs: somme('entraînement', 'positifs'), cible: cible('cible_entrainement'), cible_positifs: null },
      { jeu: 'évaluation', annotes: somme('évaluation', 'annotes'), positifs: somme('évaluation', 'positifs'), cible: cible('cible_evaluation'), cible_positifs: cible('cible_positifs_evaluation') },
      { jeu: 'total', annotes: somme(null, 'annotes'), positifs: somme(null, 'positifs'), cible: cible('cible_entrainement') + cible('cible_evaluation'), cible_positifs: null },
    ];
  },
});

dash.calc('annotation_par_site', {
  title: 'Annotations par site',
  description: 'Enregistrements annotés et positifs par site, entraînement et évaluation réunis.',
  inputs: ['annotation_sites'],
  fn: (annotation_sites) => {
    const parSite = new Map();
    for (const r of annotation_sites) {
      const x = parSite.get(r.site) || { site: r.site, annotes: 0, positifs: 0 };
      x.annotes += Number(r.annotes);
      x.positifs += Number(r.positifs);
      parSite.set(r.site, x);
    }
    return [...parSite.values()].sort((a, b) => b.annotes - a.annotes);
  },
});

dash.calc('encodage_bilan', {
  title: 'Encodeurs ayant fini',
  description: 'Nombre des encodeurs suivis qui ont encodé tout le corpus, sur le nombre suivi.',
  inputs: ['encodage'],
  fn: (encodage) => [{ finis: encodage.filter((r) => r.statut === 'fini').length, encodeurs: encodage.length }],
});

dash.calc('tests_total', {
  title: 'Tests automatiques',
  description: 'Nombre total de tests du dépôt, tous dossiers de tests/ réunis.',
  inputs: ['tests'],
  fn: (tests) => [{ tests: tests.reduce((a, r) => a + Number(r.tests), 0), dossiers: tests.length }],
});

/* ======================================================================
   Vue d'ensemble : résumé des chantiers
   ====================================================================== */
{
  const table = document.getElementById('chantiers-resume');
  dash.onData(() => GB.table(table, 'chantiers', () => GB.rows('chantiers').map((t) =>
    GB.el('tr', { 'data-clic': '', onclick: () => GB.aller('en-cours') },
      GB.td(t.titre), GB.el('td', {}, GB.etat(t.etat))))));
}

/* ======================================================================
   En développement : chantiers et derniers commits
   ====================================================================== */
{
  const cartes = document.getElementById('chantiers-cartes');
  dash.onData(() => {
    if (!GB.pret(cartes, 'chantiers')) return;
    cartes.replaceChildren(...GB.rows('chantiers').map((t) => GB.el('article', { class: 'gb-carte gb-chantier' },
      GB.etat(t.etat), GB.el('h3', {}, t.titre), GB.el('p', {}, t.detail),
      t.suite ? GB.el('p', { class: 'gb-note' }, GB.el('strong', {}, 'Ensuite : '), t.suite) : null)));
  });
  const commits = document.getElementById('commits');
  dash.onData(() => GB.table(commits, 'commits', () => GB.rows('commits').map((c) =>
    GB.el('tr', {}, GB.td(c.numero, 'n'), GB.td(GB.date(c.date)), GB.td(c.message), GB.el('td', {}, GB.el('span', { class: 'gb-mono' }, c.commit))))));
}

/* ======================================================================
   Données : heures par site, inventaire, fenêtres, contrôle qualité
   ====================================================================== */
{
  const JEUX = ['2023', '2026'];
  const coulJeu = (j) => dash.colors[JEUX.indexOf(j)];
  document.getElementById('leg-2023').style.background = coulJeu('2023');
  document.getElementById('leg-2026').style.background = coulJeu('2026');
  const box = document.getElementById('heures-sites');
  GB.dessin('donnees', () => {
    if (!GB.pret(box, 'inventaire')) return;
    const W = box.clientWidth;
    if (!W) return;
    const rows = GB.rows('inventaire');
    const lieux = d3.groups(rows, (r) => r.lieu)
      .map(([nom, parts]) => ({ nom, parts, total: d3.sum(parts, (r) => +r.heures) }))
      .sort((a, b) => b.total - a.total);
    const rh = 58, H = lieux.length * rh;
    const svg = GB.sv('svg', { width: '100%', viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': "Heures d'audio par site et par campagne" });
    box.replaceChildren(svg);
    const g = Math.ceil(d3.max(lieux, (L) => GB.mesure(svg, L.nom, 'fort'))) + 12;
    const d = Math.ceil(d3.max(lieux, (L) => GB.mesure(svg, GB.entier(L.total) + ' h', 'fort'))) + 10;
    const x = d3.scaleLinear([0, d3.max(lieux, (L) => L.total)], [g, W - d]);
    lieux.forEach((L, k) => {
      const y = k * rh;
      svg.append(GB.sv('text', { x: g - 10, y: y + 22, 'text-anchor': 'end', class: 'fort' }, L.nom));
      let x0 = 0;
      for (const jeu of JEUX) {
        const parts = L.parts.filter((r) => r.jeu === jeu);
        const h = d3.sum(parts, (r) => +r.heures);
        if (!h) continue;
        const r = GB.sv('rect', { x: x(x0), y: y + 6, width: Math.max(1, x(x0 + h) - x(x0) - 2), height: 22, rx: 4, fill: coulJeu(jeu) });
        GB.survol(r, () => GB.contenu(`${L.nom} · ${jeu}`, parts.map((p) => [coulJeu(jeu), `${p.site} : ${GB.fr(p.heures, 1)} h, ${GB.entier(p.enregistrements)} enregistrements`])));
        svg.append(r);
        x0 += h;
      }
      svg.append(GB.sv('text', { x: x(L.total) + 6, y: y + 22, class: 'fort gb-num' }, GB.entier(L.total) + ' h'));
      const releves = L.parts.filter((r) => r.jeu === '2026' && r.site !== L.nom).map((r) => r.site);
      if (releves.length) svg.append(GB.sv('text', { x: g, y: y + 45, style: 'font-size:11px' }, `2026 : relevé${releves.length > 1 ? 's' : ''} ${releves.join(', ')}`));
    });
  });

  const inv = document.getElementById('inventaire');
  dash.onData(() => GB.table(inv, 'inventaire', () => GB.rows('inventaire').map((r) =>
    GB.el('tr', {}, GB.td(r.jeu), GB.td(r.site), GB.td(GB.entier(r.enregistrements), 'n'), GB.td(GB.fr(r.heures, 1), 'n'), GB.td(GB.fr(r.go, 1), 'n')))));
  const fen = document.getElementById('fenetres');
  dash.onData(() => GB.table(fen, 'fenetres', () => GB.rows('fenetres').map((r) =>
    GB.el('tr', {}, GB.td(r.fenetre), GB.td(r.pas), GB.td(r.encodeurs), GB.td(GB.entier(r.par_enregistrement), 'n'), GB.td(GB.entier(r.total), 'n')))));
  const qc = document.getElementById('controle-qualite');
  dash.onData(() => GB.table(qc, 'controle_qualite', () => GB.rows('controle_qualite').map((r) =>
    GB.el('tr', {}, GB.td(r.motif), GB.td(r.decision), GB.td(GB.entier(r.enregistrements), 'n')))));
}

/* ======================================================================
   Chaîne de traitement : le chemin principal sur une ligne, la boucle d'annotation dessous
   ====================================================================== */
{
  const POS = {
    inventaire: [1, 1], qc: [2, 1], signal: [3, 1], encodage: [4, 1], tetes: [5, 1], fusion: [6, 1], score: [7, 1], evaluation: [8, 1],
    reentrainement: [5, 2], annotation: [6, 2], verification: [7, 2],
  };
  const BOUCLE = new Set(['verification>annotation', 'annotation>reentrainement', 'reentrainement>tetes', 'score>verification']);
  const COUL = { fait: 'var(--color-ok)', 'en cours': 'var(--color-warn)', 'à venir': 'var(--color-fg-muted)', bloqué: 'var(--color-bad)' };
  const flux = document.getElementById('chaine-flux');
  const detail = document.getElementById('chaine-detail');
  let choisi = 'encodage', noeuds = {}, liens = null;

  const suites = (e, byId) => (e.suite ? e.suite.split(',').filter(Boolean) : []).map((x) => byId[x]?.etape || x);
  function montrerDetail() {
    if (!GB.pret(detail, 'chaine')) return;
    const rows = GB.rows('chaine');
    const byId = Object.fromEntries(rows.map((e) => [e.id, e]));
    const e = byId[choisi] || rows[0];
    Object.entries(noeuds).forEach(([k, n]) => n.setAttribute('aria-pressed', String(k === e.id)));
    detail.replaceChildren(
      GB.el('div', { class: 'gb-carte-tete' }, GB.el('h3', {}, e.etape), GB.etat(e.etat)),
      GB.el('p', { class: 'gb-texte' }, e.resume),
      GB.el('dl', {},
        GB.el('dt', {}, 'Module'), GB.el('dd', {}, GB.el('code', {}, e.module)),
        GB.el('dt', {}, 'Commandes'), GB.el('dd', {}, e.commandes || 'à écrire'),
        e.repere ? [GB.el('dt', {}, 'Repère'), GB.el('dd', {}, e.repere)] : null,
        GB.el('dt', {}, 'Tests'), GB.el('dd', {}, `${GB.entier(e.tests)} dans `, GB.el('code', {}, e.dossier_tests + '/')),
        GB.el('dt', {}, 'Ensuite'), GB.el('dd', {}, suites(e, byId).join(', ') || '—')));
  }
  function tracer() {
    if (!liens || !GB.visible('chaine')) return;
    liens.replaceChildren();
    if (getComputedStyle(liens).display === 'none') return;
    const rows = GB.rows('chaine');
    if (!rows) return;
    const byId = Object.fromEntries(rows.map((e) => [e.id, e]));
    const B = flux.getBoundingClientRect();
    liens.setAttribute('viewBox', `0 0 ${B.width} ${B.height}`);
    const fleche = (id, c) => GB.sv('marker', { id, viewBox: '0 0 8 8', refX: 7, refY: 4, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, GB.sv('path', { d: 'M0 0L8 4L0 8z', style: `fill:${c};stroke:none` }));
    liens.append(GB.sv('defs', {}, fleche('gb-fl', 'var(--color-fg-muted)'), fleche('gb-flb', 'var(--gb-boucle)')));
    const R = (id) => {
      const r = noeuds[id].getBoundingClientRect();
      return { l: r.left - B.left, r: r.right - B.left, t: r.top - B.top, b: r.bottom - B.top, cx: (r.left + r.right) / 2 - B.left, cy: (r.top + r.bottom) / 2 - B.top };
    };
    for (const e of rows) for (const t of e.suite ? e.suite.split(',') : []) {
      if (!noeuds[t] || !noeuds[e.id] || !POS[e.id] || !POS[t]) continue;
      const a = R(e.id), b = R(t);
      const [ca, ra] = POS[e.id], [cb, rb] = POS[t];
      let d;
      if (ra === rb) d = cb > ca ? `M${a.r} ${a.cy} L${b.l - 2} ${b.cy}` : `M${a.l} ${a.cy} L${b.r + 2} ${b.cy}`;
      else if (ca === cb) d = rb > ra ? `M${a.cx} ${a.b} L${b.cx} ${b.t - 2}` : `M${a.cx} ${a.t} L${b.cx} ${b.b + 2}`;
      else if (rb > ra) d = `M${a.cx} ${a.b} C${a.cx} ${(a.b + b.cy) / 2} ${a.cx} ${b.cy} ${b.l - 2} ${b.cy}`;
      else d = `M${a.r} ${a.cy} C${b.cx} ${a.cy} ${b.cx} ${a.cy} ${b.cx} ${b.b + 2}`;
      const boucle = BOUCLE.has(`${e.id}>${t}`);
      const option = byId[e.id].optionnel === true || byId[t]?.optionnel === true;
      liens.append(GB.sv('path', { d, class: boucle ? 'boucle' : option ? 'option' : null, 'marker-end': boucle ? 'url(#gb-flb)' : 'url(#gb-fl)' }));
    }
  }
  GB.dessin('chaine', () => {
    if (!GB.pret(flux, 'chaine')) { noeuds = {}; liens = null; montrerDetail(); return; }
    const rows = GB.rows('chaine');
    const byId = Object.fromEntries(rows.map((e) => [e.id, e]));
    liens = GB.sv('svg', { class: 'gb-liens', 'aria-hidden': 'true' });
    noeuds = {};
    const enfants = rows.map((e) => {
      const [c, r] = POS[e.id] || [1, 3];
      const n = GB.el('button', { type: 'button', class: 'gb-noeud' + (e.optionnel === true ? ' option' : ''), style: `grid-column:${c};grid-row:${r}`, 'aria-pressed': 'false', onclick: () => { choisi = e.id; montrerDetail(); } },
        GB.el('b', {}, GB.el('i', { style: `background:${COUL[e.etat] || COUL['à venir']}`, title: e.etat }), e.etape),
        e.repere ? GB.el('small', {}, e.repere) : null,
        GB.el('small', { class: 'gb-suites' }, suites(e, byId).length ? '→ ' + suites(e, byId).join(', ') : ''));
      noeuds[e.id] = n;
      return n;
    });
    flux.replaceChildren(liens, ...enfants);
    montrerDetail();
    requestAnimationFrame(tracer);
  });
  new ResizeObserver(() => tracer()).observe(flux);

  const tests = document.getElementById('tests-barres');
  GB.dessin('chaine', () => {
    if (!GB.pret(tests, 'tests')) return;
    const W = tests.clientWidth;
    if (!W) return;
    const rows = GB.rows('tests');
    const rh = 26, H = rows.length * rh;
    const svg = GB.sv('svg', { width: '100%', viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Nombre de tests par dossier' });
    tests.replaceChildren(svg);
    const noms = rows.map((r) => r.dossier.replace(/^tests\//, ''));
    const g = Math.ceil(d3.max(noms, (s) => GB.mesure(svg, s))) + 10;
    const x = d3.scaleLinear([0, d3.max(rows, (r) => +r.tests)], [g, W - 40]);
    rows.forEach((r, i) => {
      const y = i * rh;
      svg.append(GB.sv('text', { x: g - 8, y: y + 17, 'text-anchor': 'end' }, noms[i]));
      const b = GB.sv('rect', { x: g, y: y + 5, width: Math.max(1, x(+r.tests) - g), height: 16, rx: 4, fill: dash.colors[0] });
      GB.survol(b, () => GB.contenu(r.dossier + '/', [[null, `${GB.entier(r.tests)} tests`]]));
      svg.append(b, GB.sv('text', { x: x(+r.tests) + 6, y: y + 17, class: 'fort gb-num' }, GB.entier(r.tests)));
    });
  });
}

/* ======================================================================
   Annotation : avancement, sites et micros
   ====================================================================== */
{
  const av = document.getElementById('avancement');
  const LIB = { 'entraînement': "Entraînement : extraits de 30 s", 'évaluation': 'Test : enregistrements entiers de 2 min' };
  const barre = (titre, n, cible) => GB.el('div', { class: 'gb-av' },
    GB.el('div', { class: 'gb-av-tete' }, GB.el('span', {}, titre), GB.el('span', {}, GB.el('b', {}, GB.entier(n)), ` / ${GB.entier(cible)}`)),
    GB.el('div', { class: 'gb-av-piste', role: 'img', 'aria-label': `${GB.entier(n)} sur ${GB.entier(cible)}` }, GB.el('i', { style: `width:${Math.min(100, (100 * n) / (cible || 1))}%` })));
  dash.onData(() => {
    if (!GB.pret(av, 'annotation_avancement')) return;
    const rows = GB.rows('annotation_avancement').filter((r) => r.jeu !== 'total');
    av.replaceChildren(...rows.flatMap((r) => [
      barre(LIB[r.jeu] || r.jeu, r.annotes, r.cible),
      r.cible_positifs != null ? barre('Positifs dans le jeu de test', r.positifs, r.cible_positifs) : null,
    ]).filter(Boolean));
  });

  const donut = (pos, n, taille = 84) => {
    const r = taille / 2 - 6, c = taille / 2, L = 2 * Math.PI * r, part = n ? pos / n : 0;
    return GB.sv('svg', { viewBox: `0 0 ${taille} ${taille}`, width: taille, height: taille, role: 'img', 'aria-label': `${pos} positifs sur ${n}` },
      GB.sv('circle', { cx: c, cy: c, r, fill: 'none', style: 'stroke:color-mix(in srgb, var(--color-fg) 14%, transparent)', 'stroke-width': 11 }),
      part > 0 ? GB.sv('circle', { cx: c, cy: c, r, fill: 'none', style: 'stroke:var(--gb-ambre)', 'stroke-width': 11, 'stroke-dasharray': `${part * L} ${L}`, transform: `rotate(-90 ${c} ${c})` }) : null,
      GB.sv('text', { x: c, y: c + 5, 'text-anchor': 'middle', class: 'fort gb-num', style: 'font-size:15px' }, GB.entier(n)));
  };
  const tuile = (titre, n, p, f, actif) => GB.el(f ? 'button' : 'div', { type: f ? 'button' : null, class: 'gb-donut', 'aria-pressed': f ? String(actif) : null, onclick: f },
    donut(p, n), GB.el('strong', {}, titre), GB.el('small', {}, `${GB.entier(p)} avec chant · ${GB.entier(n - p)} sans`));
  const donuts = document.getElementById('donuts'), micros = document.getElementById('micros');
  let site = null;
  dash.onData(() => {
    if (!GB.pret(donuts, 'annotation_par_site')) return;
    const rows = GB.rows('annotation_par_site');
    if (!rows.length) { donuts.replaceChildren(GB.el('p', { class: 'gb-vide' }, 'Pas encore d\'annotation comptée.')); return; }
    if (!rows.some((r) => r.site === site)) site = rows[0].site;
    donuts.replaceChildren(...rows.map((r) => tuile(r.site, +r.annotes, +r.positifs, () => { site = r.site; dessinerMicros(); donuts.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(b.querySelector('strong').textContent === site))); }, r.site === site)));
    dessinerMicros();
  });
  function dessinerMicros() {
    if (!GB.pret(micros, 'annotation_micros')) return;
    const ms = GB.rows('annotation_micros').filter((m) => m.site === site).sort((a, b) => b.annotes - a.annotes);
    micros.replaceChildren(ms.length
      ? GB.el('div', { style: 'display:grid;gap:8px' }, GB.el('h4', {}, `Micros de ${site}`), GB.el('div', { class: 'gb-donuts', style: 'min-height:0' }, ms.map((m) => tuile(m.micro, +m.annotes, +m.positifs))))
      : GB.el('p', { class: 'gb-note' }, 'Détail par micro disponible au prochain comptage.'));
  }
}

/* ======================================================================
   Temps d'encodage
   ====================================================================== */
{
  const box = document.getElementById('encodage-barres');
  GB.dessin('encodage', () => {
    if (!GB.pret(box, 'encodage')) return;
    const W = box.clientWidth;
    if (!W) return;
    const rows = GB.rows('encodage');
    const finis = rows.filter((r) => r.statut === 'fini' && r.heures !== '');
    const rh = 40, H = rows.length * rh + 4;
    const svg = GB.sv('svg', { width: '100%', viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': "Temps d'encodage du corpus par encodeur" });
    box.replaceChildren(svg);
    const g = Math.ceil(d3.max(rows, (r) => GB.mesure(svg, r.nom, 'fort'))) + 14;
    const plein = W - g - 70;
    const x = d3.scaleLinear([0, Math.max(1, d3.max(finis, (r) => +r.heures) || 1)], [0, plein]);
    rows.forEach((r, k) => {
      const y = k * rh;
      svg.append(GB.sv('text', { x: g - 12, y: y + 25, 'text-anchor': 'end', class: 'fort' }, r.nom));
      if (r.statut === 'fini' && r.heures !== '') {
        const w = Math.max(2, x(+r.heures));
        const b = GB.sv('rect', { x: g, y: y + 9, width: w, height: 22, rx: 4, fill: dash.colors[0] });
        GB.survol(b, () => GB.contenu(r.nom, [[null, `${GB.fr(r.heures, 1)} h pour le corpus`], r.s_par_enregistrement ? [null, `${GB.fr(r.s_par_enregistrement)} s par enregistrement de 2 min`] : null, r.source ? [null, r.source] : null].filter(Boolean)));
        svg.append(b, GB.sv('text', { x: g + w + 8, y: y + 25, class: 'fort gb-num' }, `${GB.fr(r.heures, +r.heures % 1 ? 1 : 0)} h`));
      } else {
        const txt = r.statut === 'en cours' && r.encodables ? `en cours : ${GB.entier(r.enregistrements)} / ${GB.entier(r.encodables)} enregistrements` : r.statut;
        svg.append(GB.sv('rect', { x: g, y: y + 9, width: Math.max(10, plein), height: 22, rx: 4, fill: 'none', style: 'stroke:var(--color-border-line)', 'stroke-dasharray': '4 4' }),
          GB.sv('text', { x: g + 10, y: y + 25 }, txt));
      }
    });
  });
}
