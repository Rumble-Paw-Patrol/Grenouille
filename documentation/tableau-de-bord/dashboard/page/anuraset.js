/* AnuraSet : le jeu réduit, le benchmark des encodeurs, l'explorateur des têtes, l'amorçage. */

dash.calc('anuraset_bilan', {
  title: 'Jeu AnuraSet réduit',
  description: "Taille du jeu réduit : sites, minutes et fenêtres de 5 s de tous les sites, espèces retenues.",
  inputs: ['anuraset_sites', 'anuraset_especes'],
  fn: (anuraset_sites, anuraset_especes) => [{
    sites: anuraset_sites.length,
    minutes: anuraset_sites.reduce((a, r) => a + Number(r.minutes), 0),
    fenetres: anuraset_sites.reduce((a, r) => a + Number(r.fenetres), 0),
    especes: anuraset_especes.length,
  }],
});

/* ======================================================================
   Le jeu réduit : les plis, qui chante où, la signature des chants
   ====================================================================== */
{
  const plis = document.getElementById('plis');
  dash.onData(() => {
    if (!GB.pret(plis, 'anuraset_sites')) return;
    const sites = GB.rows('anuraset_sites').map((r) => r.site);
    plis.style.gridTemplateColumns = `auto repeat(${sites.length}, minmax(0, 1fr))`;
    plis.replaceChildren(GB.el('span', { class: 't' }), ...sites.map((s) => GB.el('span', { class: 't', style: 'text-align:center' }, s)),
      ...sites.flatMap((test, i) => [GB.el('span', { class: 't' }, `pli ${i + 1}`), ...sites.map((s) => GB.el('span', { class: s === test ? 'test' : 'appr' }, s === test ? 'test' : 'appr.'))]));
  });

  // Matrice espèce × site, à l'unité choisie ; un clic sur un site le tient à l'écart.
  const etat = { unite: 'fenêtre', test: null };
  const boite = document.getElementById('matrice'), lecture = document.getElementById('matrice-lecture'), seg = document.getElementById('matrice-unite');
  function matrice() {
    seg.replaceChildren(GB.seg([['fenêtre', 'fenêtres de 5 s'], ['minute', 'minutes']], etat.unite, (v) => { etat.unite = v; matrice(); }, 'Unité'));
    if (!GB.pret(boite, 'anuraset_positifs', 'anuraset_sites', 'anuraset_especes')) { lecture.textContent = ''; return; }
    const sites = GB.rows('anuraset_sites'), especes = GB.rows('anuraset_especes').map((r) => r.espece);
    const noms = Object.fromEntries(GB.rows('anuraset_especes').map((r) => [r.espece, r.nom]));
    const POS = new Map(GB.rows('anuraset_positifs').map((r) => [`${r.espece}|${r.site}|${r.unite}`, r]));
    const u = etat.unite, unites = u === 'fenêtre' ? 'fenêtres' : 'minutes';
    const taille = (s) => +(u === 'fenêtre' ? s.fenetres : s.minutes);
    const tete = GB.el('tr', {}, GB.el('th', {}, 'Espèce'), sites.map((s) => {
      const actif = etat.test === s.site;
      return GB.el('th', { class: 'site' }, GB.el('button', { type: 'button', 'aria-pressed': String(actif), onclick: () => { etat.test = actif ? null : s.site; matrice(); } },
        s.site, GB.el('small', {}, `${GB.entier(taille(s))} ${unites}`), GB.el('span', {}, actif ? '✓ site de test' : '▸ tenir à l\'écart')));
    }), GB.el('th', {}, 'Total'));
    const corps = especes.map((sp) => {
      let total = 0;
      const cells = sites.map((s) => {
        const v = POS.get(`${sp}|${s.site}|${u}`);
        const n = v ? +v.positifs : 0, tot = v ? +v.total : taille(s);
        total += n;
        const hors = etat.test && etat.test !== s.site;
        if (!n) return GB.el('td', { class: 'absente' + (hors ? ' hors' : ''), title: `${sp} ne chante pas à ${s.site}` }, '—');
        const part = tot ? n / tot : 0;
        const td = GB.el('td', { class: 'cel' + (hors ? ' hors' : ''), style: `background:color-mix(in srgb, ${GB.coulEsp(sp)} ${Math.round(12 + Math.min(1, part / 0.45) * 48)}%, transparent)` },
          GB.el('strong', {}, GB.entier(n)), GB.el('small', {}, GB.fr(part * 100, 0) + ' %'));
        GB.survol(td, () => GB.contenu(`${noms[sp]} à ${s.site}`, [[GB.coulEsp(sp), `${GB.entier(n)} ${unites} positives sur ${GB.entier(tot)} (${GB.fr(part * 100, 1)} %)`]]));
        return td;
      });
      return GB.el('tr', {}, GB.el('td', { class: 'esp' }, GB.el('i', { style: `background:${GB.coulEsp(sp)}` }), GB.el('strong', {}, sp), GB.el('em', {}, noms[sp])), cells, GB.el('td', { class: 'n' }, GB.entier(total)));
    });
    boite.replaceChildren(GB.el('table', { class: 'gb-matrice' }, GB.el('thead', {}, tete), GB.el('tbody', {}, corps)));
    if (!etat.test) {
      lecture.textContent = `Nombre de ${unites} où chaque espèce chante, site par site, et la part que cela représente dans le site. Hachuré : l'espèce n'y chante pas.`;
      return;
    }
    const t = etat.test;
    const absentes = especes.filter((sp) => !(+(POS.get(`${sp}|${t}|${u}`)?.positifs || 0)));
    lecture.textContent = `Pli « ${t} tenu à l'écart » : on apprend sur ${sites.filter((s) => s.site !== t).map((s) => s.site).join(', ')}, on juge sur ${t}. ` +
      (absentes.length ? `${absentes.join(', ')} n'y chante${absentes.length > 1 ? 'nt' : ''} pas : pas d'AP par site pour ce pli, mais ses ${unites} comptent comme négatifs dans l'AP poolée.` : '');
  }
  dash.onData(matrice);

  // Signature des chants : une ligne par espèce, de la plus aiguë à la plus grave, A. blanci en tête.
  const sig = document.getElementById('signature');
  GB.dessin('anuraset', () => {
    if (!GB.pret(sig, 'anuraset_especes', 'reperes')) return;
    const W = sig.clientWidth;
    if (!W) return;
    const rangs = [...GB.rows('anuraset_especes')].sort((a, b) => b.frequence_hz - a.frequence_hz);
    const rep = Object.fromEntries(GB.rows('reperes').map((r) => [r.cle, +r.valeur]));
    const pas = 40, h = 26, b = 44, H = h + pas * (rangs.length + 1) + b - 14;
    const svg = GB.sv('svg', { width: '100%', viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Durée des chants de chaque espèce, rangées par fréquence dominante, avec A. blanci en repère' });
    sig.replaceChildren(svg);
    const etiquettes = rangs.map((r) => `${GB.fr(r.frequence_hz / 1000, 1)} kHz · ${r.nom}`);
    const L = Math.min(W * 0.42, Math.ceil(d3.max([...etiquettes, `${GB.fr(rep.blanci_bande_basse_khz, 1)}–${GB.fr(rep.blanci_bande_haute_khz, 1)} kHz · Guyane`], (s) => GB.mesure(svg, s))) + 34);
    const x = d3.scaleLog([0.05, 6], [L, W - 18]);
    const bas = H - b;
    const ticks = W - L > 320 ? [0.05, 0.1, 0.2, 0.5, 1, 2, 5] : [0.1, 0.5, 1, 5];
    for (const v of ticks) svg.append(
      GB.sv('line', { x1: x(v), x2: x(v), y1: h - 12, y2: bas, class: 'gb-grille-l' }),
      GB.sv('text', { x: x(v), y: bas + 18, 'text-anchor': 'middle' }, GB.fr(v, v < 0.1 ? 2 : v < 1 ? 1 : 0)));
    svg.append(GB.sv('text', { x: (L + W) / 2, y: H - 6, 'text-anchor': 'middle' }, "durée d'un chant, en s (échelle log)"));
    svg.append(GB.sv('text', { x: 4, y: 12 }, 'espèce · fréquence dominante'));
    for (const v of [3, 5]) svg.append(
      GB.sv('line', { x1: x(v), x2: x(v), y1: h - 12, y2: bas, style: 'stroke:var(--cds-chart-reference)', 'stroke-dasharray': '3 4' }),
      GB.sv('text', { x: x(v) - 4, y: h - 2, 'text-anchor': 'end', style: 'font-size:10.5px' }, `fenêtre ${v} s`));
    // A. blanci : une note, et sa bande de fréquence
    const yb = h + pas / 2;
    svg.append(GB.sv('rect', { x: 0, y: yb - pas / 2 + 3, width: W, height: pas - 6, rx: 6, style: 'fill:color-mix(in srgb, var(--gb-ambre) 16%, transparent)' }),
      GB.sv('text', { x: 8, y: yb - 2, class: 'fort' }, 'A. blanci'),
      GB.sv('text', { x: 8, y: yb + 12 }, `${GB.fr(rep.blanci_bande_basse_khz, 1)}–${GB.fr(rep.blanci_bande_haute_khz, 1)} kHz · Guyane`),
      GB.sv('circle', { cx: x(rep.blanci_note_s), cy: yb, r: 6, style: 'fill:var(--gb-ambre);stroke:var(--color-panel)', 'stroke-width': 2 }),
      GB.sv('text', { x: x(rep.blanci_note_s) + 12, y: yb + 4, class: 'fort' }, `note de ${GB.fr(rep.blanci_note_s, 2)} s`));
    rangs.forEach((r, i) => {
      const y = h + pas * (i + 1.5), c = GB.coulEsp(r.espece);
      const [p10, q1, med, q3, p90] = [r.p10_s, r.q1_s, r.mediane_s, r.q3_s, r.p90_s].map(Number);
      const grp = GB.sv('g', {},
        GB.sv('rect', { x: 0, y: y - pas / 2, width: W, height: pas, fill: 'transparent' }),
        GB.sv('circle', { cx: 12, cy: y - 5, r: 5, style: `fill:${c}` }),
        GB.sv('text', { x: 22, y: y - 1, class: 'fort' }, r.espece),
        GB.sv('text', { x: 22, y: y + 13 }, etiquettes[i]),
        GB.sv('line', { x1: x(p10), x2: x(p90), y1: y, y2: y, style: `stroke:${c}`, 'stroke-width': 2, 'stroke-linecap': 'round' }),
        GB.sv('rect', { x: x(q1), y: y - 6, width: Math.max(2, x(q3) - x(q1)), height: 12, rx: 4, style: `fill:${c};fill-opacity:.35;stroke:${c}`, 'stroke-width': 1.5 }),
        GB.sv('circle', { cx: x(med), cy: y, r: 6, style: `fill:${c};stroke:var(--color-panel)`, 'stroke-width': 2 }));
      GB.survol(grp, () => GB.contenu(`${r.nom} (${r.espece})`, [
        [c, `${GB.entier(r.chants)} chants datés · ${GB.entier(r.enregistrements)} enregistrements · ${GB.entier(r.sites)} sites`],
        [null, `fréquence dominante ${GB.fr(r.frequence_hz / 1000, 1)} kHz`],
        [null, `durée médiane ${GB.fr(med, 2)} s · moitié des chants entre ${GB.fr(q1, 2)} et ${GB.fr(q3, 2)} s · 80 % entre ${GB.fr(p10, 2)} et ${GB.fr(p90, 1)} s`]]));
      svg.append(grp);
    });
  });
}

/* ======================================================================
   1 / 3 : benchmark des encodeurs
   ====================================================================== */
{
  const DEFAUT = ['perch_v2', 'birdnet', 'birdnet_v3', 'audioprotopnet', 'naturebeats', 'convnext_birdset'];
  let choisis = null, tri = 'ap_site_meilleure', sens = -1;
  const NUM = new Set(['fenetres_par_s', 'ap_site_meilleure', 'ap_site_logistique']);
  const puces = document.getElementById('enc-puces'), table = document.getElementById('enc-table'), nuage = document.getElementById('enc-nuage');
  const libre = (e) => e.libre === 'oui';
  const classes = () => (GB.rows('encodeurs') || []).filter((e) => e.ap_site_meilleure !== '');
  function rendre() {
    if (!GB.pret(puces, 'encodeurs')) { GB.table(table, 'encodeurs', () => []); GB.pret(nuage, 'encodeurs'); return; }
    const E = classes();
    if (!choisis) choisis = new Set(DEFAUT.filter((x) => E.some((e) => e.encodeur === x)));
    puces.replaceChildren(...[...E].sort((a, b) => b.ap_site_meilleure - a.ap_site_meilleure).map((e) =>
      GB.choix(choisis.has(e.encodeur), e.nom, () => { choisis.has(e.encodeur) ? choisis.delete(e.encodeur) : choisis.add(e.encodeur); rendre(); },
        libre(e) ? 'var(--gb-ambre)' : 'color-mix(in srgb, var(--color-fg) 22%, transparent)', `${e.famille} · ${e.licence}`)));
    table.querySelectorAll('th').forEach((th) => th.setAttribute('aria-sort', th.dataset.field === tri ? (sens > 0 ? 'ascending' : 'descending') : 'none'));
    const rows = E.filter((e) => choisis.has(e.encodeur)).sort((a, b) => {
      let x = a[tri], y = b[tri];
      if (NUM.has(tri)) { x = x === '' ? null : +x; y = y === '' ? null : +y; }
      if (x == null || x === '') return 1;
      if (y == null || y === '') return -1;
      return (x > y ? 1 : x < y ? -1 : 0) * sens;
    });
    GB.table(table, 'encodeurs', () => rows.length ? rows.map((e) => GB.el('tr', { 'data-key': e.encodeur },
      GB.el('td', {}, GB.el('strong', {}, e.nom)), GB.td(e.famille),
      GB.el('td', { style: `color:${libre(e) ? 'var(--color-ok)' : 'inherit'}` }, e.licence),
      GB.td(e.fenetres_par_s === '' ? '—' : GB.fr(e.fenetres_par_s, 1), 'n'),
      GB.td(e.meilleure_tete),
      GB.el('td', { class: 'n' }, GB.el('span', { class: 'gb-barre-cell' }, GB.el('i', { style: `width:${Math.max(0, (e.ap_site_meilleure - 0.3) * 110)}px` }), GB.fr(e.ap_site_meilleure))),
      GB.td(GB.fr(e.ap_site_logistique), 'n')))
      : [GB.el('tr', {}, GB.el('td', { colspan: 7, class: 'gb-vide' }, 'Aucun encodeur choisi.'))]);
    if (GB.visible('encodeurs')) dessinerNuage(rows);
  }
  function dessinerNuage(rows) {
    if (!GB.pret(nuage, 'encodeurs')) return;
    const W = nuage.clientWidth;
    if (!W) return;
    const pts = rows.filter((e) => e.fenetres_par_s !== '');
    if (!pts.length) { nuage.replaceChildren(GB.el('p', { class: 'gb-vide' }, 'Aucun encodeur choisi.')); return; }
    const H = Math.max(300, Math.min(420, W * 0.45)), g = 44, d = 16, h = 16, b = 40;
    const svg = GB.sv('svg', { width: '100%', viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Débit contre qualité des encodeurs' });
    nuage.replaceChildren(svg);
    const tout = classes();
    const ext = d3.extent(tout, (e) => +e.ap_site_meilleure);
    const xs = d3.scaleLog([1, 100], [g, W - d]).clamp(true);
    const ys = d3.scaleLinear([Math.floor(ext[0] * 10) / 10 - 0.02, Math.ceil(ext[1] * 10) / 10], [H - b, h]);
    for (const v of ys.ticks(6)) svg.append(GB.sv('line', { x1: g, x2: W - d, y1: ys(v), y2: ys(v), class: 'gb-grille-l' }), GB.sv('text', { x: g - 8, y: ys(v) + 4, 'text-anchor': 'end' }, GB.fr(v, 1)));
    for (const v of W > 480 ? [1, 2, 5, 10, 20, 50, 100] : [1, 10, 100]) svg.append(GB.sv('text', { x: xs(v), y: H - b + 18, 'text-anchor': 'middle' }, String(v)));
    svg.append(GB.sv('text', { x: (g + W) / 2, y: H - 4, 'text-anchor': 'middle' }, 'fenêtres de 5 s encodées par seconde (échelle log)'));
    svg.append(GB.sv('text', { x: 4, y: 10 }, 'AP par site'));
    const poses = [];
    for (const e of [...pts].sort((a, b) => b.ap_site_meilleure - a.ap_site_meilleure)) {
      const x = xs(+e.fenetres_par_s), y = ys(+e.ap_site_meilleure), l = libre(e);
      const c = GB.sv('circle', { cx: x, cy: y, r: 6, style: l ? 'fill:var(--gb-ambre);stroke:var(--color-panel)' : 'fill:var(--color-panel);stroke:var(--color-fg-muted)', 'stroke-width': 2 });
      GB.survol(c, () => GB.contenu(e.nom, [[null, `${e.famille} · ${e.licence}`], [null, `AP par site ${GB.fr(e.ap_site_meilleure)} (${e.meilleure_tete})`], [null, `${GB.fr(e.fenetres_par_s, 1)} fenêtres/s`]]));
      const larg = GB.mesure(svg, e.nom);
      let lx = x + 9, ly = y + 4;
      if (lx + larg > W - 2) lx = x - 9 - larg;
      while (poses.some((p) => Math.abs(p.y - ly) < 13 && lx < p.x + p.w && p.x < lx + larg)) ly += 13;
      poses.push({ x: lx, y: ly, w: larg });
      svg.append(c, GB.sv('text', { x: lx, y: ly, class: e.encodeur === 'perch_v2' ? 'fort' : null }, e.nom));
    }
  }
  table.querySelectorAll('th button').forEach((bt) => bt.addEventListener('click', () => {
    const k = bt.dataset.tri;
    if (tri === k) sens = -sens; else { tri = k; sens = NUM.has(k) ? -1 : 1; }
    rendre();
  }));
  document.getElementById('enc-tous').addEventListener('click', () => { choisis = new Set(classes().map((e) => e.encodeur)); rendre(); });
  document.getElementById('enc-libres').addEventListener('click', () => { choisis = new Set(classes().filter(libre).map((e) => e.encodeur)); rendre(); });
  document.getElementById('enc-aucun').addEventListener('click', () => { choisis = new Set(); rendre(); });
  dash.onData(rendre);
  GB.dessins.encodeurs = [rendre];
}

/* ======================================================================
   2 / 3 : explorateur des têtes
   ====================================================================== */
{
  const SOURCES = {
    adaptees: { nom: 'Encodeurs × têtes (benchmark global)', ref: ['perch_v2', 'logistic'], encs: ['perch_v2', 'birdnet_v3', 'audioprotopnet'], heads: ['logistic', 'simple_prototype', 'proto_probe'], unites: { fenetre: 'fenêtre', minute: 'minute' } },
    regul: { nom: 'Têtes et régularisations (benchmarks 01–06)', ref: ['perch_v2', 'logistic'], encs: ['perch_v2'], heads: null, unites: { fenetre: 'fenêtre', minute: 'enregistrement' } },
  };
  const etat = { src: 'adaptees', lvl: 'minute', met: 'aps', vue: 'graphe', encs: null, heads: null, sps: new Set(GB.ESPECES) };
  const ctrl = document.getElementById('tetes-controles'), res = document.getElementById('tetes-resume'), zone = document.getElementById('tetes-zone');
  const IDS = ['transfert', 'comparaisons', 'tetes_01_06'];
  const lignesSource = (src) => {
    if (src === 'adaptees') return (GB.rows('transfert') || []).map((r) => ({ enc: r.encodeur, sp: r.espece, head: r.tete, u: r.unite, ap: GB.num(r.ap_poolee), aps: GB.num(r.ap_site) }));
    return (GB.rows('tetes_01_06') || []).map((r) => ({ enc: r.encodeur, sp: r.espece, head: r.tete, u: r.unite, ap: GB.num(r.ap_poolee), aps: GB.num(r.ap_site), diff: GB.num(r.ecart_logistique), sig: r.significatif === '' ? null : GB.vrai(r.significatif) }));
  };
  function init(src) {
    const S = SOURCES[src];
    etat.src = src;
    etat.encs = new Set(S.encs);
    const hs = [...new Set(lignesSource(src).map((r) => r.head))];
    etat.heads = new Set(S.heads ? S.heads.filter((h) => hs.includes(h)) : hs);
    if (src === 'regul') etat.lvl = 'fenetre';
  }
  const multi = (tous, set, nom, coul) => GB.el('div', { class: 'gb-puces' }, tous.map((v) =>
    GB.choix(set.has(v), nom(v), () => { set.has(v) ? set.delete(v) : set.add(v); controles(); }, coul ? coul(v) : null, v)));
  const raccourcis = (tous, set) => GB.el('span', { style: 'display:inline-flex;gap:8px;margin-left:8px' },
    GB.el('button', { type: 'button', class: 'gb-lien-btn', onclick: () => { tous.forEach((v) => set.add(v)); controles(); } }, 'tout'),
    GB.el('button', { type: 'button', class: 'gb-lien-btn', onclick: () => { set.clear(); controles(); } }, 'aucun'));
  function controles() {
    if (GB.statut(...IDS) !== 'ok') { ctrl.replaceChildren(); res.textContent = ''; GB.pret(zone, ...IDS); return; }
    if (!etat.encs) init(etat.src);
    const S = SOURCES[etat.src], rows = lignesSource(etat.src);
    const encs = [...new Set(rows.map((r) => r.enc))].sort(), heads = [...new Set(rows.map((r) => r.head))];
    const select = GB.el('select', { class: 'gb-select', 'aria-label': 'Données' }, Object.entries(SOURCES).map(([k, v]) => GB.el('option', { value: k, selected: k === etat.src }, v.nom)));
    select.addEventListener('change', () => { init(select.value); controles(); });
    ctrl.replaceChildren(
      GB.el('div', { class: 'gb-controles' },
        GB.el('label', { class: 'gb-champ' }, GB.el('span', {}, 'Données'), select),
        GB.el('div', { class: 'gb-champ' }, GB.el('span', {}, 'Unité'), GB.seg([['fenetre', 'fenêtre'], ['minute', S.unites.minute]], etat.lvl, (v) => { etat.lvl = v; controles(); }, 'Unité')),
        GB.el('div', { class: 'gb-champ' }, GB.el('span', {}, "Lecture de l'AP"), GB.seg([['aps', 'moyenne par site'], ['ap', 'poolée']], etat.met, (v) => { etat.met = v; controles(); }, "Lecture de l'AP")),
        GB.el('div', { class: 'gb-champ' }, GB.el('span', {}, 'Affichage'), GB.seg([['graphe', 'graphique'], ['table', 'tableau']], etat.vue, (v) => { etat.vue = v; controles(); }, 'Affichage'))),
      GB.el('div', { class: 'gb-champ' }, GB.el('span', {}, 'Encodeurs', raccourcis(encs, etat.encs)), multi(encs, etat.encs, GB.nomEnc)),
      GB.el('div', { class: 'gb-champ' }, GB.el('span', {}, 'Têtes', raccourcis(heads, etat.heads)), multi(heads, etat.heads, GB.tete)),
      GB.el('div', { class: 'gb-champ' }, GB.el('span', {}, 'Espèces'), multi(GB.ESPECES, etat.sps, (v) => v, GB.coulEsp)));
    if (GB.visible('tetes')) dessiner();
  }
  function donnees() {
    const S = SOURCES[etat.src];
    let met = etat.met, note = '';
    if (etat.src === 'regul' && met === 'aps' && etat.lvl !== 'fenetre') { met = 'ap'; note = "L'AP moyenne par site n'existe qu'à la fenêtre pour les benchmarks 01–06 : AP poolée affichée."; }
    const unite = S.unites[etat.lvl];
    const comp = new Map((GB.rows('comparaisons') || []).map((r) => [`${r.encodeur}|${r.tete}|${r.espece}|${r.unite}`, { diff: GB.num(r.ecart), sig: GB.vrai(r.significatif) }]));
    const sps = GB.ESPECES.filter((x) => etat.sps.has(x));
    const groupes = new Map();
    for (const r of lignesSource(etat.src)) {
      if (r.u !== unite || !etat.encs.has(r.enc) || !etat.heads.has(r.head) || !etat.sps.has(r.sp)) continue;
      const k = r.enc + '|' + r.head;
      if (!groupes.has(k)) groupes.set(k, { enc: r.enc, head: r.head, v: {}, sig: {} });
      const g = groupes.get(k);
      g.v[r.sp] = r[met];
      g.sig[r.sp] = etat.src === 'regul' ? (r.diff != null ? { diff: r.diff, sig: r.sig } : null) : comp.get(`${r.enc}|${r.head}|${r.sp}|${unite}`) || null;
    }
    const lignes = [...groupes.values()].map((g) => {
      const vals = sps.map((x) => g.v[x]).filter((x) => x != null);
      return { ...g, moy: vals.length === sps.length && vals.length ? d3.mean(vals) : null };
    });
    const meilleur = new Map();
    for (const l of lignes) meilleur.set(l.enc, Math.max(meilleur.get(l.enc) ?? -1, l.moy ?? -1));
    lignes.sort((a, b) => (a.enc === b.enc ? (b.moy ?? -1) - (a.moy ?? -1) : (meilleur.get(b.enc) - meilleur.get(a.enc)) || a.enc.localeCompare(b.enc)));
    const ref = lignes.find((l) => l.enc === S.ref[0] && l.head === S.ref[1]);
    return { lignes, sps, met, note, ref, unite };
  }
  function dessiner() {
    if (!GB.pret(zone, ...IDS) || !etat.encs) return;
    const { lignes, sps, met, note, ref, unite } = donnees();
    const lect = met === 'aps' ? "AP moyenne par site tenu à l'écart" : 'AP poolée (tous les sites ensemble)';
    const refTxt = etat.src === 'regul' ? 'Anneau : écart à la logistique du même encodeur significatif après Holm.' : 'Anneau : écart à perch_v2 + logistique significatif après Holm.';
    res.textContent = `${lignes.length} couples · ${lect}, par ${unite}. ${refTxt} ${note}`;
    if (!lignes.length) { zone.replaceChildren(GB.el('p', { class: 'gb-vide' }, 'Aucune combinaison : choisir au moins un encodeur, une tête et une espèce.')); return; }
    if (etat.vue === 'table') {
      zone.replaceChildren(GB.el('div', { class: 'gb-defile gb-haut' }, GB.el('table', { class: 'gb-table' },
        GB.el('thead', {}, GB.el('tr', {}, GB.el('th', {}, 'Encodeur'), GB.el('th', {}, 'Tête'), sps.map((x) => GB.el('th', { class: 'n' }, x)), GB.el('th', { class: 'n' }, 'Moyenne'))),
        GB.el('tbody', {}, lignes.map((l, i) => GB.el('tr', {},
          GB.el('td', {}, i && lignes[i - 1].enc === l.enc ? '' : GB.el('strong', {}, GB.nomEnc(l.enc))), GB.td(GB.tete(l.head)),
          sps.map((x) => GB.el('td', { class: 'n', title: l.sig[x] ? `écart ${GB.fr(l.sig[x].diff)}${l.sig[x].sig ? ', significatif (Holm)' : ''}` : null }, GB.fr(l.v[x]) + (l.sig[x]?.sig ? ' *' : ''))),
          GB.el('td', { class: 'n' }, GB.el('strong', {}, GB.fr(l.moy)))))))),
        GB.el('p', { class: 'gb-note' }, '* écart significatif après Holm.'));
      return;
    }
    const blocs = [];
    lignes.forEach((l, i) => { if (!i || lignes[i - 1].enc !== l.enc) blocs.push({ entete: l.enc }); blocs.push(l); });
    const W = Math.max(600, zone.clientWidth), rh = 24, h = 26, d = 56, H = h + blocs.length * rh + 30;
    const svg = GB.sv('svg', { width: W, viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Comparaison des têtes par encodeur et par espèce' });
    const boite = GB.el('div', { class: 'gb-defile', style: 'position:relative' }, svg);
    zone.replaceChildren(boite, GB.el('div', { class: 'gb-legende' },
      ...sps.map((x) => GB.el('span', {}, GB.el('i', { style: `background:${GB.coulEsp(x)};border-radius:50%` }), x)),
      GB.el('span', {}, GB.el('i', { class: 'gb-trait', style: 'background:var(--color-fg);height:12px;width:3px' }), 'moyenne'),
      GB.el('span', {}, GB.el('i', { class: 'gb-anneau' }), 'écart significatif'),
      ref ? GB.el('span', {}, GB.el('i', { class: 'gb-tirets', style: 'color:var(--gb-ambre)' }), 'référence : perch_v2 · logistique') : null));
    const g = Math.min(300, Math.ceil(d3.max(lignes, (l) => GB.mesure(svg, GB.tete(l.head)))) + 24);
    const xs = d3.scaleLinear([0, 1], [g, W - d]);
    for (const v of [0, 0.2, 0.4, 0.6, 0.8, 1]) svg.append(
      GB.sv('line', { x1: xs(v), x2: xs(v), y1: h - 6, y2: H - 26, class: 'gb-grille-l' }),
      GB.sv('text', { x: xs(v), y: H - 10, 'text-anchor': 'middle' }, GB.fr(v, 1)),
      GB.sv('text', { x: xs(v), y: h - 12, 'text-anchor': 'middle' }, GB.fr(v, 1)));
    if (ref?.moy != null) svg.append(GB.sv('line', { x1: xs(ref.moy), x2: xs(ref.moy), y1: h - 4, y2: H - 28, style: 'stroke:var(--gb-ambre)', 'stroke-width': 1.5, 'stroke-dasharray': '5 4' }));
    blocs.forEach((l, i) => {
      const y = h + i * rh + rh / 2;
      if (l.entete) {
        svg.append(GB.sv('rect', { x: 0, y: y - rh / 2 + 2, width: W, height: rh - 2, style: 'fill:color-mix(in srgb, var(--color-fg) 6%, transparent)' }),
          GB.sv('text', { x: 10, y: y + 5, class: 'fort', style: 'font-size:13px' }, GB.nomEnc(l.entete)));
        return;
      }
      svg.append(GB.sv('text', { x: g - 12, y: y + 4, 'text-anchor': 'end', class: ref && l === ref ? 'fort' : null }, GB.tete(l.head)));
      const vals = sps.map((x) => l.v[x]).filter((v) => v != null);
      if (vals.length > 1) svg.append(GB.sv('line', { x1: xs(d3.min(vals)), x2: xs(d3.max(vals)), y1: y, y2: y, style: 'stroke:var(--color-border-line)', 'stroke-width': 2 }));
      sps.forEach((x) => {
        const v = l.v[x];
        if (v == null) return;
        const sg = l.sig[x];
        const c = GB.sv('circle', { cx: xs(v), cy: y, r: 5, style: `fill:${GB.coulEsp(x)};stroke:${sg?.sig ? 'var(--color-fg)' : 'var(--color-panel)'}`, 'stroke-width': sg?.sig ? 2 : 1.5 });
        GB.survol(c, () => GB.contenu(`${GB.nomEnc(l.enc)} · ${GB.tete(l.head)}`, [[GB.coulEsp(x), `${x} : ${GB.fr(v)}`],
          sg ? [null, `écart à la référence ${sg.diff > 0 ? '+' : ''}${GB.fr(sg.diff)}${sg.sig ? ', significatif (Holm)' : ', non significatif'}`] : null].filter(Boolean)));
        svg.append(c);
      });
      if (l.moy != null) svg.append(GB.sv('rect', { x: xs(l.moy) - 1.5, y: y - 8, width: 3, height: 16, rx: 1.5, style: 'fill:var(--color-fg)' }), GB.sv('text', { x: W - d + 8, y: y + 4, class: 'fort gb-num' }, GB.fr(l.moy)));
    });
    svg.append(GB.sv('text', { x: W - d + 8, y: h - 12, class: 'fort' }, 'moy.'));
  }
  dash.onData(controles);
  GB.dessins.tetes = [controles];
}

/* ======================================================================
   3 / 3 : amorcer un site neuf
   ====================================================================== */
{
  const K = [0, 1, 2, 5, 10, 20, -1], KN = ['0', '1', '2', '5', '10', '20', 'tout'];
  const ORDRE = ['logistic', 'prototype', 'lda_shrunk', 'logistic+R37=glmm', 'logistic+R20'];
  const TRAITS = [null, '7 4', '2 4'];
  let selE = null, selH = null, niv = 'ap_minute';
  const boite = document.getElementById('amorcage-courbes'), pucesE = document.getElementById('amorcage-encodeurs'), pucesH = document.getElementById('amorcage-tetes'), segU = document.getElementById('amorcage-unite');
  const tetes = (rows) => [...new Set(rows.map((r) => r.tete))].sort((a, b) => (ORDRE.indexOf(a) + 1 || 99) - (ORDRE.indexOf(b) + 1 || 99));
  function controles() {
    segU.replaceChildren(GB.seg([['ap_fenetre', 'fenêtre'], ['ap_minute', 'minute']], niv, (v) => { niv = v; controles(); }, 'Unité'));
    const rows = GB.rows('amorcage');
    if (!rows) { pucesE.replaceChildren(); pucesH.replaceChildren(); GB.pret(boite, 'amorcage'); return; }
    const encs = [...new Set(rows.map((r) => r.encodeur))].sort(), heads = tetes(rows);
    if (!selE) selE = ['perch_v2'].filter((e) => encs.includes(e));
    if (!selH) selH = new Set(['logistic', 'prototype', 'lda_shrunk'].filter((h) => heads.includes(h)));
    const coulH = (h) => dash.colors[heads.indexOf(h) % dash.colors.length];
    pucesE.replaceChildren(...encs.map((e) => {
      const i = selE.indexOf(e), on = i >= 0;
      return GB.el('button', { type: 'button', class: 'gb-choix', 'aria-pressed': String(on), onclick: () => { if (on) selE.splice(i, 1); else if (selE.length < 3) selE.push(e); controles(); } },
        on ? GB.sv('svg', { width: 22, height: 8 }, GB.sv('line', { x1: 1, x2: 21, y1: 4, y2: 4, style: 'stroke:var(--color-fg)', 'stroke-width': 2, 'stroke-dasharray': TRAITS[i] })) : null, GB.nomEnc(e));
    }));
    pucesH.replaceChildren(...heads.map((h) => GB.choix(selH.has(h), GB.tete(h), () => { selH.has(h) ? selH.delete(h) : selH.add(h); controles(); }, coulH(h), h)));
    if (GB.visible('amorcage')) dessiner();
  }
  function dessiner() {
    if (!GB.pret(boite, 'amorcage')) return;
    const W = boite.clientWidth;
    if (!W) return;
    const rows = GB.rows('amorcage'), heads = tetes(rows);
    const coulH = (h) => dash.colors[heads.indexOf(h) % dash.colors.length];
    const idx = new Map(rows.map((r) => [`${r.encodeur}|${r.tete}|${r.k}`, r]));
    const series = [];
    selE.forEach((e, ie) => heads.filter((x) => selH.has(x)).forEach((hd) => {
      const v = K.map((k) => GB.num(idx.get(`${e}|${hd}|${k}`)?.[niv]));
      if (v.some((x) => x != null)) series.push({ e, hd, v, trait: TRAITS[ie] });
    }));
    if (!series.length) { boite.replaceChildren(GB.el('p', { class: 'gb-vide' }, 'Choisir au moins un encodeur et une tête.')); return; }
    const H = Math.max(280, Math.min(380, W * 0.42)), g = 40, h = 12, b = 40;
    const svg = GB.sv('svg', { width: '100%', viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': "Courbes d'amorçage par tête" });
    boite.replaceChildren(svg);
    const nomSerie = (x) => (selE.length > 1 ? `${GB.tete(x.hd)} · ${GB.nomEnc(x.e)}` : GB.tete(x.hd));
    const d = Math.min(W * 0.4, Math.ceil(d3.max(series, (x) => GB.mesure(svg, nomSerie(x), 'fort'))) + 16);
    const xk = d3.scalePoint(K.map(String), [g, W - d]);
    const yv = d3.scaleLinear([0, 1], [H - b, h]);
    for (const v of yv.ticks(5)) svg.append(GB.sv('line', { x1: g, x2: W - d, y1: yv(v), y2: yv(v), class: 'gb-grille-l' }), GB.sv('text', { x: g - 8, y: yv(v) + 4, 'text-anchor': 'end' }, GB.fr(v, 1)));
    K.forEach((k, i) => svg.append(GB.sv('text', { x: xk(String(k)), y: H - b + 18, 'text-anchor': 'middle' }, KN[i])));
    svg.append(GB.sv('text', { x: (g + W - d) / 2, y: H - 4, 'text-anchor': 'middle' }, 'enregistrements positifs du site neuf annotés (k)'));
    const fins = [];
    for (const x of series) {
      const pts = x.v.map((y, i) => (y == null ? null : [xk(String(K[i])), yv(y)])).filter(Boolean);
      svg.append(GB.sv('path', { d: 'M' + pts.map((p) => p.join(' ')).join(' L'), fill: 'none', style: `stroke:${coulH(x.hd)}`, 'stroke-width': 2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'stroke-dasharray': x.trait }));
      pts.forEach((p) => svg.append(GB.sv('circle', { cx: p[0], cy: p[1], r: 3.5, style: `fill:${coulH(x.hd)};stroke:var(--color-panel)`, 'stroke-width': 1.5 })));
      fins.push({ y: pts[pts.length - 1][1], txt: nomSerie(x), c: coulH(x.hd) });
    }
    fins.sort((a, b) => a.y - b.y);
    for (let i = 1; i < fins.length; i++) if (fins[i].y - fins[i - 1].y < 14) fins[i].y = fins[i - 1].y + 14;
    fins.forEach((f) => svg.append(GB.sv('text', { x: W - d + 8, y: f.y + 4, class: 'fort', style: `fill:${f.c}` }, f.txt)));
    const curseur = GB.sv('line', { y1: h, y2: H - b, style: 'stroke:var(--cds-chart-reference)', 'stroke-dasharray': '3 3', opacity: 0 });
    const cible = GB.sv('rect', { x: g - 20, y: 0, width: W - g - d + 40, height: H - b, fill: 'transparent' });
    cible.addEventListener('pointermove', (ev) => {
      const r = svg.getBoundingClientRect(), px = ((ev.clientX - r.left) / r.width) * W;
      let i = 0;
      K.forEach((k, j) => { if (Math.abs(xk(String(k)) - px) < Math.abs(xk(String(K[i])) - px)) i = j; });
      curseur.setAttribute('x1', xk(String(K[i]))); curseur.setAttribute('x2', xk(String(K[i]))); curseur.setAttribute('opacity', 1);
      const lignes = series.map((x) => [x, x.v[i]]).filter((z) => z[1] != null).sort((a, b) => b[1] - a[1]);
      GB.bulle(ev, GB.contenu(`k = ${KN[i]}`, lignes.map(([x, y]) => [coulH(x.hd), `${nomSerie(x)} : ${GB.fr(y)}`])));
    });
    cible.addEventListener('pointerleave', () => { GB.cacher(); curseur.setAttribute('opacity', 0); });
    svg.append(curseur, cible);
  }
  dash.onData(controles);
  GB.dessins.amorcage = [controles];
}
