/* Rapports de benchmark, tableaux de référence, bibliographie. */
{
  const lien = (href, texte) => (/^https?:\/\//.test(href || '') ? GB.el('a', { href, target: '_blank', rel: 'noopener' }, texte) : texte);

  const rapports = document.getElementById('rapports-table');
  dash.onData(() => GB.table(rapports, 'rapports', () => GB.rows('rapports').map((r) => GB.el('tr', {},
    GB.td(r.numero, 'gb-num'), GB.td(r.titre), GB.td(GB.date(r.date)), GB.td(r.statut),
    GB.td(GB.entier(r.figures), 'n'), GB.td(GB.entier(r.donnees), 'n'),
    GB.el('td', {}, GB.el('span', { class: 'gb-mono' }, lien(r.lien, r.dossier)))))));

  const tableaux = document.getElementById('tableaux-table');
  dash.onData(() => GB.table(tableaux, 'tableaux', () => GB.rows('tableaux').map((t) => GB.el('tr', { 'data-key': t.fichier },
    GB.el('td', {}, GB.el('strong', {}, lien(t.lien, t.fichier))), GB.td(t.description)))));

  // Bibliographie : sections, recherche dans le titre et le texte.
  const liste = document.getElementById('biblio-liste'), q = document.getElementById('biblio-q'), sel = document.getElementById('biblio-section'), nb = document.getElementById('biblio-nb');
  let sections = '';
  function biblio() {
    if (!GB.pret(liste, 'biblio')) { nb.textContent = ''; return; }
    const rows = GB.rows('biblio');
    const noms = [...new Set(rows.map((r) => r.section))];
    if (sections !== noms.join('|')) {
      sections = noms.join('|');
      const avant = sel.value;
      sel.replaceChildren(GB.el('option', { value: '' }, 'Toutes'), ...noms.map((s) => GB.el('option', { value: s }, s)));
      sel.value = noms.includes(avant) ? avant : '';
    }
    const mot = q.value.trim().toLowerCase();
    const garde = rows.filter((r) => (!sel.value || r.section === sel.value) && (!mot || `${r.titre} ${r.texte} ${r.liens}`.toLowerCase().includes(mot)));
    nb.textContent = `${garde.length} référence${garde.length > 1 ? 's' : ''} sur ${rows.length}`;
    if (!garde.length) { liste.replaceChildren(GB.el('p', { class: 'gb-vide' }, 'Aucune référence ne correspond.')); return; }
    liste.replaceChildren(...d3.groups(garde, (r) => r.section).map(([section, refs]) => GB.el('section', { style: 'display:grid;gap:10px' },
      GB.el('h3', {}, section),
      GB.el('ul', {}, refs.map((r) => {
        const urls = String(r.liens || '').split(' ').filter((u) => /^https?:\/\//.test(u));
        return GB.el('li', {},
          r.titre ? GB.el('strong', {}, r.titre) : null,
          r.texte ? GB.el('p', {}, r.texte) : null,
          urls.length ? GB.el('div', { class: 'gb-liens-ref' }, urls.map((u) => lien(u, u.replace(/^https?:\/\/(www\.)?/, '').replace(/\/$/, '').slice(0, 60)))) : null);
      })))));
  }
  q.addEventListener('input', biblio);
  sel.addEventListener('change', biblio);
  dash.onData(biblio);
}
