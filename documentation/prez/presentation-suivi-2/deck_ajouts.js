// Deux diapos à copier dans la présentation de suivi n° 2 : jeu de données (barres par site,
// CDR et Patawa rangés sous Kaw, RNRT = Trésor) et pré-benchmark AnuraSet (benchmark 08).
//   node documentation/prez/presentation-suivi-2/deck_ajouts.js [sortie.pptx]
// Même charte que deck.js. Barres dessinées en formes : lisibles dans tous les lecteurs.

const path = require("path");
const pptxgen = require("pptxgenjs");

const HERE = __dirname;
const OUT = process.argv[2] || path.join(HERE, "..", "presentation_suivi_2_ajouts.pptx");

const C = {
  dark: "0E2A24", dark2: "173D34", ink: "1B2A24", muted: "5B6B63", sage: "A7BFB2",
  pale: "E4EEE8", paler: "F3F7F4", white: "FFFFFF", gold: "E9B44C", goldInk: "8A5A00",
  teal: "2A7F72", terra: "C8553D", line: "CFDDD5", blue: "3E7CB1", green: "2E8B57", grey: "B8C4BD",
};
const HF = "Cambria", BF = "Calibri";
const W = 13.333, H = 7.5, MX = 0.6, CW = W - 2 * MX;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "Suivi n° 2 — diapos ajoutées";
pres.author = "Léonard Laplace-Palette";

function tx(slide, text, o) {
  slide.addText(text, Object.assign({ isTextBox: true, fontFace: BF, color: C.ink, fontSize: 14, margin: 0, valign: "top" }, o));
}
function card(slide, x, y, w, h, fill) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill, width: 0 }, rectRadius: 0.1 });
}
function bar(slide, x, y, w, h, color) {
  if (w > 0.005) slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color }, line: { color, width: 0 } });
}
function rich(parts) {
  return parts.map(([t, b, extra]) => ({ text: t, options: Object.assign({ bold: !!b }, extra || {}) }));
}
function content(rubrique, title, notes) {
  const s = pres.addSlide();
  s.background = { color: C.white };
  const wChip = 0.2 + rubrique.length * 0.085;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: 0.4, w: wChip, h: 0.32, fill: { color: C.pale }, line: { color: C.pale, width: 0 }, rectRadius: 0.16 });
  tx(s, rubrique.toUpperCase(), { x: MX, y: 0.4, w: wChip, h: 0.32, fontSize: 10, bold: true, color: C.teal, align: "center", valign: "middle", charSpacing: 1 });
  tx(s, title, { x: MX, y: 0.82, w: CW, h: 0.7, fontFace: HF, fontSize: 28, bold: true, valign: "middle" });
  s.addNotes(notes);
  return s;
}

// =====================================================================
// Jeu de données : heures par site, deux campagnes empilées
// =====================================================================
{
  const s = content("Données · 1/2", "Le jeu de données : 96 292 enregistrements, 4 sites",
    "L'inventaire est complet depuis le 29 septembre : 96 292 enregistrements de 2 min, 3 204 h, 2,2 To, sur quatre sites. Deux campagnes. La phénologie 2023-2024 : Kaw, Molokoï et Trésor, deux enregistreurs par site, un an de décembre 2023 à novembre 2024, 2 225 h. La campagne 2026 : une semaine par relevé, à Mataroni, à Kaw (relevés CDR, Patawa Est et Patawa Ouest) et à Trésor (relevé RNRT, la réserve naturelle régionale Trésor), 979 h. Tous les enregistrements font 2 min, toutes les 30 min de 5 h à 20 h, en 48 kHz stéréo. Après les drapeaux, 94 588 enregistrements restent encodables, soit 4,5 millions de fenêtres de 5 s au pas de 2,5 s. Deux conséquences : Mataroni n'a qu'une semaine de 2026, et seul le jeu 2023 couvre toutes les saisons ; les 345 positifs et 150 négatifs de Blancinet sortent de l'entraînement et de l'évaluation.");
  // heures (README, inventaire du 29/09) ; Kaw 2026 = CDR + Patawa Est + Patawa Ouest ; Trésor 2026 = RNRT
  const sites = [
    ["Kaw", 755.0, 179.7 + 87.4 + 36.6, "2026 : relevés CDR, Patawa Est, Patawa Ouest"],
    ["Trésor", 770.2, 242.7, "2026 : relevé RNRT"],
    ["Molokoï", 700.0, 0, ""],
    ["Mataroni", 0, 432.5, ""],
  ];
  tx(s, "Heures d'audio par site (enregistrements de 2 min)", { x: MX, y: 1.75, w: 7.4, h: 0.35, fontSize: 14, bold: true });
  const x0 = MX + 1.3, maxW = 5.0, maxV = 1100, rowH = 0.95, y0 = 2.35, bh = 0.5;
  sites.forEach(([n, a, b, note], i) => {
    const y = y0 + i * rowH, wa = (a / maxV) * maxW, wb = (b / maxV) * maxW;
    tx(s, n, { x: MX, y, w: 1.15, h: bh, fontSize: 15, bold: true, align: "right", valign: "middle" });
    bar(s, x0, y, wa, bh, C.blue);
    bar(s, x0 + wa + (wa > 0 ? 0.03 : 0), y, wb, bh, C.green);
    if (a) tx(s, `${Math.round(a)} h`, { x: x0, y, w: wa, h: bh, fontSize: 12, bold: true, color: C.white, align: "center", valign: "middle" });
    if (b) tx(s, `${Math.round(b)} h`, { x: x0 + wa + 0.03, y, w: Math.max(wb, 0.6), h: bh, fontSize: 12, bold: true, color: C.white, align: "center", valign: "middle" });
    tx(s, `${Math.round(a + b)} h`, { x: x0 + wa + wb + 0.12, y, w: 0.9, h: bh, fontSize: 13, color: C.muted, valign: "middle" });
    if (note) tx(s, note, { x: x0, y: y + bh + 0.03, w: 5.5, h: 0.28, fontSize: 10.5, italic: true, color: C.muted });
  });
  const ly = y0 + sites.length * rowH + 0.05;
  [[C.blue, "2023-2024 · phénologie (1 an)"], [C.green, "2026 · campagne (1 semaine par relevé)"]].forEach(([col, t], i) => {
    const x = x0 + i * 2.9;
    bar(s, x, ly + 0.08, 0.22, 0.22, col);
    tx(s, t, { x: x + 0.32, y: ly, w: 2.6, h: 0.38, fontSize: 12, valign: "middle" });
  });
  [["3 204 h", "d'audio, 2,2 To"], ["94 588", "enregistrements encodables"], ["4,5 M", "fenêtres de 5 s (pas 2,5 s)"]].forEach(([n, l], i) => {
    const y = 1.75 + i * 1.18;
    card(s, 8.3, y, 4.4, 1.02, C.paler);
    tx(s, n, { x: 8.5, y: y + 0.1, w: 2.0, h: 0.8, fontFace: HF, fontSize: 26, bold: true, color: C.goldInk, valign: "middle" });
    tx(s, l, { x: 10.5, y: y + 0.1, w: 2.1, h: 0.8, fontSize: 13, color: C.muted, valign: "middle" });
  });
  card(s, 8.3, 5.35, 4.4, 1.55, C.dark);
  tx(s, [
    { text: "Mataroni : une semaine, en 2026 seulement", options: { bullet: { indent: 14 }, breakLine: true, paraSpaceAfter: 8 } },
    { text: "Labels Blancinet (345 + 150) : sortis", options: { bullet: { indent: 14 }, bold: true, color: C.gold } },
  ], { x: 8.5, y: 5.5, w: 4.05, h: 1.3, fontSize: 13.5, color: C.white });
}

// =====================================================================
// Pré-benchmark AnuraSet (benchmark 08)
// =====================================================================
{
  const s = content("Pré-benchmark AnuraSet", "23 encodeurs sur des sites jamais vus : perch_v2, meilleur libre",
    "Protocole. AnuraSet : 1 612 enregistrements d'une minute, 4 sites du Brésil ; on garde les 5 espèces d'anoures présentes sur au moins deux sites. Pour chaque encodeur, la tête apprend sur les autres sites et on juge le site tenu à l'écart : la mesure est l'AP moyenne par site, au niveau de la minute, moyennée sur les 5 espèces. Chaque encodeur est jugé avec la tête qui correspond à sa sortie : la logistique sur l'embedding, et pour les encodeurs qui fournissent leurs jetons, une sonde à prototypes ou une sonde attentive sur les jetons. Sur le graphique, la partie claire est la logistique, la partie foncée ce que la meilleure tête ajoute. Résultats. En tête, sans écart significatif entre eux après correction de Holm : naturebeats avec la sonde à prototypes (0,84), perch_v2 avec la sonde à prototypes (0,81), convnext_birdset en logistique (0,81), puis perch_v2 en logistique (0,79), la référence. Le meilleur encodeur libre est perch_v2 : aucun encodeur non libre ne le bat après Holm. naturebeats est sous licence non commerciale, convnext_birdset n'a aucune licence déclarée. Ce que ça change. Un : la tête doit suivre la sortie de l'encodeur. Les transformers auto-supervisés jugés en logistique semblaient mauvais : Bird-MAE-Base passe de 0,51 à 0,74 et naturebeats de 0,69 à 0,84 avec la sonde à prototypes sur leurs jetons. Deux : la sonde attentive, elle, n'aide presque jamais : une seule requête apprise sur peu de positifs se cale sans doute sur un indice de site. C'est l'inconvénient annoncé dans la partie attention, et le notebook 04 permettra de le voir. Trois : sur un site sans aucune annotation, convnext_birdset part le mieux (0,83), perch_v2 juste derrière (0,81). Quatre : BirdNET 3 fait 0,74, mieux que BirdNET 2.4 mais sous perch_v2. Pour A. blanci : perch_v2, avec la logistique et la sonde à prototypes, entre au benchmark ONF ; naturebeats et convnext_birdset sont proposés comme objectifs non libres, à décider avec Élodie. Limites : d'autres anoures qu'A. blanci, 2 à 4 sites par espèce, meilleure tête choisie après coup, trois encodeurs de la vague 2 encore à ajouter.");
  // donnees/meilleures.csv (benchmark 08), 12 premiers ; libre : encodeurs.csv (n° 156)
  const rows = [
    ["naturebeats", 0.694, 0.842, "sonde à prototypes", "non"],
    ["perch_v2", 0.788, 0.813, "sonde à prototypes", "oui"],
    ["convnext_birdset", 0.812, 0.812, "logistique", "non"],
    ["esp-aves2 sl_beats_bio", 0.760, 0.793, "sonde à prototypes", "non"],
    ["esp-aves2 sl_beats_all", 0.728, 0.782, "sonde à prototypes", "non"],
    ["perch_bird", 0.782, 0.782, "logistique", "sous réserve"],
    ["birdmae_base", 0.505, 0.738, "sonde à prototypes", "non"],
    ["birdnet_v3", 0.734, 0.736, "logistique", "oui"],
    ["esp-aves2 sl_eat_bio", 0.703, 0.703, "logistique", "non"],
    ["birdmae_large", 0.497, 0.690, "sonde à prototypes", "non"],
    ["birdnet 2.4", 0.685, 0.688, "logistique", "non"],
    ["esp-aves2 sl_eat_all", 0.599, 0.602, "LDA rétrécie", "non"],
  ];
  tx(s, "AP moyenne par site tenu à l'écart (minute, 5 espèces)", { x: MX, y: 1.65, w: 7.6, h: 0.32, fontSize: 13, bold: true });
  const lx = MX + 2.05, maxW = 4.6, rh = 0.33, y0 = 2.08;
  // grille 0,2 ; 0,4 ; 0,6 ; 0,8 ; 1
  [0.2, 0.4, 0.6, 0.8, 1.0].forEach((v) => {
    const x = lx + v * maxW;
    s.addShape(pres.shapes.LINE, { x, y: y0 - 0.05, w: 0, h: rows.length * rh + 0.05, line: { color: "E3E8E5", width: 0.75 } });
    tx(s, v.toFixed(1).replace(".", ","), { x: x - 0.3, y: y0 + rows.length * rh + 0.02, w: 0.6, h: 0.25, fontSize: 9.5, color: C.muted, align: "center" });
  });
  rows.forEach(([n, lo, best, head, libre], i) => {
    const y = y0 + i * rh, free = libre === "oui", hot = n === "perch_v2";
    tx(s, n, { x: MX, y, w: 1.95, h: rh, fontSize: 11.5, bold: hot, align: "right", valign: "middle", color: hot ? C.ink : C.ink });
    bar(s, lx, y + 0.06, lo * maxW, rh - 0.12, free ? "8CC7A1" : C.grey);
    bar(s, lx + lo * maxW, y + 0.06, (best - lo) * maxW, rh - 0.12, free ? C.green : C.muted);
    tx(s, `${best.toFixed(2).replace(".", ",")}${best - lo > 0.005 ? " · " + head : ""}`, { x: lx + best * maxW + 0.08, y, w: 2.2, h: rh, fontSize: 10, color: C.muted, valign: "middle" });
  });
  const ly = y0 + rows.length * rh + 0.32;
  [["8CC7A1", "libre : logistique"], [C.green, "libre : gain de la tête adaptée"], [C.grey, "non libre : logistique"], [C.muted, "non libre : gain de la tête adaptée"]].forEach(([col, t], i) => {
    const x = MX + i * 1.95, y = ly;
    bar(s, x + 0.2, y + 0.06, 0.2, 0.18, col);
    tx(s, t, { x: x + 0.45, y, w: 1.55, h: 0.42, fontSize: 9.5, valign: "middle" });
  });
  tx(s, "11 autres encodeurs entre 0,40 et 0,59 · aucun écart significatif entre les quatre premiers (Holm)", { x: MX, y: ly + 0.5, w: 7.6, h: 0.28, fontSize: 10, italic: true, color: C.muted });
  // à droite : quatre enseignements
  const lessons = [
    ["Meilleur libre : perch_v2", "0,81 avec la sonde à prototypes ; aucun non libre ne le bat après Holm.", C.dark, C.gold, C.white],
    ["La tête doit suivre la sortie", "Bird-MAE-Base 0,51 → 0,74, naturebeats 0,69 → 0,84 avec la sonde à prototypes sur leurs jetons.", C.paler, C.teal, C.ink],
    ["La sonde attentive n'aide presque jamais", "une seule requête, peu de positifs : elle se cale sans doute sur le site.", C.paler, C.terra, C.ink],
    ["Site sans annotation", "convnext_birdset part le mieux (0,83), perch_v2 juste derrière (0,81).", C.paler, C.teal, C.ink],
  ];
  lessons.forEach(([t, d, bg, tc, dc], i) => {
    const y = 1.65 + i * 1.32;
    card(s, 8.55, y, 4.18, 1.18, bg);
    tx(s, t, { x: 8.75, y: y + 0.1, w: 3.85, h: 0.35, fontSize: 14, bold: true, color: tc });
    tx(s, d, { x: 8.75, y: y + 0.48, w: 3.85, h: 0.65, fontSize: 11.5, color: dc });
  });
  tx(s, "AnuraSet : 1 612 min, 4 sites du Brésil, 5 espèces sur ≥ 2 sites · un site tenu à l'écart à la fois · benchmark 08, statut indicateur", { x: MX, y: 7.0, w: CW, h: 0.28, fontSize: 10, italic: true, color: C.muted });
}

pres.writeFile({ fileName: OUT }).then((f) => console.log("écrit :", f));
