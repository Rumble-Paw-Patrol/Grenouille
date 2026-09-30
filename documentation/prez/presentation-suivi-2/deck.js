// Présentation de suivi n° 2 — généré par pptxgenjs.
//   node documentation/presentation-suivi-2/deck.js [sortie.pptx]
// Écrit aussi script.md (notes orateur de chaque diapo) à côté de ce fichier.
// Figures : generer_figures.py. Charte : voisine de presentation_point_etape.pptx (même
// famille de verts, Cambria + Calibri), avec un accent or et des pastilles de rubrique.

const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const HERE = __dirname;
const FIG = (f) => path.join(HERE, "figures", f);
const OUT = process.argv[2] || path.join(HERE, "..", "presentation_suivi_2.pptx");

// ---------- charte ----------
const C = {
  dark: "0E2A24", dark2: "173D34", ink: "1B2A24", muted: "5B6B63", sage: "A7BFB2",
  pale: "E4EEE8", paler: "F3F7F4", white: "FFFFFF", gold: "E9B44C", goldInk: "8A5A00",
  teal: "2A7F72", terra: "C8553D", line: "CFDDD5",
};
const CHART = ["2E8B57", "D9822B", "3E7CB1", "C0492F", "8A6BBE"]; // validée (dataviz)
const HF = "Cambria", BF = "Calibri";
const W = 13.333, H = 7.5, MX = 0.6, CW = W - 2 * MX;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "Détection acoustique d'Anomaloglossus blanci — suivi n° 2";
pres.author = "Léonard";

const script = []; // [titre, notes]
let pageNo = 0;

// ---------- aides ----------
function tx(slide, text, o) {
  slide.addText(text, Object.assign({ isTextBox: true, fontFace: BF, color: C.ink, fontSize: 14, margin: 0, valign: "top" }, o));
}
function card(slide, x, y, w, h, fill, line) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, fill: { color: fill || C.paler }, line: { color: line || fill || C.paler, width: line ? 1 : 0 }, rectRadius: 0.1,
  });
}
function dot(slide, x, y, n, fill, size = 0.44) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: size, h: size, fill: { color: fill || C.dark }, line: { color: fill || C.dark, width: 0 } });
  tx(slide, String(n), { x, y, w: size, h: size, fontSize: 14, bold: true, color: C.white, align: "center", valign: "middle" });
}
function arrow(slide, x, y, w) {
  slide.addShape(pres.shapes.LINE, { x, y, w, h: 0, line: { color: C.sage, width: 1.5, endArrowType: "triangle" } });
}
function notes(slide, title, n) {
  slide.addNotes(n);
  script.push([title, n]);
}
function content(rubrique, title, n) {
  const s = pres.addSlide();
  s.background = { color: C.white };
  pageNo += 1;
  // pastille de rubrique (motif de la présentation n° 2)
  const wChip = 0.2 + rubrique.length * 0.085;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: 0.4, w: wChip, h: 0.32, fill: { color: C.pale }, line: { color: C.pale, width: 0 }, rectRadius: 0.16 });
  tx(s, rubrique.toUpperCase(), { x: MX, y: 0.4, w: wChip, h: 0.32, fontSize: 10, bold: true, color: C.teal, align: "center", valign: "middle", charSpacing: 1 });
  tx(s, title, { x: MX, y: 0.82, w: CW, h: 0.7, fontFace: HF, fontSize: 28, bold: true, color: C.ink, valign: "middle" });
  tx(s, String(pageNo + 1), { x: W - 1.0, y: H - 0.42, w: 0.5, h: 0.25, fontSize: 9, color: C.sage, align: "right" });
  notes(s, `${pageNo + 1}. ${title}`, n);
  return s;
}
function bullets(items, o = {}) {
  return items.map((it, i) => {
    const last = i === items.length - 1;
    const base = { bullet: { indent: 14 }, breakLine: !last, paraSpaceAfter: o.gap || 6 };
    if (typeof it === "string") return { text: it, options: base };
    return { text: it.t, options: Object.assign(base, it.o || {}) };
  });
}
function rich(parts) {
  return parts.map(([t, b, extra]) => ({ text: t, options: Object.assign({ bold: !!b }, extra || {}) }));
}
function table(slide, rows, o) {
  const head = rows[0].map((h) => ({ text: h, options: { bold: true, color: C.white, fill: { color: C.dark }, fontSize: o.fs || 12 } }));
  const body = rows.slice(1).map((r, i) => r.map((c) => {
    const cell = typeof c === "object" && c !== null && c.text !== undefined ? c : { text: String(c) };
    cell.options = Object.assign({ fontSize: o.fs || 12, color: C.ink, fill: { color: i % 2 ? C.white : C.paler } }, cell.options || {});
    return cell;
  }));
  slide.addTable([head, ...body], Object.assign({ fontFace: BF, border: { type: "solid", color: C.line, pt: 0.5 }, margin: 0.07, valign: "middle" }, o));
}

// =====================================================================
// 1. Titre
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.dark };
  s.addImage({ path: FIG("titre.png"), x: 8.6, y: 0, w: 4.733, h: 7.5, sizing: { type: "cover", w: 4.733, h: 7.5 } });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: 1.25, w: 3.9, h: 0.36, fill: { color: C.dark2 }, line: { color: C.dark2, width: 0 }, rectRadius: 0.18 });
  tx(s, "STAGE ONF GUYANE · SUIVI N° 2", { x: MX, y: 1.25, w: 3.9, h: 0.36, fontSize: 11, bold: true, color: C.gold, align: "center", valign: "middle", charSpacing: 1 });
  tx(s, [
    { text: "Détection acoustique d'", options: { breakLine: true } },
    { text: "Anomaloglossus blanci", options: { italic: true } },
  ], { x: MX, y: 1.9, w: 7.8, h: 1.8, fontFace: HF, fontSize: 42, bold: true, color: C.white });
  tx(s, "Attention · état de l'art · BirdCLEF+ 2026 · données et annotation · régularisations", { x: MX, y: 3.85, w: 7.6, h: 0.7, fontSize: 17, color: C.sage });
  tx(s, "Léonard · 30 septembre 2026", { x: MX, y: 4.75, w: 6, h: 0.35, fontSize: 13, color: C.sage });
  tx(s, "Fond : un chant d'A. blanci enregistré à Mataroni (spectrogramme, 1–9 kHz)", { x: 8.75, y: 7.05, w: 4.4, h: 0.3, fontSize: 9, color: C.white, italic: true });
  notes(s, "1. Titre", "Deuxième point de suivi. Plan : un cours sur l'attention en trois diapos, un état de l'art condensé, BirdCLEF+ 2026, l'ouverture du GitHub, le jeu de données et la nouvelle stratégie d'annotation décidée avec Élodie, les régularisations, et ce que je fais d'ici la prochaine réunion. La partie pré-benchmark AnuraSet est en cours de mise à jour. L'image de fond est un vrai chant d'A. blanci enregistré à Mataroni : les petites notes vers 4,5 kHz.");
}

// =====================================================================
// 2–4. Attention
// =====================================================================
{
  const s = content("Attention · 1/3", "Le mécanisme interne : requêtes, clés, valeurs",
    "Un encodeur découpe le spectrogramme d'une fenêtre en morceaux : les jetons. perch_v2 en rend 16 × 4 = 64 pour une fenêtre de 5 s. Chaque jeton x_i est projeté trois fois par des matrices apprises. La clé k_i dit « ce que je contiens », la valeur v_i « ce que je transmets », la requête q « ce que je cherche ». Étape 1 : le score d'un jeton est le produit scalaire q · k_i, divisé par la racine de la dimension d pour que les scores ne grandissent pas avec la dimension (sinon le softmax sature). Étape 2 : le softmax transforme les scores en poids positifs de somme 1. Étape 3 : la sortie est la somme des valeurs pondérées. Analogie : une bibliothèque. La requête est ma question, les clés sont les titres des livres, les valeurs leur contenu ; je lis surtout les livres dont le titre répond à ma question. Deux usages : dans un transformer, c'est de l'auto-attention, chaque jeton émet sa propre requête, 64 requêtes et 64 sorties : les jetons se « parlent ». Dans notre sonde attentive, l'encodeur est gelé et on n'apprend qu'une seule requête, « où est A. blanci ? », plus une couche linéaire : une sortie par fenêtre.");
  const x0 = MX, y0 = 1.85;
  tx(s, "64 jetons d'une fenêtre", { x: x0, y: y0, w: 2.2, h: 0.3, fontSize: 12, bold: true, color: C.muted });
  const lab = ["x₁", "x₂", "x₃", "x₄", "x₆₄"];
  lab.forEach((l, i) => {
    const hot = i === 2;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x0 + 0.2, y: y0 + 0.4 + i * 0.62, w: 1.4, h: 0.48, fill: { color: hot ? C.gold : C.pale }, line: { color: hot ? C.gold : C.line, width: 0.75 }, rectRadius: 0.08 });
    tx(s, l, { x: x0 + 0.2, y: y0 + 0.4 + i * 0.62, w: 1.4, h: 0.48, fontSize: 14, align: "center", valign: "middle", bold: hot });
  });
  tx(s, "x₃ : la note", { x: x0 + 0.2, y: y0 + 3.55, w: 1.6, h: 0.3, fontSize: 11, color: C.goldInk, italic: true });
  const px = 2.75;
  [["W_Q", "requête q", "ce que je cherche", C.goldInk], ["W_K", "clés kᵢ", "ce que je contiens", C.teal], ["W_V", "valeurs vᵢ", "ce que je transmets", "3E7CB1"]].forEach(([m, n, d, col], i) => {
    const y = y0 + 0.35 + i * 1.1;
    s.addShape(pres.shapes.LINE, { x: x0 + 1.65, y: y0 + 1.9, w: px - x0 - 1.65, h: y + 0.42 - (y0 + 1.9), line: { color: C.sage, width: 1, endArrowType: "triangle" } });
    card(s, px, y, 2.6, 0.84, C.white, col);
    tx(s, [{ text: `${m} → ${n}`, options: { bold: true, color: col, breakLine: true } }, { text: d, options: { color: C.muted, fontSize: 11 } }], { x: px + 0.15, y: y + 0.1, w: 2.35, h: 0.7, fontSize: 14 });
  });
  arrow(s, 5.3, 3.13, 0.4);
  [["1 · scores", "sᵢ = q · kᵢ / √d", "le jeton répond-il à la question ?"],
   ["2 · softmax", "αᵢ = e^sᵢ / Σⱼ e^sⱼ", "poids ≥ 0, de somme 1"],
   ["3 · somme pondérée", "z = Σᵢ αᵢ vᵢ", "un vecteur par fenêtre → score"]].forEach(([t, f, d], i) => {
    const x = 5.75 + i * 2.35, y = 2.25, hot = i === 2;
    card(s, x, y, 2.1, 1.75, hot ? C.dark : C.paler);
    tx(s, t, { x: x + 0.15, y: y + 0.15, w: 1.85, h: 0.3, fontSize: 12, bold: true, color: hot ? C.gold : C.goldInk });
    tx(s, f, { x: x + 0.15, y: y + 0.52, w: 1.95, h: 0.45, fontSize: 15, bold: true, fontFace: HF, color: hot ? C.white : C.ink });
    tx(s, d, { x: x + 0.15, y: y + 1.05, w: 1.95, h: 0.6, fontSize: 11.5, color: hot ? C.sage : C.muted });
    if (i < 2) arrow(s, x + 2.1, y + 0.88, 0.25);
  });
  card(s, 5.75, 4.3, 6.95, 0.75, C.pale);
  tx(s, rich([["En une ligne : ", true], ["Attention(Q, K, V) = softmax(Q Kᵀ / √d) · V", false, { fontFace: HF }], ["   Vaswani et al. 2017", false, { color: C.muted, fontSize: 11 }]]), { x: 6.05, y: 4.3, w: 6.5, h: 0.75, fontSize: 15, valign: "middle" });
  card(s, MX, 5.8, 6.0, 1.05, C.paler);
  tx(s, rich([["Auto-attention (transformer) : ", true], ["chaque jeton émet sa requête ; les jetons se « parlent ».", false, { color: C.muted }]]), { x: MX + 0.25, y: 5.8, w: 5.6, h: 1.05, fontSize: 14, valign: "middle" });
  card(s, 6.85, 5.8, 5.85, 1.05, C.paler);
  tx(s, rich([["Sonde attentive (notre tête) : ", true], ["une seule requête apprise, encodeur gelé.", false, { color: C.muted }]]), { x: 7.1, y: 5.8, w: 5.45, h: 1.05, fontSize: 14, valign: "middle" });
}

{
  const s = content("Attention · 2/3", "L'effet : pondérer les jetons au lieu de les moyenner",
    "En haut, une vraie fenêtre de 5 s de Mataroni découpée en 64 jetons. À droite, des poids d'attention en illustration : ils ne viennent pas d'un modèle, je les ai calculés à la main sur la nouveauté de chaque morceau, pour montrer l'idée. Les jetons qui portent les notes d'A. blanci, vers 4,5 kHz, prennent le poids ; la bande d'insectes à 8 kHz, constante, n'en prend pas. Le problème que ça résout : avec la moyenne, chaque jeton pèse 1/64, et la note, qui occupe 2 à 4 jetons, est noyée dans le fond. Le calcul à la main du graphique : quatre jetons, une requête qui cherche l'énergie dans la bande de la note. Le jeton de la note reçoit 0,66 du poids, contre 0,25 avec la moyenne (0,66 = e^2,12 / (e^0,14 + e^0,28 + e^2,12 + e^0,57)). Entre les deux extrêmes, la moyenne et le maximum, l'attention choisit les poids selon le contenu de la fenêtre ; GeM est une version sans requête, avec un seul paramètre p qui glisse de la moyenne (p = 1) au maximum (p infini).");
  s.addImage({ path: FIG("patchs_attention.png"), x: MX + 0.55, y: 1.6, w: 11.0, h: 11.0 * 680 / 2200 });
  s.addChart(pres.charts.BAR, [
    { name: "attention", labels: ["x₁ fond", "x₂ fond", "x₃ note", "x₄ oiseau"], values: [0.09, 0.11, 0.66, 0.14] },
    { name: "moyenne", labels: ["x₁ fond", "x₂ fond", "x₃ note", "x₄ oiseau"], values: [0.25, 0.25, 0.25, 0.25] },
  ], {
    x: MX, y: 5.15, w: 5.4, h: 1.8, barDir: "col", barGapWidthPct: 60,
    chartColors: [CHART[1], "B8C4BD"], showLegend: true, legendPos: "r", legendFontSize: 11, legendColor: C.muted,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelColor: C.ink, dataLabelFormatCode: "0.00",
    valAxisHidden: true, valGridLine: { style: "none" }, catAxisLabelColor: C.muted, catAxisLabelFontSize: 10,
    valAxisMaxVal: 0.85, valAxisMinVal: 0, showTitle: false,
  });
  const tri = [["Moyenne", "chaque jeton pèse 1/64 : la note est noyée", C.paler, C.ink], ["Maximum", "un seul jeton : sensible à un clic", C.paler, C.ink], ["Attention", "poids appris, selon le contenu", C.dark, C.gold]];
  tri.forEach(([t, d, bg, tc], i) => {
    const x = 6.3 + i * 2.17;
    card(s, x, 5.2, 2.02, 1.65, bg);
    tx(s, t, { x: x + 0.15, y: 5.32, w: 1.75, h: 0.4, fontSize: 16, bold: true, color: tc });
    tx(s, d, { x: x + 0.15, y: 5.75, w: 1.75, h: 1.0, fontSize: 12.5, color: bg === C.dark ? C.white : C.muted });
  });
}

{
  const s = content("Attention · 3/3", "Dans notre projet : avantages et inconvénients",
    "Avantages. Un : la note d'A. blanci dure 0,09 s et occupe 2 à 4 jetons sur 64 ; l'attention est faite pour ce cas. Deux : les transformers, Bird-MAE et BEATs, n'ont pas appris à résumer leur fenêtre en un vecteur ; ils ne donnent leur mesure qu'avec une tête sur leurs jetons. La revue de Schwinger et al. le mesure : BEATs passe de 94,10 à 97,98 AUROC sur BEANS en changeant seulement de tête. Trois : la carte des poids se lit, elle montre où le modèle a entendu, utile pour vérifier une détection. Inconvénients. Un : il faut stocker les jetons, 64 vecteurs par fenêtre au lieu d'un, sur 4,5 millions de fenêtres. Deux : une tête de plus à apprendre avec peu de positifs, donc un risque de sur-apprentissage. Trois : le gain n'est pas garanti ; dans BirdCLEF+ 2026, l'équipe DS@GT a perdu 0,013 à 0,035 point au classement avec une tête à attention. Quatre : parmi nos encodeurs, perch_v2 est le seul dont l'adaptateur sort déjà les jetons. Conclusion, conforme à la feuille de route V5 : la logistique reste la tête par défaut ; la sonde attentive est essayée dans le benchmark ONF pour un transformer libre, et gardée seulement si elle bat la logistique sur des points tenus à l'écart.");
  const colW2 = (CW - 0.3) / 2;
  const cols = [
    ["Avantages", C.teal, [
      ["Fait pour une note brève", "0,09 s : 2 à 4 jetons sur 64, noyés par la moyenne"],
      ["Indispensable aux transformers", "BEATs : 94,10 → 97,98 AUROC en passant à l'attentive"],
      ["Lisible", "la carte des poids montre où le modèle a entendu"],
    ]],
    ["Inconvénients", C.terra, [
      ["Mémoire", "64 jetons par fenêtre au lieu d'un vecteur, × 4,5 M fenêtres"],
      ["Sur-apprentissage", "une tête de plus à apprendre avec peu de positifs"],
      ["Gain incertain", "BirdCLEF+ 2026 (DS@GT) : −0,013 à −0,035 avec une tête à attention"],
    ]],
  ];
  cols.forEach(([t, col, items], c) => {
    const x = MX + c * (colW2 + 0.3);
    tx(s, t, { x, y: 1.7, w: colW2, h: 0.4, fontSize: 18, bold: true, color: col });
    items.forEach(([h, d], i) => {
      const y = 2.2 + i * 1.12;
      card(s, x, y, colW2, 0.98, C.paler);
      s.addShape(pres.shapes.OVAL, { x: x + 0.22, y: y + 0.33, w: 0.32, h: 0.32, fill: { color: col }, line: { color: col, width: 0 } });
      tx(s, c === 0 ? "+" : "−", { x: x + 0.22, y: y + 0.33, w: 0.32, h: 0.32, fontSize: 14, bold: true, color: C.white, align: "center", valign: "middle" });
      tx(s, h, { x: x + 0.72, y: y + 0.12, w: colW2 - 0.9, h: 0.35, fontSize: 15, bold: true });
      tx(s, d, { x: x + 0.72, y: y + 0.5, w: colW2 - 0.9, h: 0.4, fontSize: 12.5, color: C.muted });
    });
  });
  card(s, MX, 5.75, CW, 1.1, C.dark);
  tx(s, rich([["Verdict : ", true, { color: C.gold }], ["la logistique reste la tête par défaut. La sonde attentive est essayée pour un transformer libre, et gardée seulement si elle bat la logistique sur des points tenus à l'écart.", false, { color: C.white }]]), { x: MX + 0.3, y: 5.75, w: CW - 0.6, h: 1.1, fontSize: 15, valign: "middle" });
}

// =====================================================================
// 5. État de l'art (condensé)
// =====================================================================
{
  const s = content("État de l'art", "La bioacoustique en cinq étapes",
    "Script. Étape 1, avant 2018 : un détecteur par espèce, gabarits ou MFCC plus SVM, réglés à la main, une espèce à la fois. Étape 2, 2018–2021 : les réseaux convolutifs sur spectrogrammes, entraînés sur Xeno-canto ; BirdNET (Kahl et al. 2021), fenêtres de 3 s à 48 kHz, environ 6 000 espèces, devient le standard des études de terrain ; les concours BirdCLEF popularisent la tâche. Étape 3, 2021–2023 : on n'utilise plus la sortie des classifieurs mais leur avant-dernière couche, l'embedding, pour d'autres espèces ; Ghani et al. 2023 montrent que les embeddings d'oiseaux (Perch 1) transfèrent mieux que ceux de modèles audio généraux, y compris hors des oiseaux. Étape 4, 2022–2025 : l'apprentissage auto-supervisé et les transformers, AVES, BEATs, Bird-MAE, apprennent sans étiquettes, sur des jetons, avec de l'attention. Étape 5, 2025–2026 : les modèles de fondation multi-taxons, Perch 2.0 (Google DeepMind, 12 M de paramètres, 1,5 million d'enregistrements, le meilleur en sondage linéaire), NatureLM-audio (Earth Species Project), ESP-AVES2 (ICLR 2026 : mélanger audio général et bioacoustique aide), BirdNET+ V3 en préversion, MetaPerch (ICML 2026, les métadonnées comme pertes d'entraînement) ; et des outils pour les comparer : bacpipe, qui rassemble 26 encodeurs derrière une même interface, BEANS, BirdSet. Trois leçons de la revue de Schwinger et al. (Ecological Informatics 2026) : la tête doit suivre la sortie de l'encodeur ; plus gros n'est pas meilleur (8, 88 et 300 millions de paramètres font à peu près jeu égal sur BirdSet) ; le classement BEANS ne contient aucun anoure. Où se place le projet : à l'étape 3, avec un encodeur de l'étape 5 gelé et une petite tête apprise. Pas de fine-tuning : quelques centaines d'exemples ne contraignent pas des millions de paramètres.");
  const steps = [
    ["< 2018", "Détecteurs par espèce", "gabarits, MFCC + SVM"],
    ["2018–21", "CNN sur spectrogrammes", "BirdNET"],
    ["2021–23", "Embeddings réutilisés", "Perch 1, transfert"],
    ["2022–25", "Auto-supervisé", "BEATs, Bird-MAE"],
    ["2025–26", "Modèles de fondation", "Perch 2.0, NatureLM, ESP-AVES2, MetaPerch"],
  ];
  const y = 2.55, step = CW / 5;
  s.addShape(pres.shapes.LINE, { x: MX + 0.2, y, w: CW - 0.4, h: 0, line: { color: C.sage, width: 2 } });
  steps.forEach(([d, t, e], i) => {
    const x = MX + i * step, hot = i === 4;
    s.addShape(pres.shapes.OVAL, { x: x + step / 2 - 0.17, y: y - 0.17, w: 0.34, h: 0.34, fill: { color: hot ? C.gold : C.dark }, line: { color: C.white, width: 2 } });
    tx(s, d, { x, y: 1.85, w: step, h: 0.35, fontSize: 14, bold: true, color: hot ? C.goldInk : C.muted, align: "center" });
    card(s, x + 0.08, 3.0, step - 0.16, 1.9, hot ? C.dark : C.paler);
    tx(s, t, { x: x + 0.25, y: 3.18, w: step - 0.5, h: 0.75, fontSize: 16, bold: true, color: hot ? C.white : C.ink });
    tx(s, e, { x: x + 0.25, y: 3.98, w: step - 0.5, h: 0.8, fontSize: 13, color: hot ? C.sage : C.muted });
  });
  card(s, MX, 5.3, CW, 1.45, C.pale);
  tx(s, rich([["Nous : ", true], ["étape 3 avec un encodeur de l'étape 5, gelé, et une petite tête apprise.  ", false], ["Leçon : ", true], ["la tête doit suivre la sortie de l'encodeur ; plus gros n'est pas meilleur.", false]]), { x: MX + 0.3, y: 5.3, w: CW - 0.6, h: 1.45, fontSize: 16, valign: "middle" });
}

// =====================================================================
// 6–7. BirdCLEF+ 2026
// =====================================================================
{
  const s = content("BirdCLEF+ 2026 · 1/2", "Le concours : le Pantanal, et un classement serré",
    "BirdCLEF+ 2026, sur Kaggle, organisé par Cornell dans le cadre de LifeCLEF ; clôture le 27 mai 2026, working notes présentées à CLEF 2026 à Iéna du 21 au 24 septembre. Données : 344 h d'enregistrements focaux, 178 h de paysages sonores de 23 sites du Pantanal brésilien, dont une heure seulement étiquetée. 234 taxons : oiseaux, amphibiens, insectes, reptiles, mammifères. Métrique : ROC-AUC macro par fenêtre de 5 s. Contrainte : inférence sur CPU seul en 90 minutes, la même situation que les portables de l'ONF. 4 094 équipes. Podium au score privé : 1er Nikita Babych, 0,966. Sa méthode : distiller les embeddings de Perch v2 et d'AudioProtoPNet dans un EfficientNetV2, puis l'affiner et l'entraîner en Noisy Student itératif sur les paysages sonores non étiquetés. 2e tennogh, 0,960 : d'après le titre de sa note, un ensemble divers, des pseudo-labels et un spécialiste des taxons non-oiseaux. 3e kapenon, 0,960, et 4e une équipe, 0,959 : je n'ai pas lu leurs méthodes. 5e Jiacheng Ma, 0,958, code public. Les cinq premiers se tiennent en moins d'un centième. Leçons : Perch v2 est partout, comme enseignant, initialisation ou branche de l'ensemble ; les pseudo-labels sur l'audio non étiqueté font la différence (le 13e passe de 0,881 à 0,934 avec de bons pseudo-labels) ; et la validation hors ligne ne prédit plus le classement au-dessus de 0,92 (Lu et Tsai, 13e).");
  const facts = [["234", "taxons, dont amphibiens et insectes"], ["23", "sites du Pantanal (Brésil)"], ["CPU", "seul, 90 min : comme l'ONF"], ["4 094", "équipes"]];
  facts.forEach(([n, l], i) => {
    const y = 1.75 + i * 1.28;
    card(s, MX, y, 3.3, 1.12, i === 2 ? C.dark : C.paler);
    tx(s, n, { x: MX + 0.2, y: y + 0.08, w: 2.9, h: 0.6, fontFace: HF, fontSize: 28, bold: true, color: i === 2 ? C.gold : C.goldInk });
    tx(s, l, { x: MX + 0.2, y: y + 0.68, w: 2.9, h: 0.35, fontSize: 12.5, color: i === 2 ? C.white : C.muted });
  });
  table(s, [
    ["Rang", "Équipe", "Score privé", "Approche"],
    ["1", "Nikita Babych", "0,966", "distillation de Perch v2 + AudioProtoPNet, puis Noisy Student sur l'audio non étiqueté"],
    ["2", "tennogh", "0,960", "ensemble divers, pseudo-labels, spécialiste des non-oiseaux"],
    ["3", "kapenon", "0,960", "non lue"],
    ["4", "équipe « BirdCLEF+ 2026 »", "0,959", "non lue"],
    ["5", "Jiacheng Ma", "0,958", "code public"],
  ], { x: 4.25, y: 1.75, w: 8.48, colW: [0.65, 2.15, 1.15, 4.53], fs: 13, rowH: 0.55 });
  const les = [["Perch v2 partout", "enseignant, initialisation ou branche"], ["Pseudo-labels", "le 13e : 0,881 → 0,934"], ["Validation trompeuse", "au-dessus de 0,92, elle ne prédit plus le classement"]];
  les.forEach(([t, d], i) => {
    const x = 4.25 + i * 2.88;
    card(s, x, 5.35, 2.72, 1.5, C.pale);
    tx(s, t, { x: x + 0.2, y: 5.5, w: 2.35, h: 0.4, fontSize: 15, bold: true, color: C.teal });
    tx(s, d, { x: x + 0.2, y: 5.92, w: 2.35, h: 0.85, fontSize: 13, color: C.ink });
  });
}

{
  const s = content("BirdCLEF+ 2026 · 2/2", "Une équipe a fait notre prototype différentiel sur Perch v2",
    "Article : Miyaguchi, Gustineli et Cheung (Georgia Tech, équipe DS@GT), « Can Tokens Compete? », arXiv 2607.14474. Leur constat de départ est le nôtre : les amphibiens font 1,3 % des enregistrements focaux d'entraînement, mais 76,6 % des fenêtres de paysages sonores étiquetées. 72 des 234 taxons ne sont pas dans les classes de Perch. Pour eux, ils construisent une tête à prototypes sur les embeddings de Perch v2 gelé : μ+ est la moyenne des fenêtres où l'espèce est présente, μ− la moyenne de celles où elle est absente, et le score est cos(e, μ+) − cos(e, μ−). C'est notre prototype différentiel, en similarité cosinus. Résultat : sur les 66 paysages sonores étiquetés, AP macro de 0,895 sur les non-oiseaux, contre 0,074 pour leur réseau SED seul, et +0,011 au classement privé. Trois réserves : l'AP est mesurée sur les mêmes 66 fichiers qui construisent les prototypes, sans pli indiqué, donc optimiste ; en validation site tenu à l'écart (leur annexe), pour les espèces sans aucun enregistrement focal, la recherche par prototypes tombe au hasard (AUROC 0,26), et la régression ridge aussi (0,50) : sans exemple de l'espèce, aucune tête ne la trouve ; enfin leurs pseudo-labels et leur tête à attention ont fait baisser leur score. Ce que j'en retiens : Perch v2 porte de l'information sur les anoures même sans les avoir appris ; le prototype différentiel est un bon amorçage d'un nouveau site ; la logistique régularisée prend le relais dès qu'il y a des exemples, comme dans notre benchmark 07. Et les taxons se recouvrent en fréquence (1 à 4 kHz pour grenouilles, oiseaux et insectes) : une porte spectrale seule ne sépare pas les espèces.");
  card(s, MX, 1.75, 5.3, 5.1, C.dark);
  tx(s, "LEUR TÊTE À PROTOTYPES", { x: MX + 0.3, y: 1.95, w: 4.7, h: 0.3, fontSize: 11, bold: true, color: C.gold, charSpacing: 1 });
  tx(s, "score = cos(e, μ₊) − cos(e, μ₋)", { x: MX + 0.3, y: 2.35, w: 4.8, h: 0.55, fontFace: HF, fontSize: 21, bold: true, color: C.white });
  tx(s, bullets([
    "e : embedding Perch v2 gelé de la fenêtre",
    "μ₊ : moyenne des fenêtres où l'espèce chante ; μ₋ : où elle ne chante pas",
    { t: "= notre prototype différentiel, en cosinus", o: { bold: true, color: C.gold } },
  ], { gap: 14 }), { x: MX + 0.3, y: 3.2, w: 4.7, h: 2.8, fontSize: 16, color: C.white });
  tx(s, "Miyaguchi, Gustineli, Cheung (Georgia Tech) · arXiv 2607.14474", { x: MX + 0.3, y: 6.3, w: 4.7, h: 0.4, fontSize: 11, italic: true, color: C.sage });
  const st = [["1,3 % → 76,6 %", "part des amphibiens : enregistrements focaux → paysages sonores étiquetés"], ["0,895 vs 0,074", "AP macro des non-oiseaux : prototypes vs leur réseau SED seul"]];
  st.forEach(([n, l], i) => {
    const x = 6.2 + i * 3.3;
    card(s, x, 1.75, 3.15, 1.75, C.paler);
    tx(s, n, { x: x + 0.2, y: 1.88, w: 2.8, h: 0.6, fontFace: BF, fontSize: 26, bold: true, color: C.goldInk });
    tx(s, l, { x: x + 0.2, y: 2.52, w: 2.8, h: 0.9, fontSize: 12.5, color: C.muted });
  });
  card(s, 6.2, 3.7, 6.45, 1.55, C.white, C.line);
  tx(s, "RÉSERVES", { x: 6.4, y: 3.8, w: 6, h: 0.3, fontSize: 11, bold: true, color: C.terra, charSpacing: 1 });
  tx(s, bullets([
    "AP mesurée sur les fichiers qui construisent les prototypes : optimiste",
    "Site tenu à l'écart, espèce sans exemple d'entraînement : prototypes et ridge au hasard",
  ], { gap: 5 }), { x: 6.4, y: 4.15, w: 6.1, h: 1.05, fontSize: 13 });
  card(s, 6.2, 5.45, 6.45, 1.4, C.pale);
  tx(s, rich([["Pour nous : ", true, { color: C.teal }], ["prototype différentiel pour amorcer un nouveau site, logistique régularisée dès qu'il y a des exemples. Perch v2 porte de l'information sur les anoures.", false]]), { x: 6.4, y: 5.45, w: 6.1, h: 1.4, fontSize: 14, valign: "middle" });
}

// =====================================================================
// 8. GitHub
// =====================================================================
{
  const s = content("GitHub", "Le dépôt s'ouvre aux tuteurs",
    "Proposition : ajouter les tuteurs comme collaborateurs du dépôt rumble-paw-patrol/grenouille, en lecture, ou en écriture s'ils veulent commenter les pull requests. Ce qu'ils y trouvent : le README (installation, commandes, inventaire), la feuille de route V5, DECISIONS.md qui trace chaque choix avec sa date (160 à ce jour), les rapports de benchmarks avec leurs données et le script qui refait leurs figures, les tableaux en PNG lisibles partout, et le code avec ses tests. Les données de l'ONF ne sont pas dans le dépôt : seuls 66 clips de 10 s servent d'échantillon de test. J'ai besoin de vos identifiants GitHub.");
  card(s, MX, 1.75, 5.6, 5.1, C.dark);
  const tree = [
    ["grenouille/", C.gold, true, ""],
    ["├─ README.md", C.white, false, "installer, lancer"],
    ["├─ DECISIONS.md", C.white, false, "160 choix datés"],
    ["├─ blanci/", C.white, false, "le code"],
    ["├─ tests/", C.white, false, "tests automatiques"],
    ["├─ notebooks/", C.white, false, "3 carnets"],
    ["├─ echantillon/", C.white, false, "66 clips de 10 s"],
    ["├─ biblio/", C.white, false, "bibliographie"],
    ["└─ documentation/", C.white, false, ""],
    ["   ├─ feuille-de-route-V5.md", C.sage, false, ""],
    ["   └─ benchmarks/", C.sage, false, "rapports + données"],
  ];
  tree.forEach(([t, col, b, c], i) => {
    tx(s, t, { x: MX + 0.3, y: 1.98 + i * 0.42, w: 3.2, h: 0.36, fontFace: "Courier New", fontSize: 12.5, color: col, bold: b });
    if (c) tx(s, c, { x: MX + 3.5, y: 1.98 + i * 0.42, w: 2.0, h: 0.36, fontSize: 11.5, color: C.sage, italic: true });
  });
  [["Lire", "chaque benchmark a son rapport : question, méthode, résultats, limites."], ["Commenter", "dans les pull requests ou les issues : une question laisse une trace."], ["Vérifier", "chaque chiffre de cette présentation renvoie à un fichier du dépôt."]].forEach(([t, d], i) => {
    const y = 1.75 + i * 1.3;
    card(s, 6.5, y, 6.2, 1.12, C.paler);
    dot(s, 6.75, y + 0.34, i + 1, C.teal);
    tx(s, t, { x: 7.4, y: y + 0.14, w: 5.1, h: 0.35, fontSize: 16, bold: true });
    tx(s, d, { x: 7.4, y: y + 0.52, w: 5.1, h: 0.5, fontSize: 13, color: C.muted });
  });
  card(s, 6.5, 5.7, 6.2, 1.15, C.gold);
  tx(s, rich([["À me donner : ", true], ["vos identifiants GitHub. Hors du dépôt : l'audio de l'ONF et les stocks d'embeddings.", false]]), { x: 6.7, y: 5.7, w: 5.8, h: 1.15, fontSize: 14, valign: "middle", color: C.dark });
}

// =====================================================================
// 9–10. Données et annotation
// =====================================================================
{
  const s = content("Données · 1/2", "Le jeu de données : 96 292 enregistrements, 8 sites",
    "L'inventaire est complet depuis le 29 septembre. Deux jeux. La phénologie 2023-2024 : trois sites (Kaw, Molokoï, Trésor), deux enregistreurs par site, un an de décembre 2023 à novembre 2024, 66 779 enregistrements, 2 225 h. La campagne 2026 : cinq sites (Mataroni, RNRT, CDR, Patawa Est, Patawa Ouest), un relevé d'environ une semaine par site, 29 513 enregistrements, 979 h. Tous les enregistrements font 2 min, toutes les 30 min de 5 h à 20 h, en 48 kHz stéréo. Au total 3 204 h et 2,2 To. Après les drapeaux (durée anormale, hors relevé, micro dans le sac, horloge douteuse), 94 588 enregistrements restent encodables, soit 4,5 millions de fenêtres de 5 s au pas de 2,5 s. Deux conséquences pour la suite. En 2026, chaque site n'a été enregistré qu'une semaine : le mois est confondu avec le site. Seul le jeu 2023 couvre toutes les saisons. Et les 345 positifs et 150 négatifs importés de Blancinet sortent de l'entraînement et de l'évaluation : ils avaient été choisis par un détecteur et ne couvraient que Mataroni.");
  const sites = [["Trésor", 770.2, 0], ["Kaw", 755.0, 0], ["Molokoï", 700.0, 0], ["Mataroni", 0, 432.5], ["RNRT", 0, 242.7], ["CDR", 0, 179.7], ["Patawa Est", 0, 87.4], ["Patawa Ouest", 0, 36.6]];
  s.addChart(pres.charts.BAR, [
    { name: "2023-2024 · phénologie (1 an)", labels: sites.map((x) => x[0]), values: sites.map((x) => x[1]) },
    { name: "2026 · campagne (1 semaine par site)", labels: sites.map((x) => x[0]), values: sites.map((x) => x[2]) },
  ], {
    x: MX, y: 1.65, w: 7.3, h: 5.25, barDir: "bar", barGrouping: "stacked", barGapWidthPct: 35,
    chartColors: [CHART[2], CHART[0]], catAxisOrientation: "maxMin", catAxisLabelColor: C.ink, catAxisLabelFontSize: 12,
    valAxisLabelColor: C.muted, valAxisLabelFontSize: 10, valGridLine: { color: "E3E8E5", size: 0.5 },
    showValue: true, dataLabelPosition: "ctr", dataLabelFontSize: 10, dataLabelColor: C.white, dataLabelFormatCode: "0;;;",
    showLegend: true, legendPos: "b", legendFontSize: 11, legendColor: C.ink,
    showTitle: true, title: "Heures d'audio par site (enregistrements de 2 min)", titleFontSize: 13, titleColor: C.ink,
  });
  const st = [["3 204 h", "d'audio, 2,2 To"], ["94 588", "enregistrements encodables"], ["4,5 M", "fenêtres de 5 s (pas 2,5 s)"]];
  st.forEach(([n, l], i) => {
    const y = 1.75 + i * 1.18;
    card(s, 8.3, y, 4.4, 1.02, C.paler);
    tx(s, n, { x: 8.5, y: y + 0.1, w: 2.0, h: 0.8, fontFace: HF, fontSize: 26, bold: true, color: C.goldInk, valign: "middle" });
    tx(s, l, { x: 10.5, y: y + 0.1, w: 2.1, h: 0.8, fontSize: 13, color: C.muted, valign: "middle" });
  });
  card(s, 8.3, 5.35, 4.4, 1.55, C.dark);
  tx(s, bullets([
    "2026 : une semaine par site, le mois est confondu avec le site",
    { t: "Labels Blancinet (345 + 150) : sortis", o: { color: C.gold, bold: true } },
  ], { gap: 8 }), { x: 8.5, y: 5.5, w: 4.05, h: 1.3, fontSize: 13.5, color: C.white });
}

{
  const s = content("Données · 2/2", "Annoter en partant de zéro (décision avec Élodie)",
    "Décision de l'entretien avec Élodie, le 29 septembre : on repart de zéro. J'annote seul ; Élodie et Benoît vérifient un sous-échantillon ; aucun benchmark sur ces données avant leur go. Étape 1, avant toute écoute, je partage les points d'écoute : un site 2026 entier tenu à l'écart pour l'évaluation (choisi avec Élodie, au moins 10 points), 20 % des points des autres sites, une station 2023 sur deux ; le reste sert à l'entraînement. Étape 2, le lot d'entraînement : environ 450 extraits de 30 s ; en 2026, trois par point (pic du matin, pic du soir, une autre tranche) ; en 2023, 50 par station, la moitié en saison haute dont deux tiers en février-mars. Étape 3, le jeu d'évaluation : environ 250 enregistrements de 2 min écoutés en entier, sur les points tenus à l'écart, avec au moins 60 positifs. Étape 4, le paquet de vérification, à l'aveugle : mon label est caché et l'ordre mélangé ; environ 15 à 20 % de ce que j'ai annoté, soit 2 h 30 d'écoute chacun ; ils écoutent avec leur outil habituel et répondent dans un tableur. Étape 5, la règle du go : zéro erreur sur 60 positifs, au plus un chant manqué sur 100 négatifs ; le go est écrit dans DECISIONS.md. Les principes : aucun détecteur dans le tirage, sinon le jeu hérite de ses angles morts ; le point d'écoute est l'unité qui compte, donc beaucoup de points et peu d'enregistrements par point ; chaque strate qui donne des positifs donne aussi des négatifs, sinon la tête apprend l'heure ou le site ; tout tirage est fait par un script à graine fixée. À l'écoute, je note les intervalles où A. blanci chante, solo ou chœur, et la qualité A/B/C ; la décision reste binaire. Budget : environ 25 h pour moi.");
  const flow = [
    ["Partition des points", "avant écoute : un site entier + 20 % des points tenus à l'écart"],
    ["Lot d'entraînement", "≈ 450 extraits de 30 s, 3 par point, 60 % aux heures de pic"],
    ["Jeu d'évaluation", "≈ 250 enregistrements entiers, ≥ 60 positifs"],
    ["Vérification à l'aveugle", "Élodie et Benoît : 15–20 %, ≈ 2 h 30 chacun"],
    ["Go écrit", "0 erreur / 60 positifs, ≤ 1 chant manqué / 100"],
  ];
  const step = CW / 5;
  flow.forEach(([t, d], i) => {
    const x = MX + i * step, hot = i === 3;
    card(s, x + 0.06, 1.75, step - 0.3, 2.35, hot ? C.dark : C.paler);
    dot(s, x + 0.26, 1.93, i + 1, hot ? C.gold : C.teal, 0.42);
    tx(s, t, { x: x + 0.26, y: 2.47, w: step - 0.6, h: 0.65, fontSize: 15, bold: true, color: hot ? C.white : C.ink });
    tx(s, d, { x: x + 0.26, y: 3.12, w: step - 0.6, h: 0.9, fontSize: 12.5, color: hot ? C.sage : C.muted });
    if (i < 4) arrow(s, x + step - 0.24, 2.9, 0.18);
  });
  const pr = [
    ["Aucun détecteur dans le tirage", "sinon le jeu hérite de ses angles morts"],
    ["Le point d'écoute est l'unité", "beaucoup de points, peu d'enregistrements par point"],
    ["Positifs et négatifs des mêmes strates", "sinon la tête apprend l'heure ou le site"],
  ];
  pr.forEach(([t, d], i) => {
    const y = 4.4 + i * 0.83;
    card(s, MX, y, 7.2, 0.7, C.pale);
    tx(s, rich([[t + " : ", true], [d, false, { color: C.muted }]]), { x: MX + 0.25, y, w: 6.8, h: 0.7, fontSize: 13.5, valign: "middle" });
  });
  card(s, 8.05, 4.4, 4.65, 2.36, C.dark);
  tx(s, "À L'ÉCOUTE : ANNOTER FIN, DÉCIDER GROS", { x: 8.3, y: 4.55, w: 4.2, h: 0.3, fontSize: 11, bold: true, color: C.gold, charSpacing: 1 });
  tx(s, bullets([
    "Intervalles de chant, à 0,5 s près",
    "Solo ou chœur, qualité A/B/C",
    "Décision binaire : présente ou non",
    "Budget : ≈ 25 h d'écoute",
  ], { gap: 7 }), { x: 8.3, y: 4.95, w: 4.2, h: 1.75, fontSize: 14, color: C.white });
}

// =====================================================================
// 11. Pré-benchmark AnuraSet (réservée)
// =====================================================================
{
  const s = content("Pré-benchmark AnuraSet", "Pré-benchmark AnuraSet",
    "Section en cours de mise à jour dans une autre session : à compléter.");
  card(s, MX, 1.9, CW, 4.9, C.paler, C.line);
  tx(s, "Section en cours de mise à jour", { x: MX, y: 3.6, w: CW, h: 0.6, fontFace: HF, fontSize: 24, bold: true, color: C.muted, align: "center" });
  tx(s, "à compléter", { x: MX, y: 4.25, w: CW, h: 0.4, fontSize: 14, color: C.sage, align: "center", italic: true });
}

// =====================================================================
// 12–13. Régularisations
// =====================================================================
{
  const s = content("Régularisations · 1/2", "Les classiques, et les élégantes",
    "La L2, ou ridge : on ajoute lambda fois la norme des poids au carré à la perte. Les poids restent petits, la frontière ne repose pas sur quelques dimensions ; indispensable avec 1 536 dimensions pour quelques dizaines d'enregistrements positifs. Le paramètre C, l'inverse de lambda, se choisit par validation groupée par point d'écoute, avec la règle du « 1 écart-type » : le C le plus régularisant qui reste à moins d'un écart-type du meilleur. C'est en place. SpecAugment (Park et al. 2019) : pendant l'entraînement, on masque au hasard des bandes de fréquence et des tranches de temps du spectrogramme ; le réseau apprend à ne pas dépendre d'un seul indice. Chez nous, l'encodeur est gelé et les embeddings calculés une fois : SpecAugment ne sert que si l'on entraîne un réseau (distillation, modèle maison), ou si l'on ré-encode des copies augmentées. Et un masque sur la bande 4,4–5,5 kHz efface la note : il faut exclure cette bande du tirage. Les élégantes ont une idée commune : ne pas tirer les poids vers zéro, mais vers une solution déjà raisonnable. R30 : une L2 centrée sur le prototype différentiel ; avec peu d'annotations, la tête reste proche du prototype, puis s'en écarte quand les données le justifient. R31 : l'analyse discriminante linéaire à covariance rétrécie, un continuum entre le prototype (gamma = 1) et la logistique (gamma = 0), en solution fermée. L2-SP : pour affiner un réseau, on pénalise l'écart aux poids pré-entraînés plutôt qu'à zéro.");
  // L2
  card(s, MX, 1.75, 3.9, 5.1, C.paler);
  tx(s, "L2 · RIDGE", { x: MX + 0.25, y: 1.92, w: 3.4, h: 0.3, fontSize: 11, bold: true, color: C.teal, charSpacing: 1 });
  tx(s, "perte + λ ‖w‖²", { x: MX + 0.25, y: 2.3, w: 3.5, h: 0.6, fontFace: HF, fontSize: 24, bold: true });
  tx(s, bullets(["Poids petits : la frontière ne repose pas sur quelques dimensions", "C choisi par validation groupée par point d'écoute", { t: "En place", o: { bold: true, color: C.teal } }], { gap: 14 }), { x: MX + 0.25, y: 3.1, w: 3.45, h: 3.6, fontSize: 15.5 });
  // SpecAugment
  const x2 = MX + 4.1;
  card(s, x2, 1.75, 4.0, 5.1, C.paler);
  tx(s, "SPECAUGMENT", { x: x2 + 0.25, y: 1.92, w: 3.5, h: 0.3, fontSize: 11, bold: true, color: C.teal, charSpacing: 1 });
  s.addImage({ path: FIG("specaugment.png"), x: x2 + 0.15, y: 2.3, w: 3.7, h: 3.7 * 640 / 2000 });
  tx(s, bullets(["Masques aléatoires en temps et en fréquence", "Sans effet sur un encodeur gelé : pour un réseau entraîné", { t: "Exclure 4,4–5,5 kHz : sinon la note est effacée", o: { bold: true, color: C.terra } }], { gap: 8 }), { x: x2 + 0.25, y: 3.6, w: 3.55, h: 3.1, fontSize: 13.5 });
  // Élégantes
  const x3 = MX + 8.3;
  card(s, x3, 1.75, CW - 8.3, 5.1, C.dark);
  tx(s, "LES ÉLÉGANTES", { x: x3 + 0.25, y: 1.92, w: 3.3, h: 0.3, fontSize: 11, bold: true, color: C.gold, charSpacing: 1 });
  tx(s, "tirer vers ce qu'on sait déjà, pas vers zéro", { x: x3 + 0.25, y: 2.25, w: 3.35, h: 0.6, fontSize: 13, italic: true, color: C.sage });
  [["R30", "λ ‖w − w_proto‖²", "rester près du prototype tant qu'il y a peu d'exemples"], ["R31", "LDA rétrécie", "du prototype à la logistique, en solution fermée"], ["L2-SP", "λ ‖θ − θ₀‖²", "affiner un réseau sans oublier le pré-entraîné"]].forEach(([k, f, d], i) => {
    const y = 3.0 + i * 1.25;
    tx(s, rich([[k + "  ", true, { color: C.gold }], [f, true, { color: C.white, fontFace: HF }]]), { x: x3 + 0.25, y, w: 3.35, h: 0.4, fontSize: 15 });
    tx(s, d, { x: x3 + 0.25, y: y + 0.42, w: 3.35, h: 0.7, fontSize: 12.5, color: C.sage });
  });
}

{
  const s = content("Régularisations · 2/2", "Pour de nouveaux points d'écoute : retirer ce qui dit le site",
    "Le problème : la note d'A. blanci occupe environ 2 % d'une fenêtre de 3 s ; le reste, c'est le fond du site et du micro. Les embeddings se regroupent alors par site plutôt que par espèce, et une tête peut apprendre le site au lieu du chant. Les régularisations de cette diapo retirent de l'embedding ce qui dit le site ou le micro. R19 : soustraire la moyenne des embeddings du point d'écoute, dans l'esprit du prototype différentiel. R20 : centrer et réduire par site, à la manière d'AdaBN. R21 : retirer les directions qui prédisent le micro (INLP, LEACE) ; interdit avec R19, car la combinaison efface le chant. R37 : un biais par point d'écoute, rétréci vers zéro, comme dans un modèle mixte. DANN : apprendre une représentation dont on ne peut pas deviner le micro. Côté données, R1 colle des notes sur le fond d'autres sites. Et une régularisation qui vient de la nouvelle stratégie d'annotation : noter solo ou chœur et la qualité A/B/C, puis les apprendre comme tâches auxiliaires, oblige la tête à s'appuyer sur ce qui est commun à tous les A. blanci, la note et le rythme ; essayé seulement si chaque sous-classe compte au moins 30 enregistrements indépendants. Toutes se jugent sur des points tenus à l'écart, jamais en validation croisée sur des points connus : dans ce cas, le raccourci du site reste vrai et les classements s'inversent.");
  card(s, MX, 1.75, 5.2, 2.3, C.dark);
  tx(s, [
    { text: "embedding = ", options: { color: C.white } },
    { text: "chant", options: { color: C.gold, bold: true } },
    { text: " + ", options: { color: C.white } },
    { text: "fond du point", options: { color: C.sage, bold: true } },
  ], { x: MX + 0.3, y: 1.95, w: 4.7, h: 0.55, fontFace: HF, fontSize: 22 });
  tx(s, "La note ≈ 2 % de la fenêtre : le fond domine, les embeddings se groupent par site. Retirer le fond rend le chant comparable d'un point à l'autre.", { x: MX + 0.3, y: 2.6, w: 4.65, h: 1.3, fontSize: 13.5, color: C.white });
  card(s, MX, 4.25, 5.2, 2.6, C.pale);
  tx(s, "RÈGLE", { x: MX + 0.3, y: 4.4, w: 4.6, h: 0.3, fontSize: 11, bold: true, color: C.teal, charSpacing: 1 });
  tx(s, "Toujours jugées sur des points tenus à l'écart, jamais en validation croisée sur des points connus : le raccourci du site y reste vrai.", { x: MX + 0.3, y: 4.8, w: 4.6, h: 1.9, fontSize: 15 });
  table(s, [
    ["Régularisation", "Ce qu'elle fait"],
    ["R19 · centrage par point", "soustrait la moyenne des embeddings du point"],
    ["R20 · AdaBN par site", "centre et réduit chaque site"],
    ["R21 · retrait de directions", "efface ce qui prédit le micro (INLP, LEACE) ; jamais avec R19"],
    ["R37 · biais par point", "un biais par point, rétréci vers 0 (modèle mixte)"],
    ["DANN", "représentation où le micro est indevinable"],
    ["R1 · mélange de fond", "notes collées sur le fond d'autres sites"],
    [{ text: "Tâches auxiliaires", options: { bold: true, color: C.goldInk } }, "solo / chœur et qualité A/B/C, apprises à côté de la décision"],
  ], { x: 6.1, y: 1.75, w: 6.63, colW: [2.35, 4.28], fs: 13, rowH: 0.63 });
}

// =====================================================================
// 14. Conclusion
// =====================================================================
{
  const s = pres.addSlide();
  pageNo += 1;
  s.background = { color: C.dark };
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: 0.55, w: 3.6, h: 0.36, fill: { color: C.dark2 }, line: { color: C.dark2, width: 0 }, rectRadius: 0.18 });
  tx(s, "D'ICI LA PROCHAINE RÉUNION", { x: MX, y: 0.55, w: 3.6, h: 0.36, fontSize: 11, bold: true, color: C.gold, align: "center", valign: "middle", charSpacing: 1 });
  tx(s, "Les deux à trois prochaines semaines", { x: MX, y: 1.05, w: 12, h: 0.8, fontFace: HF, fontSize: 32, bold: true, color: C.white });
  const steps = [
    ["Annoter mon jeu", "lot d'entraînement (≈ 450 extraits de 30 s) et jeu d'évaluation (≈ 250 enregistrements entiers), tirés sans détecteur"],
    ["Vérifier avec Élodie et Benoît", "paquet à l'aveugle, ≈ 2 h 30 d'écoute chacun ; corrections ; go écrit"],
    ["Benchmarker sur mes données", "même gabarit que le benchmark AnuraSet, sur des points tenus à l'écart"],
    ["Présenter le benchmark", "et conclure à la prochaine réunion"],
  ];
  const step = CW / 4;
  s.addShape(pres.shapes.LINE, { x: MX + 0.3, y: 2.5, w: CW - 0.6, h: 0, line: { color: C.teal, width: 2 } });
  steps.forEach(([t, d], i) => {
    const x = MX + i * step;
    dot(s, x + 0.1, 2.26, i + 1, i === 3 ? C.gold : C.teal, 0.48);
    card(s, x + 0.05, 3.0, step - 0.25, 2.6, C.dark2);
    tx(s, t, { x: x + 0.3, y: 3.2, w: step - 0.7, h: 0.75, fontSize: 18, bold: true, color: i === 3 ? C.gold : C.white });
    tx(s, d, { x: x + 0.3, y: 4.0, w: step - 0.7, h: 1.5, fontSize: 13.5, color: C.sage });
  });
  card(s, MX, 5.95, CW, 0.9, C.gold);
  tx(s, rich([["Questions : ", true], ["vos identifiants GitHub ? · peut-on encoder dans le cloud plutôt que sur un portable (droits sur l'audio, stockage, budget) ?", false]]), { x: MX + 0.3, y: 5.95, w: CW - 0.6, h: 0.9, fontSize: 15, valign: "middle", color: C.dark });
  notes(s, `${pageNo + 1}. D'ici la prochaine réunion`, "D'ici la prochaine réunion, dans deux à trois semaines. Un : annoter mon jeu, le lot d'entraînement et le jeu d'évaluation, tirés par script sans aucun détecteur. Deux : faire vérifier un sous-échantillon par Élodie et Benoît, à l'aveugle, corriger, et écrire le go. Trois : lancer le benchmark sur mes propres données, avec le même gabarit que le benchmark AnuraSet, sur des points tenus à l'écart. Quatre : présenter ce benchmark et conclure à la prochaine réunion. Deux questions pour vous : vos identifiants GitHub pour ouvrir le dépôt, et la possibilité d'encoder dans le cloud plutôt que sur mon portable ou sur les machines de l'ONF.");
}

// ---------- sorties ----------
const md = ["# Présentation de suivi n° 2 — script", "", "Notes orateur de chaque diapo (générées par `deck.js`).", ""];
script.forEach(([t, n]) => md.push(`## ${t}`, "", n, ""));
fs.writeFileSync(path.join(HERE, "script.md"), md.join("\n"));
pres.writeFile({ fileName: OUT }).then((f) => console.log("écrit :", f));
