// Présentation de suivi n° 2 — généré par pptxgenjs.
//   node documentation/presentation-suivi-2/deck.js [sortie.pptx]
// Écrit aussi script.md (notes orateur de chaque diapo) à côté de ce fichier.
// Figures et audio : generer_figures.py. Charte de la présentation n° 2, à garder pour les
// suivantes : famille de verts de la n° 1, Cambria + Calibri, accent or, pastilles de rubrique.

const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const HERE = __dirname;
const FIG = (f) => path.join(HERE, "figures", f);
const AUD = (f) => path.join(HERE, "audio", f);
const OUT = process.argv[2] || path.join(HERE, "..", "presentation_suivi_2.pptx");

// ---------- charte ----------
const C = {
  dark: "0E2A24", dark2: "173D34", ink: "1B2A24", muted: "5B6B63", sage: "A7BFB2",
  pale: "E4EEE8", paler: "F3F7F4", white: "FFFFFF", gold: "E9B44C", goldInk: "8A5A00",
  teal: "2A7F72", terra: "C8553D", line: "CFDDD5", blue: "3E7CB1", green: "2E8B57",
};
const CHART = ["2E8B57", "D9822B", "3E7CB1", "C0492F", "8A6BBE"]; // validée (dataviz)
const HF = "Cambria", BF = "Calibri";
const W = 13.333, H = 7.5, MX = 0.6, CW = W - 2 * MX;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "Détection acoustique d'Anomaloglossus blanci — suivi n° 2";
pres.author = "Léonard Laplace-Palette";

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
function label(slide, text, x, y, w, color) {
  tx(slide, text.toUpperCase(), { x, y, w, h: 0.3, fontSize: 11, bold: true, color: color || C.teal, charSpacing: 1 });
}
function notes(slide, title, n) {
  slide.addNotes(n);
  script.push([title, n]);
}
function content(rubrique, title, n) {
  const s = pres.addSlide();
  s.background = { color: C.white };
  pageNo += 1;
  const wChip = 0.2 + rubrique.length * 0.085;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: 0.4, w: wChip, h: 0.32, fill: { color: C.pale }, line: { color: C.pale, width: 0 }, rectRadius: 0.16 });
  tx(s, rubrique.toUpperCase(), { x: MX, y: 0.4, w: wChip, h: 0.32, fontSize: 10, bold: true, color: C.teal, align: "center", valign: "middle", charSpacing: 1 });
  if (title) tx(s, title, { x: MX, y: 0.82, w: CW, h: 0.7, fontFace: HF, fontSize: 28, bold: true, color: C.ink, valign: "middle" });
  tx(s, String(pageNo + 1), { x: W - 1.0, y: H - 0.42, w: 0.5, h: 0.25, fontSize: 9, color: C.sage, align: "right" });
  notes(s, `${pageNo + 1}. ${title || rubrique}`, n);
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
function richBullets(paras, gap = 6) {
  // paras : liste de paragraphes, chacun une liste [texte, gras] ; une puce par paragraphe
  const out = [];
  paras.forEach((runs, pi) => {
    runs.forEach(([t, b], ri) => {
      const o = { bold: !!b };
      if (ri === 0) { o.bullet = { indent: 14 }; o.paraSpaceAfter = gap; }
      if (ri === runs.length - 1 && pi < paras.length - 1) o.breakLine = true;
      out.push({ text: t, options: o });
    });
  });
  return out;
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
// Titre
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
  tx(s, "Attention · état de l'art · BirdCLEF+ 2026 · données et annotation · module séquentiel · régularisations", { x: MX, y: 3.85, w: 7.6, h: 0.7, fontSize: 17, color: C.sage });
  tx(s, "Léonard Laplace-Palette · 30 septembre 2026", { x: MX, y: 4.75, w: 6, h: 0.35, fontSize: 13, color: C.sage });
  tx(s, "Fond : un chant d'A. blanci enregistré à Mataroni (spectrogramme, 1–9 kHz)", { x: 8.75, y: 7.05, w: 4.4, h: 0.3, fontSize: 9, color: C.white, italic: true });
  notes(s, "1. Titre", "Deuxième point de suivi. Plan : un cours sur l'attention en trois diapos, ce qui se fait de mieux aujourd'hui en bioacoustique, BirdCLEF+ 2026, le dépôt GitHub, le jeu de données et la nouvelle stratégie d'annotation décidée avec Élodie, le module séquentiel, le pré-benchmark AnuraSet, les régularisations, et ce que je fais d'ici la prochaine réunion. L'image de fond est un vrai chant d'A. blanci enregistré à Mataroni : les petites notes vers 4,5 kHz.");
}

// =====================================================================
// Attention 1/3 : l'effet, visible
// =====================================================================
{
  const s = content("Attention · 1/3", "Ce que fait l'attention : pondérer les jetons au lieu de les moyenner",
    "On commence par ce qu'on voit. Un encodeur découpe le spectrogramme d'une fenêtre en morceaux, les jetons : perch_v2 en rend 16 × 4 = 64 pour une fenêtre de 5 s. En haut à gauche, une vraie fenêtre de Mataroni découpée en 64 jetons. À droite, des poids d'attention en illustration : je les ai calculés à la main sur la nouveauté de chaque morceau pour montrer l'idée. Les jetons qui portent les notes d'A. blanci, vers 4,5 kHz, prennent le poids ; la bande d'insectes à 8 kHz, constante, n'en prend pas. Le problème que ça résout : avec la moyenne, chaque jeton pèse 1/64, et la note, qui occupe 2 à 4 jetons, est noyée dans le fond. Le graphique est un calcul à la main sur quatre jetons : le jeton de la note reçoit 0,66 du poids, contre 0,25 avec la moyenne. Et la bonne nouvelle : ces cartes, on peut les tracer avec une vraie tête d'attention entraînée. C'est fait dans le notebook 04 : il superpose les poids de la tête au spectrogramme et en tire une mesure, la part d'attention dans la bande de la note comparée à une attention uniforme. Une tête qui écoute le fond se voit à l'œil, même si son AP est bonne.");
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
  card(s, 6.3, 5.2, 6.43, 1.65, C.dark);
  tx(s, rich([["Visible avec une vraie tête : ", true, { color: C.gold }], ["le notebook 04 trace les poids de la tête attentive sur le spectrogramme et mesure la part d'attention posée sur la bande de la note. Une tête qui écoute le fond se voit à l'œil.", false, { color: C.white }]]), { x: 6.55, y: 5.2, w: 6.0, h: 1.65, fontSize: 14, valign: "middle" });
}

// =====================================================================
// Attention 2/3 : le mécanisme en arrière-plan
// =====================================================================
{
  const s = content("Attention · 2/3", "En arrière-plan : requêtes, clés, valeurs",
    "Comment la tête choisit ces poids. Chaque jeton x_i est projeté trois fois par des matrices apprises. La clé k_i dit « ce que je contiens », la valeur v_i « ce que je transmets », la requête q « ce que je cherche ». Étape 1 : le score d'un jeton est le produit scalaire q · k_i, divisé par la racine de la dimension d pour que les scores ne grandissent pas avec la dimension (sinon le softmax sature). Étape 2 : le softmax transforme les scores en poids positifs de somme 1 : ce sont les poids de la diapo précédente. Étape 3 : la sortie est la somme des valeurs pondérées, un vecteur par fenêtre, puis une couche linéaire donne le score. Analogie : une bibliothèque. La requête est ma question, les clés sont les titres des livres, les valeurs leur contenu ; je lis surtout les livres dont le titre répond à ma question. Deux usages : dans un transformer, c'est de l'auto-attention, chaque jeton émet sa propre requête et les jetons se « parlent ». Dans notre sonde attentive, l'encodeur est gelé et on n'apprend qu'une seule requête, « où est A. blanci ? », plus la couche linéaire.");
  const x0 = MX, y0 = 1.85;
  tx(s, "64 jetons d'une fenêtre", { x: x0, y: y0, w: 2.2, h: 0.3, fontSize: 12, bold: true, color: C.muted });
  ["x₁", "x₂", "x₃", "x₄", "x₆₄"].forEach((l, i) => {
    const hot = i === 2;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x0 + 0.2, y: y0 + 0.4 + i * 0.62, w: 1.4, h: 0.48, fill: { color: hot ? C.gold : C.pale }, line: { color: hot ? C.gold : C.line, width: 0.75 }, rectRadius: 0.08 });
    tx(s, l, { x: x0 + 0.2, y: y0 + 0.4 + i * 0.62, w: 1.4, h: 0.48, fontSize: 14, align: "center", valign: "middle", bold: hot });
  });
  tx(s, "x₃ : la note", { x: x0 + 0.2, y: y0 + 3.55, w: 1.6, h: 0.3, fontSize: 11, color: C.goldInk, italic: true });
  const px = 2.75;
  [["W_Q", "requête q", "ce que je cherche", C.goldInk], ["W_K", "clés kᵢ", "ce que je contiens", C.teal], ["W_V", "valeurs vᵢ", "ce que je transmets", C.blue]].forEach(([m, n, d, col], i) => {
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
    tx(s, f, { x: x + 0.15, y: y + 0.52, w: 1.85, h: 0.45, fontSize: 15, bold: true, fontFace: HF, color: hot ? C.white : C.ink });
    tx(s, d, { x: x + 0.15, y: y + 1.05, w: 1.85, h: 0.6, fontSize: 11.5, color: hot ? C.sage : C.muted });
    if (i < 2) arrow(s, x + 2.1, y + 0.88, 0.25);
  });
  card(s, 5.75, 4.3, 6.95, 0.75, C.pale);
  tx(s, rich([["En une ligne : ", true], ["Attention(Q, K, V) = softmax(Q Kᵀ / √d) · V", false, { fontFace: HF }], ["   Vaswani et al. 2017", false, { color: C.muted, fontSize: 11 }]]), { x: 5.95, y: 4.3, w: 6.6, h: 0.75, fontSize: 15, valign: "middle" });
  card(s, MX, 5.8, 6.0, 1.05, C.paler);
  tx(s, rich([["Auto-attention (transformer) : ", true], ["chaque jeton émet sa requête ; les jetons se « parlent ».", false, { color: C.muted }]]), { x: MX + 0.25, y: 5.8, w: 5.6, h: 1.05, fontSize: 14, valign: "middle" });
  card(s, 6.85, 5.8, 5.85, 1.05, C.paler);
  tx(s, rich([["Sonde attentive (notre tête) : ", true], ["une seule requête apprise, encodeur gelé.", false, { color: C.muted }]]), { x: 7.1, y: 5.8, w: 5.45, h: 1.05, fontSize: 14, valign: "middle" });
}

// =====================================================================
// Attention 3/3 : avantages et inconvénients
// =====================================================================
{
  const s = content("Attention · 3/3", "Dans notre projet : avantages et inconvénients",
    "Avantages. Un : la note d'A. blanci dure 0,09 s et occupe 2 à 4 jetons sur 64 ; la moyenne la noie, l'attention peut la retrouver. Deux : plusieurs encodeurs candidats sont des transformers, Bird-MAE, BEATs, NatureBEATs. Jugés avec une tête linéaire sur leur embedding moyen, ils sont sous-estimés : BEATs passe de 94,10 à 97,98 AUROC sur BEANS en changeant seulement de tête. Sans sonde attentive, on écarterait à tort des encodeurs peut-être meilleurs, et peut-être libres. Trois : la carte des poids se contrôle à l'œil et donne une mesure de plus (notebook 04). Inconvénients. Un : il faut stocker les jetons, 64 vecteurs par fenêtre au lieu d'un. Deux : la tête apprend une requête de 1 536 valeurs en plus des 1 536 poids de la logistique, soit deux fois plus de paramètres pour le même petit nombre d'enregistrements positifs ; avec peu d'exemples, elle risque d'apprendre par cœur le fond de ses exemples d'entraînement au lieu du chant : c'est le sur-apprentissage. Trois : le gain n'est pas garanti. Dans BirdCLEF+ 2026, l'équipe DS@GT a perdu 0,013 à 0,035 point de score avec une tête à attention ; ils l'attribuent à des fenêtres d'entraînement et d'inférence de durées différentes. Ce n'est pas lié à l'amorçage d'un site. Verdict : la logistique reste la tête par défaut. La sonde attentive s'essaie sur tout encodeur qui fournit ses jetons, pas seulement les transformers : perch_v2, un CNN, les fournit déjà. On ne la garde que si elle fait mieux sur des points d'écoute tenus à l'écart, c'est-à-dire des couples site et micro qui n'ont servi ni à l'entraînement ni au réglage : la situation d'une future campagne sur de nouveaux points.");
  const colW2 = (CW - 0.3) / 2;
  const cols = [
    ["Avantages", C.teal, [
      ["Fait pour une note brève", "0,09 s : 2 à 4 jetons sur 64, noyés par la moyenne"],
      ["Juger équitablement les transformers", "Bird-MAE, BEATs : sous-estimés en linéaire (BEATs 94,10 → 97,98 AUROC)"],
      ["Contrôlable à l'œil", "carte des poids + part d'attention sur la bande (notebook 04)"],
    ]],
    ["Inconvénients", C.terra, [
      ["Mémoire", "64 jetons par fenêtre au lieu d'un vecteur"],
      ["Deux fois plus de paramètres", "même peu de positifs : risque d'apprendre le fond par cœur"],
      ["Gain non garanti", "BirdCLEF+ 2026 (DS@GT) : −0,013 à −0,035 avec une tête à attention"],
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
  tx(s, rich([["Verdict : ", true, { color: C.gold }], ["la logistique reste la tête par défaut. La sonde attentive s'essaie sur tout encodeur qui fournit ses jetons (perch_v2 déjà), et n'est gardée que si elle fait mieux sur des points d'écoute jamais vus à l'entraînement.", false, { color: C.white }]]), { x: MX + 0.3, y: 5.75, w: CW - 0.6, h: 1.1, fontSize: 15, valign: "middle" });
}

// =====================================================================
// État de l'art : ce qui se fait de mieux
// =====================================================================
{
  const s = content("État de l'art", "Ce qui se fait de mieux aujourd'hui",
    "Quatre angles. Les encodeurs : Perch 2.0 (Google DeepMind, 2025) est la référence en sondage linéaire, avec une ROC-AUC de 0,908 sur BirdSet ; MetaPerch (ICML 2026) fait jeu égal en moyenne et mieux en Amérique du Sud, grâce aux métadonnées apprises pendant l'entraînement ; Bird-MAE et NatureBEATs sont excellents mais seulement avec une tête sur leurs jetons ; BirdNET+ V3 est en préversion, 11 000 espèces. Les chaînes de traitement, telles que les ont construites les cinq premiers de BirdCLEF+ 2026 : Perch v2 sert de professeur, ses embeddings sont copiés (distillés) dans des réseaux convolutifs plus rapides ; le modèle annote ensuite lui-même les heures d'audio non étiquetées (pseudo-labels) et on le réentraîne dessus ; enfin des a priori de site et d'heure et un lissage dans le temps. Les ensembles : les cinq premiers combinent tous de 3 à 10 modèles différents, et la diversité des modèles rapporte plus que le meilleur modèle seul. Les anoures : il n'existe pas de modèle de fondation dédié ; les embeddings appris sur les oiseaux transfèrent aux anoures (BirdNET et Perch sur AnuraSet) ; une étude de 2025 dans Ecological Informatics sur un anoure tropical en danger critique trouve que le transfert d'un modèle de fondation plus une tête simple bat les gabarits, un CNN dédié, la recherche par similarité et le zéro-shot ; à BirdCLEF+ 2026, le premier a ajouté un spécialiste amphibiens et insectes et un modèle au niveau du genre pour les amphibiens rares. Ce qu'on en garde : la base de toutes ces solutions est la nôtre, Perch v2 gelé plus une tête. Ce qui s'ajoute au-dessus, distillation, pseudo-labels, ensembles, demande un GPU et des centaines d'heures non étiquetées : on le garde pour après le choix de l'encodeur, et seulement si l'i5 de l'ONF peut le faire tourner.");
  const cols = [
    ["Encodeurs", C.teal, [
      [["Perch 2.0", true], [" : référence en linéaire, BirdSet 0,908", false]],
      [["MetaPerch", true], [" (2026) : mieux en Amérique du Sud", false]],
      [["Bird-MAE, NatureBEATs", true], [" : forts avec une tête sur jetons", false]],
      [["BirdNET+ V3", true], [" : préversion, 11 000 espèces", false]],
    ]],
    ["Chaînes (BirdCLEF+ 2026)", C.blue, [
      [["Distillation", true], [" : Perch v2 professeur d'un CNN rapide", false]],
      [["Pseudo-labels", true], [" : le modèle annote l'audio non étiqueté", false]],
      [["A priori", true], [" de site et d'heure, lissage dans le temps", false]],
    ]],
    ["Ensembles", C.goldInk, [
      [["Tout le top 5", true], [" : 3 à 10 modèles combinés", false]],
      [["La diversité", true], [" rapporte plus que le meilleur modèle seul", false]],
    ]],
    ["Anoures, amphibiens", C.terra, [
      [["Pas de modèle dédié", true], [" : les embeddings d'oiseaux transfèrent", false]],
      [["Transfert + tête simple", true], [" bat gabarits, CNN dédié et zéro-shot (2025)", false]],
      [["BirdCLEF+ 2026", true], [" : spécialiste amphibiens, modèle par genre", false]],
    ]],
  ];
  const cw = (CW - 0.45) / 4;
  cols.forEach(([t, col, items], i) => {
    const x = MX + i * (cw + 0.15);
    card(s, x, 1.7, cw, 3.75, C.paler);
    s.addShape(pres.shapes.OVAL, { x: x + 0.22, y: 1.9, w: 0.2, h: 0.2, fill: { color: col }, line: { color: col, width: 0 } });
    tx(s, t, { x: x + 0.52, y: 1.82, w: cw - 0.65, h: 0.4, fontSize: 15, bold: true, color: col });
    tx(s, richBullets(items, 12), { x: x + 0.22, y: 2.4, w: cw - 0.4, h: 2.95, fontSize: 14 });
  });
  card(s, MX, 5.65, CW, 1.2, C.dark);
  tx(s, rich([["Pour nous : ", true, { color: C.gold }], ["la base de toutes ces solutions est la nôtre, Perch v2 gelé + une tête. Distillation, pseudo-labels et ensembles viennent au-dessus : à essayer après le choix de l'encodeur, s'ils tiennent sur l'i5 de l'ONF.", false, { color: C.white }]]), { x: MX + 0.3, y: 5.65, w: CW - 0.6, h: 1.2, fontSize: 15, valign: "middle" });
}

// =====================================================================
// BirdCLEF+ 2026 · 1/2
// =====================================================================
{
  const s = content("BirdCLEF+ 2026 · 1/2", "Le concours et les solutions du top 5",
    "Le concours. BirdCLEF+ 2026, sur Kaggle, organisé par le Cornell Lab of Ornithology dans le cadre de LifeCLEF, du 11 mars au 3 juin 2026 ; 4 094 équipes ; working notes présentées à CLEF 2026, à Iéna, du 21 au 24 septembre. Les données : 234 classes. Ce sont surtout des espèces : 162 oiseaux, 35 amphibiens, 8 mammifères, 1 reptile ; plus 28 « sonotypes » d'insectes, des types de sons reconnus mais sans espèce nommée. Pour l'entraînement : 344 h d'enregistrements ciblés venant de Xeno-canto et d'iNaturalist, un animal au premier plan, et 178 h de paysages sonores de 23 sites du Pantanal, dont une heure seulement étiquetée. Le test : environ 600 enregistrements d'une minute, cachés. La contrainte : la soumission est un notebook Kaggle sur processeur seul, 4 cœurs, sans GPU ni internet, qui doit prédire tout le jeu de test en 90 minutes au plus. L'entraînement se fait ailleurs, sans limite. Le score : ROC-AUC macro, par fenêtre de 5 s. Le « score final » du tableau est celui du classement dit privé : le jeu de test est coupé en deux parties : l'une sert au classement affiché pendant le concours, l'autre reste cachée et ne sert qu'au classement final, pour punir ceux qui ont réglé leur modèle sur le classement affiché. Les approches du top 5, lues dans leurs comptes rendus. Premier, Nikita Babych, 0,966 : neuf réseaux convolutifs dont on copie d'abord les embeddings de Perch v2 (un seul copie AudioProtoPNet), puis affinés et entraînés en Noisy Student ; plus un spécialiste amphibiens et insectes entraîné sur 1 903 espèces de Xeno-canto et d'iNaturalist, un modèle qui prédit le genre plutôt que l'espèce pour les amphibiens rares, et Perch v2 tel quel. Noisy Student, c'est de l'auto-apprentissage : un modèle professeur annote l'audio non étiqueté, un élève apprend sur ces annotations avec du bruit ajouté (augmentations), puis l'élève devient le professeur du tour suivant ; chez lui, 0,935 sans, 0,950 après deux tours. Deuxième, tennogh, 0,960 : cinq tours de pseudo-labels, des réseaux pré-entraînés sur Xeno-canto par la deuxième équipe de 2025, une perte qui optimise directement l'AUC, un spécialiste insectes, et deux chaînes publiques (Perch et un SED distillé) ; il a volontairement renoncé à distiller Perch dans ses modèles pour qu'ils restent différents. Troisième, kapenon, 0,960 : trois modèles seulement, deux EfficientNetV2 repris de la deuxième place 2025 et un SEResNeXt distillé de Perch v2, un seul tour de pseudo-labels, des a priori de site et d'heure, et le moteur OpenVINO pour aller vite sur CPU. Quatrième, Tony Li, Yiheng Wang et Starry, 0,959 : Perch v2 tel quel pour 40 % de l'ensemble, deux SED et deux SED distillés de Perch ; seuls les pseudo-labels tirés de Perch ont aidé ; ajouter AnuraSet en données externes a fait baisser leur score, par écart de domaine. Cinquième, Jiacheng Ma, 0,958 : quatre CNN (HGNetV2, EfficientNetV2 en 5 et 10 s, EfficientNet-B3) plus une chaîne ProtoSSM et un SED distillé ; distillation de Perch, auto-distillation, pseudo-labels, et une augmentation qui change le gain par bande de fréquence ; son code est public.");
  const facts = [
    ["234 classes", "162 oiseaux, 35 amphibiens, 28 types de sons d'insectes, 8 mammifères, 1 reptile"],
    ["522 h d'audio", "344 h d'enregistrements ciblés (Xeno-canto, iNaturalist) + 178 h de paysages sonores de 23 sites, dont 1 h étiquetée"],
    ["CPU, 90 min", "un notebook sur 4 cœurs, sans GPU ni internet, pour prédire tout le test"],
  ];
  facts.forEach(([n, l], i) => {
    const y = 1.7 + i * 1.72;
    card(s, MX, y, 3.2, 1.58, i === 2 ? C.dark : C.paler);
    tx(s, n, { x: MX + 0.2, y: y + 0.1, w: 2.85, h: 0.5, fontFace: HF, fontSize: 20, bold: true, color: i === 2 ? C.gold : C.goldInk });
    tx(s, l, { x: MX + 0.2, y: y + 0.62, w: 2.85, h: 0.9, fontSize: 11.5, color: i === 2 ? C.white : C.muted });
  });
  table(s, [
    ["Rang", "Équipe", "Score final", "Approche"],
    ["1", "Nikita Babych", "0,966", "9 CNN qui copient Perch v2 (distillation), puis Noisy Student ; spécialiste amphibiens/insectes, modèle par genre, Perch v2 tel quel"],
    ["2", "tennogh", "0,960", "5 tours de pseudo-labels ; CNN pré-entraînés sur Xeno-canto ; perte AUC ; spécialiste insectes ; sans distillation, pour la diversité"],
    ["3", "kapenon", "0,960", "3 modèles : 2 EfficientNetV2 repris de 2025 + 1 distillé de Perch ; 1 tour de pseudo-labels ; a priori site et heure"],
    ["4", "T. Li, Y. Wang, Starry", "0,959", "Perch v2 tel quel (40 %) + 4 SED dont 2 distillés ; pseudo-labels de Perch ; AnuraSet en plus : score en baisse"],
    ["5", "Jiacheng Ma", "0,958", "4 CNN + ProtoSSM + SED distillé ; auto-distillation, pseudo-labels ; code public"],
  ], { x: 4.0, y: 1.7, w: 8.73, colW: [0.55, 1.6, 0.95, 5.63], fs: 11, rowH: 0.62 });
  tx(s, "Score final : ROC-AUC macro sur la partie cachée du test, révélée à la clôture (« classement privé »).", { x: 4.0, y: 5.55, w: 8.73, h: 0.3, fontSize: 10.5, italic: true, color: C.muted });
  const les = [
    ["Perch v2 partout", "tel quel dans l'ensemble, ou comme professeur copié par un réseau plus rapide"],
    ["Pseudo-labels", "le modèle annote l'audio non étiqueté puis s'y réentraîne : 0,935 → 0,950 chez le 1er"],
    ["Ensembles", "3 à 10 modèles différents dans tout le top 5"],
  ];
  les.forEach(([t, d], i) => {
    const x = 4.0 + i * 2.95;
    card(s, x, 5.95, 2.83, 0.95, C.pale);
    tx(s, rich([[t + " : ", true, { color: C.teal }], [d, false]]), { x: x + 0.15, y: 5.95, w: 2.55, h: 0.95, fontSize: 11, valign: "middle" });
  });
}

// =====================================================================
// BirdCLEF+ 2026 · 2/2 : l'article DS@GT
// =====================================================================
{
  const s = content("BirdCLEF+ 2026 · 2/2", "Une équipe a fait notre prototype différentiel sur Perch v2",
    "Article : Miyaguchi, Gustineli et Cheung (Georgia Tech, équipe DS@GT), « Can Tokens Compete? », arXiv 2607.14474. Leur constat de départ ressemble au nôtre : les amphibiens font 1,3 % des enregistrements ciblés d'entraînement, mais on en entend dans 76,6 % des fenêtres de paysages sonores étiquetées. Et 72 des 234 classes ne sont pas dans les classes de Perch. Pour elles, ils construisent une tête à prototypes sur les embeddings de Perch v2 gelé : mu plus est la moyenne des fenêtres où l'espèce est présente, mu moins la moyenne de celles où elle est absente, et le score est cos(e, mu plus) − cos(e, mu moins). C'est notre prototype différentiel, en similarité cosinus. Résultat : AP macro de 0,895 sur les non-oiseaux, contre 0,074 pour leur réseau entraîné seul, et +0,011 au score final. Deux réserves. Un : cette AP est mesurée sur les 66 fichiers étiquetés qui servent aussi à construire les prototypes ; aucun découpage entraînement / test n'est indiqué, donc le chiffre est probablement trop beau. Deux : dans une autre expérience de leur annexe, validée site par site, pour les 28 espèces qui n'ont aucun enregistrement ciblé, les prototypes tombent au hasard, et leur régression ridge aussi : sans exemple de l'espèce, aucune tête ne la trouve. Pour nous : A. blanci n'est pas dans les classes de Perch. Cet article montre que les prototypes fonctionnent sur les embeddings de Perch même pour des espèces qu'il n'a pas appris à nommer : c'est ce qui justifie le prototype différentiel pour amorcer un nouveau point, et la logistique dès qu'il y a des exemples.");
  card(s, MX, 1.75, 5.3, 5.1, C.dark);
  label(s, "Leur tête à prototypes", MX + 0.3, 1.95, 4.7, C.gold);
  tx(s, "score = cos(e, μ₊) − cos(e, μ₋)", { x: MX + 0.3, y: 2.35, w: 4.8, h: 0.55, fontFace: HF, fontSize: 21, bold: true, color: C.white });
  tx(s, bullets([
    "e : embedding Perch v2 gelé de la fenêtre",
    "μ₊ : moyenne des fenêtres où l'espèce chante ; μ₋ : où elle ne chante pas",
    { t: "= notre prototype différentiel, en cosinus", o: { bold: true, color: C.gold } },
  ], { gap: 14 }), { x: MX + 0.3, y: 3.2, w: 4.7, h: 2.8, fontSize: 16, color: C.white });
  tx(s, "Miyaguchi, Gustineli, Cheung (Georgia Tech) · arXiv 2607.14474", { x: MX + 0.3, y: 6.3, w: 4.7, h: 0.4, fontSize: 11, italic: true, color: C.sage });
  const st = [
    ["Leur problème", "Les amphibiens font 1,3 % des enregistrements d'entraînement, mais s'entendent dans 76,6 % des fenêtres de terrain."],
    ["Leur résultat", "Sur les espèces non-oiseaux : AP 0,895 avec les prototypes, contre 0,074 pour leur réseau entraîné seul."],
  ];
  st.forEach(([t, l], i) => {
    const x = 6.2 + i * 3.3;
    card(s, x, 1.75, 3.15, 1.95, C.paler);
    tx(s, t, { x: x + 0.2, y: 1.88, w: 2.8, h: 0.4, fontSize: 15, bold: true, color: C.goldInk });
    tx(s, l, { x: x + 0.2, y: 2.3, w: 2.8, h: 1.35, fontSize: 13, color: C.ink });
  });
  card(s, 6.2, 3.85, 6.45, 1.45, C.white, C.line);
  label(s, "Réserves", 6.4, 3.95, 6, C.terra);
  tx(s, bullets([
    "AP mesurée sur les fichiers qui construisent les prototypes : trop belle",
    "Espèce sans aucun exemple d'entraînement : prototypes et ridge au hasard",
  ], { gap: 5 }), { x: 6.4, y: 4.3, w: 6.1, h: 0.95, fontSize: 13 });
  card(s, 6.2, 5.45, 6.45, 1.4, C.pale);
  tx(s, rich([["Pour nous : ", true, { color: C.teal }], ["A. blanci n'est pas dans les classes de Perch, et les prototypes marchent quand même sur ses embeddings. Prototype pour amorcer un nouveau point, logistique dès qu'il y a des exemples.", false]]), { x: 6.4, y: 5.45, w: 6.1, h: 1.4, fontSize: 14, valign: "middle" });
}

// =====================================================================
// GitHub
// =====================================================================
{
  const s = content("GitHub", "",
    "Le dépôt du projet est sur GitHub, à mon nom : Rumble-Paw-Patrol, dépôt Grenouille. Vous y trouvez le code, la feuille de route, le journal de toutes les décisions, les rapports de benchmarks et cette présentation. Les enregistrements audio de l'ONF restent sur les disques : seuls 66 extraits de 10 s servent d'échantillon de test. Il me faut vos identifiants GitHub pour vous ajouter au dépôt.");
  s.addImage({ path: FIG("avatar.jpg"), x: 5.17, y: 1.2, w: 3.0, h: 3.0, rounding: true });
  tx(s, "Léonard Laplace-Palette", { x: MX, y: 4.4, w: CW, h: 0.6, fontFace: HF, fontSize: 30, bold: true, align: "center" });
  tx(s, "@Rumble-Paw-Patrol", { x: MX, y: 5.0, w: CW, h: 0.4, fontSize: 17, color: C.muted, align: "center" });
  card(s, 3.67, 5.65, 6.0, 0.72, C.dark);
  s.addImage({ path: FIG("github-mark.png"), x: 3.92, y: 5.79, w: 0.44, h: 0.44 });
  tx(s, [{ text: "github.com/Rumble-Paw-Patrol/Grenouille", options: { hyperlink: { url: "https://github.com/Rumble-Paw-Patrol/Grenouille" }, color: C.white, bold: true } }], { x: 4.55, y: 5.65, w: 5.0, h: 0.72, fontSize: 17, valign: "middle" });
  tx(s, "Les enregistrements audio restent sur les disques de l'ONF : le dépôt ne contient que le code, la documentation et 66 extraits de 10 s pour les tests.", { x: MX, y: 6.8, w: CW - 0.6, h: 0.3, fontSize: 11, italic: true, color: C.muted, align: "center" });
}

// =====================================================================
// Données 1/2 : le jeu de données (barres dessinées, lisibles partout)
// =====================================================================
{
  const s = content("Données · 1/2", "Le jeu de données : 96 292 enregistrements, 8 sites",
    "L'inventaire est complet depuis le 29 septembre. Deux jeux. La phénologie 2023-2024 : trois sites (Kaw, Molokoï, Trésor), deux enregistreurs par site, un an de décembre 2023 à novembre 2024, 66 779 enregistrements, 2 225 h. La campagne 2026 : cinq sites (Mataroni, RNRT, CDR, Patawa Est, Patawa Ouest), un relevé d'environ une semaine par site, 29 513 enregistrements, 979 h. Tous les enregistrements font 2 min, toutes les 30 min de 5 h à 20 h, en 48 kHz stéréo. Au total 3 204 h et 2,2 To. Après les drapeaux (durée anormale, hors relevé, micro dans le sac, horloge douteuse), 94 588 enregistrements restent encodables, soit 4,5 millions de fenêtres de 5 s au pas de 2,5 s. Deux conséquences. En 2026, chaque site n'a été enregistré qu'une semaine : le mois est confondu avec le site, seul le jeu 2023 couvre toutes les saisons. Et les 345 positifs et 150 négatifs importés de Blancinet sortent de l'entraînement et de l'évaluation : ils avaient été choisis par un détecteur et ne couvraient que Mataroni.");
  const sites = [["Trésor", 770.2, 0], ["Kaw", 755.0, 0], ["Molokoï", 700.0, 0], ["Mataroni", 432.5, 1], ["RNRT", 242.7, 1], ["CDR", 179.7, 1], ["Patawa Est", 87.4, 1], ["Patawa Ouest", 36.6, 1]];
  tx(s, "Heures d'audio par site (enregistrements de 2 min)", { x: MX, y: 1.7, w: 7.3, h: 0.35, fontSize: 14, bold: true });
  const x0 = MX + 1.45, maxW = 5.3, maxV = 800, rowH = 0.5, y0 = 2.2;
  sites.forEach(([n, v, set], i) => {
    const y = y0 + i * rowH, w = (v / maxV) * maxW, col = set ? C.green : C.blue;
    tx(s, n, { x: MX, y, w: 1.35, h: rowH - 0.1, fontSize: 13, align: "right", valign: "middle" });
    s.addShape(pres.shapes.RECTANGLE, { x: x0, y: y + 0.06, w, h: rowH - 0.22, fill: { color: col }, line: { color: col, width: 0 } });
    tx(s, `${Math.round(v)} h`, { x: x0 + w + 0.1, y, w: 0.9, h: rowH - 0.1, fontSize: 12, color: C.muted, valign: "middle" });
  });
  const ly = y0 + sites.length * rowH + 0.15;
  [[C.blue, "2023-2024 · phénologie (1 an, 3 sites)"], [C.green, "2026 · campagne (1 semaine par site)"]].forEach(([col, t], i) => {
    const x = MX + 1.45 + i * 3.1;
    s.addShape(pres.shapes.RECTANGLE, { x, y: ly + 0.08, w: 0.22, h: 0.22, fill: { color: col }, line: { color: col, width: 0 } });
    tx(s, t, { x: x + 0.32, y: ly, w: 2.8, h: 0.38, fontSize: 12, valign: "middle" });
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

// =====================================================================
// Données 2/2 : annoter en partant de zéro
// =====================================================================
{
  const s = content("Données · 2/2", "Annoter en partant de zéro (décision avec Élodie)",
    "Décision de l'entretien avec Élodie, le 29 septembre : on repart de zéro. J'annote seul ; Élodie et Benoît vérifient un sous-échantillon ; aucun benchmark sur ces données avant leur go. Un fait structure tout le tirage : les prochaines campagnes seront toutes d'une semaine, en saison des pluies, à Mataroni, sur de nouveaux points d'écoute. Le jeu de test doit donc imiter cette situation : des points de Mataroni jamais vus à l'entraînement, enregistrés en saison des pluies. Étape 1, avant toute écoute, je répartis les points d'écoute entre entraînement et test : aucun point ne sert aux deux. Étape 2, le jeu d'entraînement : environ 450 extraits de 30 s, trois par point (pic du matin, pic du soir, une autre tranche) ; la validation, qui sert à régler la tête, se fait à l'intérieur par validation croisée groupée par point. Étape 3, le jeu de test : environ 250 enregistrements de 2 min écoutés en entier sur les points tenus à l'écart, avec au moins 60 positifs. Étape 4, la vérification à l'aveugle par Élodie et Benoît : mon label est caché et l'ordre mélangé ; environ 2 h 30 d'écoute chacun. Étape 5, le go : zéro erreur sur 60 positifs, au plus un chant manqué sur 100 négatifs. Trois principes. Aucun détecteur dans le tirage, sinon le jeu hérite de ses angles morts. Le point d'écoute, c'est-à-dire le couple site et micro, compte plus que le nombre d'enregistrements : quarante enregistrements d'un même micro partagent le même fond sonore et n'apprennent presque rien de plus qu'un seul ; mieux vaut trois enregistrements sur cent points que trois cents sur un. Enfin, positifs et négatifs sont tirés dans les mêmes conditions, même point et même tranche horaire : si les positifs venaient tous du matin et les négatifs de l'après-midi, la tête apprendrait « matin » au lieu du chant.");
  const flow = [
    ["Répartir les points", "avant écoute : chaque point va à l'entraînement ou au test, jamais aux deux"],
    ["Jeu d'entraînement", "≈ 450 extraits de 30 s, 3 par point ; validation croisée par point"],
    ["Jeu de test", "≈ 250 enregistrements entiers, points jamais vus, ≥ 60 positifs"],
    ["Vérification", "à l'aveugle, par Élodie et Benoît : ≈ 2 h 30 d'écoute chacun"],
    ["Go écrit", "0 erreur / 60 positifs, ≤ 1 chant manqué / 100"],
  ];
  const step = CW / 5;
  flow.forEach(([t, d], i) => {
    const x = MX + i * step, hot = i === 3;
    card(s, x + 0.06, 1.7, step - 0.3, 2.3, hot ? C.dark : C.paler);
    dot(s, x + 0.26, 1.86, i + 1, hot ? C.gold : C.teal, 0.42);
    tx(s, t, { x: x + 0.26, y: 2.38, w: step - 0.6, h: 0.45, fontSize: 15, bold: true, color: hot ? C.white : C.ink });
    tx(s, d, { x: x + 0.26, y: 2.85, w: step - 0.6, h: 1.1, fontSize: 12.5, color: hot ? C.sage : C.muted });
    if (i < 4) arrow(s, x + step - 0.24, 2.85, 0.18);
  });
  card(s, MX, 4.2, 5.3, 2.65, C.dark);
  label(s, "Les futures campagnes", MX + 0.25, 4.35, 4.8, C.gold);
  tx(s, "1 semaine · saison des pluies · Mataroni · nouveaux points", { x: MX + 0.25, y: 4.72, w: 4.85, h: 0.65, fontFace: HF, fontSize: 17, bold: true, color: C.white });
  tx(s, "Le jeu de test les imite : des points de Mataroni jamais vus à l'entraînement, en saison des pluies.", { x: MX + 0.25, y: 5.5, w: 4.85, h: 1.2, fontSize: 14, color: C.sage });
  const pr = [
    ["Aucun détecteur dans le tirage", "sinon le jeu hérite de ses angles morts"],
    ["Beaucoup de points, peu d'enregistrements par point", "40 enregistrements d'un même micro partagent le même fond : ils n'apprennent presque rien de plus qu'un seul"],
    ["Positifs et négatifs dans les mêmes conditions", "même point, même tranche horaire ; sinon la tête apprend « matin » au lieu du chant"],
  ];
  pr.forEach(([t, d], i) => {
    const y = 4.2 + i * 0.9;
    card(s, 6.1, y, 6.63, 0.8, C.pale);
    tx(s, [{ text: t, options: { bold: true, breakLine: true } }, { text: d, options: { color: C.muted, fontSize: 11.5 } }], { x: 6.3, y, w: 6.3, h: 0.8, fontSize: 13, valign: "middle" });
  });
}

// =====================================================================
// Module séquentiel
// =====================================================================
{
  const s = content("Module séquentiel", "Le rythme du chant, mesuré directement sur le signal",
    "Le module séquentiel mesure sur l'audio ce que l'encodeur ne voit pas dans une fenêtre de 5 s : le rythme. Une note d'A. blanci dure 0,09 à 0,10 s, dans la bande 4,4–5,5 kHz, et les notes reviennent toutes les 1,2 à 1,9 s. Le module détecte les débuts de notes dans cette bande (traits cyan) et mesure les intervalles entre elles, sans aucun apprentissage. En bas de chaque spectrogramme, l'énergie dans la bande et le seuil de détection en orange. Cliquez sur l'icône pour écouter chaque extrait de 10 s. À gauche, un chant net : 5 notes retenues, intervalle médian 1,61 s, 75 % des intervalles dans la plage d'A. blanci. Au centre, un faux ami, Adenomera andreae : même bande, intervalle médian 1,66 s, 50 % des intervalles dans la plage. Le rythme seul ne sépare donc pas les deux : c'est pour ça que le module ne met jamais de veto et que ses descripteurs sont combinés au score de la tête, qui, lui, voit la forme de la note. À droite, un chant faible au point RB04, bien audible : aucune note retenue. Les notes ne dépassent le seuil que 10 à 60 ms, alors que le filtre de durée en demande 70 à 130. À corriger avant de s'en servir : mesurer la durée à un seuil plus bas que celui qui déclenche la détection (seuil à hystérésis).");
  const items = [
    ["seq_blanci_net.png", "blanci_net.wav", "A. blanci, chant net", "5 notes · intervalle médian 1,61 s · 75 % dans la plage d'A. blanci", C.teal],
    ["seq_faux_ami.png", "faux_ami.wav", "Faux ami : A. andreae", "même bande · 1,66 s · 50 % : le rythme seul ne tranche pas", C.goldInk],
    ["seq_blanci_faible.png", "blanci_faible.wav", "A. blanci, chant faible", "audible, mais 0 note retenue : seuil de durée à revoir", C.terra],
  ];
  const cw = (CW - 0.3) / 3;
  items.forEach(([img, aud, t, d, col], i) => {
    const x = MX + i * (cw + 0.15);
    s.addImage({ path: FIG(img), x, y: 1.6, w: cw, h: cw * 660 / 1000 });
    const y = 1.6 + cw * 660 / 1000 + 0.1;
    card(s, x, y, cw, 1.35, C.paler);
    s.addMedia({ type: "audio", path: AUD(aud), x: x + 0.15, y: y + 0.22, w: 0.52, h: 0.52 });
    tx(s, t, { x: x + 0.8, y: y + 0.15, w: cw - 0.95, h: 0.4, fontSize: 14, bold: true, color: col });
    tx(s, d, { x: x + 0.8, y: y + 0.58, w: cw - 0.95, h: 0.85, fontSize: 12, color: C.ink });
  });
  card(s, MX, 5.95, CW, 0.9, C.dark);
  tx(s, rich([["L'intérêt : ", true, { color: C.gold }], ["une information que l'encodeur ne voit pas, le rythme sur tout l'enregistrement, mesurée sans apprentissage et combinée au score de la tête. Jamais de veto.", false, { color: C.white }]]), { x: MX + 0.3, y: 5.95, w: CW - 0.6, h: 0.9, fontSize: 14.5, valign: "middle" });
}

// =====================================================================
// Pré-benchmark AnuraSet (réservée)
// =====================================================================
{
  const s = content("Pré-benchmark AnuraSet", "Pré-benchmark AnuraSet",
    "Section en cours de mise à jour dans une autre session : à compléter.");
  card(s, MX, 1.9, CW, 4.9, C.paler, C.line);
  tx(s, "Section en cours de mise à jour", { x: MX, y: 3.6, w: CW, h: 0.6, fontFace: HF, fontSize: 24, bold: true, color: C.muted, align: "center" });
  tx(s, "à compléter", { x: MX, y: 4.25, w: CW, h: 0.4, fontSize: 14, color: C.sage, align: "center", italic: true });
}

// =====================================================================
// Régularisations 1/2 : rétrécissement vers une référence
// =====================================================================
{
  const s = content("Régularisations · 1/2", "Régulariser vers une référence plutôt que vers zéro",
    "Une régularisation classique, la L2, tire les poids vers zéro. Les trois méthodes présentées ici les tirent vers une solution qu'on sait déjà raisonnable : on parle de rétrécissement vers un a priori (shrinkage). En langage bayésien, c'est un a priori gaussien centré sur une référence au lieu de zéro. Un : rétrécir la logistique vers le prototype. On ajoute à la perte lambda fois la distance au carré entre les poids w et le prototype différentiel, mu plus moins mu moins. Avec lambda grand, la tête reste le prototype, qui marche avec très peu d'exemples ; avec lambda petit, elle devient une logistique libre. Lambda se règle par validation croisée groupée par point d'écoute : avec peu d'annotations sur un point neuf, la tête reste près du prototype, puis s'en écarte à mesure que les exemples arrivent. Deux : l'analyse discriminante linéaire à covariance rétrécie. Ses poids sont l'inverse de la covariance fois l'écart des moyennes. On mélange la covariance estimée S avec une covariance sphérique : gamma égal un donne exactement le prototype différentiel, gamma égal zéro une covariance entièrement apprise, proche de la logistique. Gamma se calcule directement par la formule de Ledoit et Wolf, sans réglage. Sur AnuraSet, c'est la meilleure tête pour Bird-MAE et ProtoCLR. Trois : L2-SP, pour le jour où l'on affine l'encodeur lui-même (LoRA, distillation) : on pénalise l'écart des poids du réseau aux poids pré-entraînés, au lieu de leur norme. Le réseau s'adapte à nos données sans oublier ce qu'il a appris sur 1,5 million d'enregistrements. L'outil est prêt, pour plus tard.");
  const items = [
    ["Rétrécir la logistique vers le prototype", "perte + λ ‖w − (μ₊ − μ₋)‖²", [
      "λ grand : la tête reste le prototype, qui marche avec très peu d'exemples",
      "λ petit : logistique libre, qui profite des exemples",
      "λ réglé par validation croisée par point : suit l'amorçage d'un nouveau point",
    ], "programmée"],
    ["LDA à covariance rétrécie", "w = Σ̂⁻¹ (μ₊ − μ₋)\nΣ̂ = (1 − γ) S + γ σ² I", [
      "γ = 1 : exactement le prototype différentiel",
      "γ = 0 : covariance apprise, proche de la logistique",
      "γ calculé par Ledoit-Wolf, sans réglage ; meilleure tête de Bird-MAE sur AnuraSet",
    ], "dans le benchmark des têtes"],
    ["L2-SP : rester près du pré-entraîné", "perte + λ ‖θ − θ₀‖²", [
      "pour affiner l'encodeur lui-même (LoRA, distillation)",
      "θ₀ : poids appris sur 1,5 M d'enregistrements",
      "le réseau s'adapte sans oublier ce qu'il savait",
    ], "outil prêt, pour plus tard"],
  ];
  const cw = (CW - 0.3) / 3;
  items.forEach(([t, f, bs, st], i) => {
    const x = MX + i * (cw + 0.15);
    card(s, x, 1.7, cw, 4.0, i === 1 ? C.dark : C.paler);
    const light = i === 1;
    tx(s, t, { x: x + 0.25, y: 1.88, w: cw - 0.5, h: 0.7, fontSize: 16, bold: true, color: light ? C.white : C.ink });
    tx(s, f, { x: x + 0.25, y: 2.6, w: cw - 0.5, h: 0.75, fontFace: HF, fontSize: 15, bold: true, color: light ? C.gold : C.goldInk });
    tx(s, bullets(bs, { gap: 9 }), { x: x + 0.25, y: 3.45, w: cw - 0.5, h: 1.8, fontSize: 13, color: light ? C.white : C.ink });
    tx(s, st, { x: x + 0.25, y: 5.25, w: cw - 0.5, h: 0.3, fontSize: 11.5, bold: true, italic: true, color: light ? C.sage : C.teal });
  });
  card(s, MX, 5.9, CW, 0.95, C.pale);
  tx(s, rich([["Le principe commun : ", true, { color: C.teal }], ["un a priori centré sur une solution raisonnable (prototype, modèle pré-entraîné) au lieu de zéro. Avec peu d'exemples, c'est souvent plus juste.", false]]), { x: MX + 0.3, y: 5.9, w: CW - 0.6, h: 0.95, fontSize: 14.5, valign: "middle" });
}

// =====================================================================
// Régularisations 2/2 : pour de nouveaux points d'écoute
// =====================================================================
{
  const s = content("Régularisations · 2/2", "Pour de nouveaux points d'écoute : retirer ce qui dit le site",
    "Le problème : la note d'A. blanci occupe environ 2 % d'une fenêtre de 3 s ; le reste, c'est le fond du site et du micro. Les embeddings se regroupent alors par site plutôt que par espèce, et une tête peut apprendre le site au lieu du chant. Ces régularisations retirent de l'embedding ce qui dit le point d'écoute. Centrage par point : soustraire la moyenne des embeddings du point, dans l'esprit du prototype différentiel. AdaBN : centrer et réduire chaque site. Retrait de directions : effacer les directions qui prédisent le micro, par les méthodes INLP ou LEACE ; jamais combiné au centrage par point, car la combinaison efface le chant. Biais par point : un décalage par point d'écoute, rétréci vers zéro, comme dans un modèle mixte. DANN : apprendre une représentation dont on ne peut pas deviner le micro. Mélange de fond : coller des notes sur le fond d'autres sites. Et une régularisation qui vient de la nouvelle stratégie d'annotation : noter solo ou chœur et la qualité A/B/C, puis les apprendre comme tâches auxiliaires, oblige la tête à s'appuyer sur ce qui est commun à tous les A. blanci, la note et le rythme ; essayé seulement si chaque sous-classe compte au moins 30 enregistrements indépendants. Toutes se jugent sur des points tenus à l'écart, jamais en validation croisée sur des points connus : dans ce cas, le raccourci du site reste vrai et les classements s'inversent.");
  card(s, MX, 1.75, 5.2, 2.3, C.dark);
  tx(s, [
    { text: "embedding = ", options: { color: C.white } },
    { text: "chant", options: { color: C.gold, bold: true } },
    { text: " + ", options: { color: C.white } },
    { text: "fond du point", options: { color: C.sage, bold: true } },
  ], { x: MX + 0.3, y: 1.95, w: 4.7, h: 0.55, fontFace: HF, fontSize: 22 });
  tx(s, "La note ≈ 2 % de la fenêtre : le fond domine, les embeddings se groupent par site. Retirer le fond rend le chant comparable d'un point à l'autre.", { x: MX + 0.3, y: 2.6, w: 4.65, h: 1.3, fontSize: 13.5, color: C.white });
  card(s, MX, 4.25, 5.2, 2.6, C.pale);
  label(s, "Règle", MX + 0.3, 4.4, 4.6);
  tx(s, "Toujours jugées sur des points tenus à l'écart, jamais en validation croisée sur des points connus : le raccourci du site y reste vrai.", { x: MX + 0.3, y: 4.8, w: 4.6, h: 1.9, fontSize: 15 });
  table(s, [
    ["Régularisation", "Ce qu'elle fait"],
    ["Centrage par point", "soustrait la moyenne des embeddings du point"],
    ["AdaBN par site", "centre et réduit chaque site"],
    ["Retrait de directions", "efface ce qui prédit le micro (INLP, LEACE) ; jamais avec le centrage par point"],
    ["Biais par point", "un décalage par point, rétréci vers 0 (modèle mixte)"],
    ["DANN", "représentation où le micro est indevinable"],
    ["Mélange de fond", "notes collées sur le fond d'autres sites"],
    [{ text: "Tâches auxiliaires", options: { bold: true, color: C.goldInk } }, "solo / chœur et qualité A/B/C, apprises à côté de la décision"],
  ], { x: 6.1, y: 1.75, w: 6.63, colW: [2.35, 4.28], fs: 13, rowH: 0.63 });
}

// =====================================================================
// Conclusion
// =====================================================================
{
  const s = pres.addSlide();
  pageNo += 1;
  s.background = { color: C.dark };
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: 0.55, w: 3.6, h: 0.36, fill: { color: C.dark2 }, line: { color: C.dark2, width: 0 }, rectRadius: 0.18 });
  tx(s, "D'ICI LA PROCHAINE RÉUNION", { x: MX, y: 0.55, w: 3.6, h: 0.36, fontSize: 11, bold: true, color: C.gold, align: "center", valign: "middle", charSpacing: 1 });
  tx(s, "Les deux à trois prochaines semaines", { x: MX, y: 1.05, w: 12, h: 0.8, fontFace: HF, fontSize: 32, bold: true, color: C.white });
  const steps = [
    ["Annoter mon jeu", "jeu d'entraînement (≈ 450 extraits de 30 s) et jeu de test (≈ 250 enregistrements entiers), tirés sans détecteur"],
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
  notes(s, `${pageNo + 1}. D'ici la prochaine réunion`, "D'ici la prochaine réunion, dans deux à trois semaines. Un : annoter mon jeu, le jeu d'entraînement et le jeu de test, tirés par script sans aucun détecteur. Deux : faire vérifier un sous-échantillon par Élodie et Benoît, à l'aveugle, corriger, et écrire le go. Trois : lancer le benchmark sur mes propres données, avec le même gabarit que le benchmark AnuraSet, sur des points tenus à l'écart. Quatre : présenter ce benchmark et conclure à la prochaine réunion. Deux questions pour vous : vos identifiants GitHub pour ouvrir le dépôt, et la possibilité d'encoder dans le cloud plutôt que sur mon portable ou sur les machines de l'ONF.");
}

// ---------- sorties ----------
const md = ["# Présentation de suivi n° 2 — script", "", "Notes orateur de chaque diapo (générées par `deck.js`).", ""];
script.forEach(([t, n]) => md.push(`## ${t}`, "", n, ""));
fs.writeFileSync(path.join(HERE, "script.md"), md.join("\n"));
pres.writeFile({ fileName: OUT }).then((f) => console.log("écrit :", f));
