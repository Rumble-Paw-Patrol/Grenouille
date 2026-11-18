# Présentation n° 4 en cinq paquets

La présentation n° 4 (inventaire et contrôle qualité, v2, 26 diapos) est revue paquet par paquet, une session par paquet, pour la réunion du **20 novembre 2026**. Les paquets ont été découpés le 18/11 dans `../presentation_4_inventaire_qc_v2.pptx` (version avec les retouches à la main de Léonard ; les paquets 2 à 5 redécoupés à 11 h 33, après ses retouches des diapos 7 et 11), par `../scripts/decouper_pptx.py`. Le fichier complet n'est plus modifié par les sessions : chacune ne touche que son paquet.

| Paquet | Diapos | Fichier | État |
|---|---|---|---|
| 1 | 1 à 5, plus 5 bis | `presentation_4_inventaire_qc_v2_paquet_1_diapos_01-05.pptx` | corrigé le 18/11, à relire par Léonard |
| 2 | 6, 7, 9, 9 bis, 9 ter, 11, 10, 8 | `…_paquet_2_diapos_06-11.pptx` | fait le 18/11 (`../scripts/presentation_4_paquet_2.py`), à relire par Léonard |
| 3 | 12 à 15 | `…_paquet_3_diapos_12-15.pptx` | consignes à dicter |
| 4 | 16 à 20 | `…_paquet_4_diapos_16-20.pptx` | consignes à dicter |
| 5 | 21 à 26 (26 = références) | `…_paquet_5_diapos_21-26.pptx` | consignes à dicter |

## Règles communes

- **Numéros** : chaque diapo garde son numéro d'origine (écrit en dur en bas à droite) ; la diapo 6 reste la 6 dans le paquet 2. Une diapo ajoutée prend le numéro de celle qui la précède suivi de « bis », « ter » (diapo « 5 bis », pastille « INVENTAIRE · 3 bis/7 »). La renumérotation se fait une seule fois, au réassemblage.
- **Retouches de Léonard** : il modifie les .pptx à la main. Ne jamais régénérer un paquet ni écraser ce qu'il a écrit ; si une de ses retouches est fausse, le lui dire dans la réponse au lieu de la corriger en silence. Ne jamais piloter PowerPoint (COM) ni écrire dans un fichier qu'il a ouvert (fichier verrou `~$…`).
- **Modifier un paquet** : en éditant le XML du .pptx (bibliothèque standard, ou python-pptx si le réseau permet de l'installer), à partir du paquet tel qu'il est sur le disque. Le script du paquet 1 (forme par forme, en gardant les formes non touchées) est un modèle : il est décrit plus bas.
- **Figures** : `../scripts/presentation_4_figures.py`, à compléter par chaque session (une fonction par figure), qui lit la base en lecture seule et écrit dans `../figures/inventaire_qc/`. Chaque figure est dessinée à la taille exacte de sa place sur la diapo (1 pt de police = 1 pt affiché), en Calibri, titres de graphique en gras 13-14 pt, légendes 11 pt au moins : c'est ce que Léonard demande (« titres des graphiques et légendes plus visibles »).
- **Encadrés** : plus de fond vert foncé avec texte jaune (« moche et illisible ») ; fond pâle `F3F7F4` ou `E4EEE8`, texte `1B2A24`, intertitres en `2A7F72` (vert) ou `9A6B0F` (or foncé lisible sur fond clair). Marges 0,6 in, contenu à partir de 1,75 in, rien sous 6,95 in.
- **Chiffres** : recalculés dans la base du jour, jamais recopiés. Le 18/11 : 96 310 enregistrements (2023 et 2026 ; la table en a 96 349 avec 39 fichiers de test de 2018 et de référence), 3 207 h, 164 points (site, micro), 95 enregistreurs ; l'ancien 96 292 est périmé. Léonard réannote des enregistrements écartés à tort : le nombre d'écartés (681) va bouger, à recompter dans les paquets 2, 4 et 5 et à mettre d'accord avec le tableau de bord.
- **Sites** : CDR, Patawa Est et Patawa Ouest sont des sous-sites de Kaw, RNRT de Trésor (`lieux` dans `documentation/tableau-de-bord/structure.yaml`). Quatre sites : Kaw, Trésor, Molokoï, Mataroni.
- **Notes orateur** : mises à jour dans le .pptx et reportées dans `../scripts/presentation_4_inventaire_qc_v2_script.md` (une section par diapo, « 5 bis » compris). Méthodologie à chaque sujet : effectif, loi ou référence, limite.
- **Partie « Difficultés rencontrées »** : annoncée sur la diapo de titre ; elle regroupe en fin de présentation les erreurs passées (ancienne règle de la diapo 8, « ce que j'aurais dû faire », diapo 23…). Les paquets 2 à 5 y renvoient ce qui relève d'une erreur passée au lieu de le montrer au fil de l'eau.
- **Réassemblage** : dans PowerPoint, Accueil > Nouvelle diapositive > Réutiliser les diapositives, en gardant la mise en forme source, paquet par paquet ; puis renuméroter.

## Paquet 1 (fait le 18/11)

- Diapo 1 : sous-titre « Inventaire du corpus · contrôle qualité · difficultés rencontrées : … », date du 20 novembre 2026.
- Diapo 3 : encadré « Comment l'inventaire est fait » retiré ; à sa place, les enregistrements par programme (5 h → 19 h 30, 30 par jour : suivi 2023-24, CDR, presque tout Patawa Ouest ; 6 h → 18 h 30, 26 par jour : Mataroni, presque tout RNRT et Patawa Est). À gauche, les relevés empilés sur leur site.
- Diapo 4 : calendrier à axe coupé, relevés sur la ligne de leur site. Vu en passant : 25 enregistrements de 12 micros les 30 et 31/12/2025 dans le relevé de Mataroni, six jours avant la pose, non marqués hors relevé (trou de 6 jours, la règle en demande plus de 7) ; à vérifier, à signaler au paquet 2 (diapo 10).
- Diapo 5 : trois cartes, le fichier brut dans un terminal (vraie commande : `../scripts/entete_wav.py`, sortie GUANO réelle lue le 05/11 dans le notebook 05), la ligne de `recordings`, ce que l'inventaire en tire (même enregistreur 2LA04185 partout).
- Diapo 5 bis (nouvelle) : vue de la table `recordings` (vraies lignes, dessinées par `presentation_4_figures.py`, pas une capture d'écran d'un logiciel) et chemin de la base.

## Paquet 2

Voir `plan_paquet_2.md` : ordre cible (6, 7, 9, changement de partie, sommaire du contrôle qualité, 11, 10, 8), consignes de Léonard recopiées, et la figure de Molokoï retrouvée dans `sources_paquet_2/`.

## Paquets 3 à 5

Consignes à dicter par Léonard. Déjà connu : la passation « notion 7 » (diapos 18, 22, 23, 24), le recomptage des écartés, et pour le paquet 5 la partie « Difficultés rencontrées », alimentée par la section « Pour le paquet 5 » de `plan_paquet_2.md`. Paquet 3 : la diapo 12 montre le dictionnaire `qc_flags` de 2LA04185 ; la diapo 11 (paquet 2) montre désormais une ligne avec un drapeau, éviter le doublon.
