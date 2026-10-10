# Audit du dépôt — 10/10/2026

Plage : commits de main n° 42 → n° 189 (`aef9603` → `c69127d`, après la fusion de l'audit
précédent `daac868`), tout le dépôt, 3 agents en lecture seule, puis 4 agents de correction sur
des fichiers disjoints et une relecture par un agent distinct. Corrections sur la branche
`claude/zen-johnson-1qb3iv`.
Tests avant : 758 réussis, 0 échoué (62 lents non lancés), ruff 9 erreurs. Après : voir la fin
du §1.

## 1. Erreurs trouvées et corrigées

### Annotation par intervalles (n° 182) pas répercutée partout

Depuis le n° 182, on annote par extraits et intervalles (table `spans`). Beaucoup de code ne
lisait encore que les labels de fenêtres.

| Problème | Effet | Correction |
|---|---|---|
| Plis calculés sur les labels de fenêtres seuls | `train`, `heads`, `benchmark` plantaient dès qu'un micro n'était annoté que par intervalles (les lots du 08/10) | Plis sur labels ∪ intervalles, hors jeu gelé |
| Jeu gelé retiré des extraits mais pas des intervalles | `KeyError` dans `training_set` dès qu'un enregistrement gelé avait un extrait | Intervalles gelés retirés aussi |
| Fenêtres « A. blanci ? » et bords remises dans les négatifs | Tirées en négatifs présumés, en premier car les plus proches du chant | Jamais négatifs présumés |
| Files de candidats, carte, groupes, minage de négatifs, prévalence (R73), calibrage du QC | Enregistrements écoutés reproposés ; prévalence partielle ; un enregistrement à A. blanci pris pour négatif | Un seul critère « écoutée » (`annotation_status`) : label ou extrait |
| Contrôle qualité | Un enregistrement à A. blanci tracé en intervalle n'était pas protégé des drapeaux ; « pluie » ou « micro dans sac » dit sur un extrait ne posait rien | Les extraits protègent et posent les drapeaux ; `append_span` les applique |
| Baselines et détecteurs | Jugés sur d'autres fenêtres que les têtes | Mêmes fenêtres que les têtes |
| `freeze` | Ne comptait pas les extraits retirés de l'entraînement | Compté et affiché |

### Plan de tirage (n° 196)

| Problème | Effet | Correction |
|---|---|---|
| Relancer `candidates --plan` retirait du vivier les enregistrements écoutés depuis le tirage | Lot 1 et test v1 différents, `prob_tirage` faussées, files écrasées sans prévenir | Seuls les labels antérieurs au tirage excluent (`plan.labels_before`, 08/10) ; une file différente est refusée sans `--force` |

### Évaluation

| Problème | Effet | Correction |
|---|---|---|
| p-valeur bootstrap = 2·k/n, nulle quand aucun tirage ne traverse 0 | Environ la moitié des comparaisons de tous les benchmarks ont p = 0 et passent Holm | p = 2·(k+1)/(n+1). La plus petite p vaut alors 2/(n+1) : avec Holm sur M comparaisons, il faut n_boot ≳ 2M/α |
| `proto_probe` absent des têtes à jetons | `blanci heads` sans `--methods` plantait sur tout encodeur sans jetons, après toutes les têtes | Corrigé |
| `qc-calibrate` : micro dans sac au seuil seul | Drapeaux « actuels » surestimés | Règle des suites, comme `qc` |
| Anciennes sources `external/…` | Listées « à relancer » sans commande pour le faire | Ignorées, signalées une fois |

### Encodage

| Problème | Effet | Correction |
|---|---|---|
| Règle des suites du micro dans le sac (n° 194) contournée par `embed` et `ingest --qc` | Un enregistrement grave isolé (orage, singes) écarté et jamais encodé | Règle appliquée en fin de passage |
| Identité du stock écrite en fin de passage seulement | Premier encodage interrompu puis relancé avec un autre canal : stock mélangé sans refus | Écrite dès le premier lot |
| Identité incomplète | `window_s`, précision OpenVINO, réglages des portes pouvaient changer sans refus | Ajoutés ; un ancien stock sans ces clés reste accepté |
| `grid_hop_ratio` jamais pris | Réglage sans effet | Il l'emporte quand il est fixé |
| Table `onsets` relue en entier pour chaque enregistrement | Coût quadratique | Filtre SQL |

### Poste d'annotation et commandes

| Problème | Correction |
|---|---|
| Passe-bande : Web Audio lit le Q en dB, le code donnait un Q linéaire. Bosse de +1,9 dB à la coupure au défaut, +17 dB à 96 dB/oct | Q en dB : −3,0 dB à la coupure, sans bosse |
| `export-labels` sans l'annotateur | Colonne `annotator` |
| Aide de `label` qui proposait `--source import` (refusée) | Corrigée |
| `throughput` ignorait `config/local.yaml` | Le lit |
| Messages de `qc` et `embed` qui comptaient le micro dans sac parmi les écartés | Corrigés |

### Scripts cassés

| Problème | Correction |
|---|---|
| 7 `generer.py` de benchmark et le modèle cherchaient `tableaux/` au mauvais niveau | `parents[1]` ; modèle déplacé dans `documentation/benchmarks/` |
| `anuraset/encoder.py` : argument `signal_cfg` disparu ; `anuraset.yaml` sans `resample` | Corrigés (`resample: window`, celui des stocks existants) |
| `generer_figures.py` (prez) : `heads.sequential` renommé | `heads.signal_processing` |

### Documentation

| Problème | Correction |
|---|---|
| Rapport 08 périmé depuis audioprotopnet : il est devant naturebeats (0,907 contre 0,884) et convnext_birdset à tout k ; légende « aucune case verte » fausse | Remis à jour, écarts dits faibles |
| Branches `resultats-anuraset-*` citées pour reproduire, supprimées | Renvois vers l'historique (`git show 6f2ff13^:resultats/…`, `443f889`) |
| README : 23 encodeurs, journal « n° 1 à 165 », « aucun enregistrement » dans le dépôt | 24, n° 1 à 200, liste des extraits audio versionnés |
| `annotations/LISEZMOI.md`, `tableaux/LISEZMOI.md`, `commandes.md` inexacts | Corrigés |

### Rangement et code désuet

- `resultats/` (4,3 Mo, revenu sur main par une fusion) et `app-finale/` (une phrase, reportée dans `en_cours.yaml`) retirés.
- Sprites sortis du tableau de bord : `documentation/sprites/` ; brouillon `tomopterna/premier-jet` supprimé.
- Audits dans `documentation/audits/` ; ancienne prez sans source dans `documentation/old/`.
- Commandes `queue` et `search` retirées : `select --method active|similarity` les couvre (options manquantes ajoutées).
- Retirés car plus appelés : `SCORE_CHUNK`, `clip_spectrogram`, `snr_bins`, le doublon de `NEEDS_ENCODER`.
- `finetune.py`, détecteurs `distilled` et `homemade`, aides R61–R63 : **gardés**, marqués « EN ATTENTE » dans le code, le README et le skill d'audit.
- Tests renommés d'après ce qu'ils testent ; ruff propre.

Relecture par un agent distinct : trois défauts mineurs, corrigés. `qc-calibrate` ignorait les
extraits d'un enregistrement jamais encodé ; le compte « suggéré » du micro dans sac oubliait la
règle des suites ; un label de fenêtre et un intervalle en désaccord étaient tranchés
différemment selon le module (désormais : écoutée, jamais positive ferme, comme l'entraînement).

Tests après : 799 réussis, 0 échoué (62 lents non lancés), ruff propre.

## 2. Erreurs trouvées, non corrigées

| Point | Pourquoi |
|---|---|
| Comparaisons des benchmarks 01 à 08 : les « significatif après Holm » ne sont pas établis | Il faut les sorties hors-pli et beaucoup plus de tirages. Une note en tête de chaque rapport le dit ; les intervalles de confiance restent valables |
| Figure 4 du rapport 08 sans audioprotopnet | `generer.py` ne l'a pas dans `COURBE` ; la légende le dit, figure non régénérée |
| Trois rassembleurs AnuraSet presque identiques | Archives de reproductibilité des rapports 02–08, laissées telles quelles |
| `avex` (10 encodeurs `esp_aves2_*`) absent de `pyproject.toml` | Installé à part ; à déclarer quand ces encodeurs serviront sur les données ONF |
| L'estimation de la part d'enregistrements où A. blanci chante (`prevalence`, R73) ne se sert que des écoutes tirées au hasard uniforme ; elle laisse de côté le test v1 | Le test v1 surreprésente les heures de pic et la période haute : l'y ajouter tel quel gonflerait l'estimation. Il faudrait peser chaque enregistrement par 1 / `prob_tirage`. Rien d'urgent |

## 3. À savoir

- **Relire les rapports de benchmark à la lumière de la note sur Holm** : « après Holm » veut dire, pour l'instant, « intervalle à 95 % qui exclut 0 ».
- **Files du plan existantes (lot 1, test v1) : gardées telles quelles.** `candidates --plan` ne réécrit jamais une file existante sans `--force` ; ne pas le passer. Aucune autre commande n'y touche.
- **Relancer `blanci qc`** une fois : les drapeaux posés par extraits n'existent pas encore dans la base.
- Les stocks d'embeddings existants ne sont pas à réencoder : l'identité ancienne reste acceptée.
- `queue` et `search` n'existent plus : `select --method active` et `select --method similarity`.
- Tableau de bord à régénérer (`/tableau-de-bord`) : arborescence, commandes et rapport 08 ont changé.
