# Annotations

Annotations de Léonard exportées par `blanci export-labels`, qui écrit les trois CSV directement
dans ce dossier (`--dossier` pour un autre). À pousser sur GitHub à la fin de chaque session
d'écoute.

L'export ne prend que les annotations de Léonard (`--annotateur` pour un autre nom) : rien
d'importé (vérifications et scores de Biophonia), rien d'un autre annotateur.

- `extraits_annotes.csv` : extraits écoutés dans le poste d'annotation (files `lot1`, `lot2`…) ;
- `intervalles_annotes.csv` : un intervalle marqué (A. blanci, chœur, faux ami…) par ligne ;
- `fenetres_annotees.csv` : fenêtres annotées une à une.

Les chemins sont ceux des enregistrements sur le disque d'origine, relatifs à sa racine
(`paths.raw` dans `config/local.yaml`). `audio.renamed` dans `config/default.yaml` donne leur nom
sur les disques BLANCI_2023-24 et BLANCI_2025-26.

`files_ecoute/` : copie des files d'écoute de `paths.queues` (16/10/2026), pour les retrouver
sur une autre machine (y pointer `paths.queues` dans `config/local.yaml`). Des références
seulement (enregistrement, début, durée), pas d'audio.

**Reprendre l'écoute sur une autre machine** : `git pull`, puis `blanci import-labels` (relit
les trois CSV de ce dossier dans la base de la machine, sans doublon ; à relancer après chaque
`git pull`). Le poste d'annotation coche alors les candidats déjà écoutés.
