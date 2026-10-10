# Annotations

Copie versionnée des annotations exportées par `blanci export-labels`. La commande écrit dans le
dossier des rapports (`paths.reports`, soit `data/reports/` par défaut, ignoré par git). La copie
dans ce dossier est manuelle : copier les trois CSV ici à la fin d'une session d'écoute, puis les
pousser sur GitHub.

L'export prend tous les annotateurs, y compris les vérifications et les scores importés, pas
seulement ceux de Léonard. La colonne `annotator` dit qui a écrit chaque ligne.

- `extraits_annotes.csv` : extraits écoutés dans le poste d'annotation (files `lot1`, `lot2`…) ;
- `intervalles_annotes.csv` : un intervalle marqué (A. blanci, chœur, faux ami…) par ligne ;
- `fenetres_annotees.csv` : fenêtres annotées une à une.

Les chemins sont ceux des enregistrements sur le disque d'origine, relatifs à sa racine
(`paths.raw` dans `config/local.yaml`).
