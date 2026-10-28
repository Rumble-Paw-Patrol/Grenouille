# Annotations

`files_ecoute/` : les files d'écoute.
`annotateurs.txt` : les annotateurs du projet. Sert à s'identifier sur le poste d'annotation streamlit.

Mes annotations sont exportées par `blanci export-labels` qui écrit les CSV directement dans ce dossier. À faire après chaque séance d'annotation.

- `extraits_annotes.csv` : une ligne par extrait annoté dans le poste d'annotation (files `lot1`, `lot2`…) ;
- `intervalles_annotes.csv` : une ligne par intervalle marqué (A. blanci, chœur, faux ami…) ;
- `fenetres_annotees.csv` : les fenêtres annotées une à une (méthode par fenêtres) ;
- `reannotations.csv` : un extrait réécouté par ligne, avec sa première et sa dernière version côte à côte (classes, intervalles, commentaire, date) et `n_versions`. Sert de liste de suivi pendant les corrections, et d'exclusion pour les tirages de vérification (`scripts/verification/`). Il n'est pas relu par `import-labels`.

**Réécouter un extrait** : la base est en ajout seul, une réannotation ajoute une ligne, l'ancienne reste. Dans `extraits_annotes.csv`, `remplace_par` donne le `span_id` de la dernière version d'un extrait réécouté, vide sur cette dernière et sur les extraits jamais réécoutés : **les lignes à lire sont celles où `remplace_par` est vide**. Dans `intervalles_annotes.csv`, c'est `valable`. Un extrait est « le même » s'il a le même enregistrement, les mêmes bornes (au centième) et le même annotateur ; avec d'autres bornes, c'est un autre extrait et rien ne le relie à l'ancien.


**Reprendre l'écoute sur une autre machine** : `git pull`, puis `blanci import-labels` (relit les trois CSV de ce dossier dans la base de la machine, sans doublon ; à relancer après chaque `git pull`). Le poste d'annotation coche alors les candidats déjà écoutés.
