# Annotations

`files_ecoute/` : les files d'écoute.

Mes annotations sont exportées par `blanci export-labels` qui écrit les deux CSV directement dans ce dossier. À faire après chaque séance d'annotation.

- `extraits_annotes.csv` : une ligne par extrait annoté dans le poste d'annotation (files `lot1`, `lot2`…) ;
- `intervalles_annotes.csv` : une ligne par intervalle marqué (A. blanci, chœur, faux ami…) par ligne ;

`annotateurs.txt` : les annotateurs du projet. Sert à s'identifier sur le poste d'annotation streamlit.

**Reprendre l'écoute sur une autre machine** : `git pull`, puis `blanci import-labels` (relit les trois CSV de ce dossier dans la base de la machine, sans doublon ; à relancer après chaque `git pull`). Le poste d'annotation coche alors les candidats déjà écoutés.
