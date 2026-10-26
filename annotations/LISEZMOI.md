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

**Annotations des encadrants (sans GitHub ni disques)** : Léonard leur envoie un paquet
d'écoute, un .zip qui contient une page web (le poste), la file d'écoute, l'audio de ses
extraits et ses intervalles :

    uv run blanci --config config/local.yaml paquet doutes_lots1-2 --file lot1 --file lot2 \
        --intervalle blanci_uncertain --message "Votre avis sur mes « A. blanci ? »"

(paquet écrit sous `paths.exports/paquets/`, à déposer sur un drive : ~50 Mo pour 18
extraits de 30 s). L'encadrant le dézippe et double-clique sur « Ouvrir le poste.html » :
rien à installer, le poste s'ouvre dans le navigateur (Chrome ou Edge de préférence). Les
intervalles de Léonard sont affichés à son nom ; « Reprendre les intervalles de Léonard »
quand il est d'accord. Ses réponses sont gardées dans le navigateur et, avec Chrome ou Edge,
réécrites après chaque envoi dans un dossier qu'il choisit au premier envoi ; ailleurs,
« Télécharger mes réponses ». À la fin, la page lui dit quoi envoyer
(`Nom_extraits_annotes.csv`, `Nom_intervalles_annotes.csv`). Il les envoie par mail à
Léonard, qui les dépose dans ce dossier (`annotations/`), lance `blanci import-labels` et
pousse sur GitHub.
L'import est sans doublon (on peut redéposer un fichier plus récent par-dessus l'ancien) et
ajoute à `annotateurs.txt` un nom qui n'y est pas encore. Dans le poste, leurs intervalles
apparaissent en lecture seule à côté des siens ; un commentaire sur un extrait de Léonard
arrive de la même façon (leur propre envoi sur cet extrait, avec son commentaire).

`annotateurs.txt` : la liste du menu « Annotateur » du poste.

**Reprendre l'écoute sur une autre machine** : `git pull`, puis `blanci import-labels` (relit
les trois CSV de ce dossier dans la base de la machine, sans doublon ; à relancer après chaque
`git pull`). Le poste d'annotation coche alors les candidats déjà écoutés.
