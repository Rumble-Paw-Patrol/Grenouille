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
d'écoute, un .zip qui contient le poste, une file d'écoute, l'audio de ses extraits et ses
intervalles :

    uv run blanci --config config/local.yaml paquet doutes_lots1-2 --file lot1 --file lot2 \
        --intervalle blanci_uncertain --message "Votre avis sur mes « A. blanci ? »"

(paquet écrit sous `paths.exports/paquets/`). L'encadrant le dézippe et double-clique sur
« Ouvrir le poste d'annotation » (.bat sous Windows, .command sur Mac ; la première fois, uv
installe Python et le poste, quelques minutes, internet nécessaire). La file du paquet est
déjà ouverte, les intervalles de Léonard sont affichés à son nom ; « Reprendre les
intervalles de Léonard » quand il est d'accord. Son poste réécrit après chaque envoi ses
annotations dans le dossier `A_RENVOYER` du paquet (`Nom_extraits_annotes.csv`,
`Nom_intervalles_annotes.csv`) et le lui rappelle à la fin de la file. Il les envoie par mail
à Léonard, qui les dépose dans ce dossier (`annotations/`), lance `blanci import-labels` et
pousse sur GitHub.
L'import est sans doublon (on peut redéposer un fichier plus récent par-dessus l'ancien) et
ajoute à `annotateurs.txt` un nom qui n'y est pas encore. Dans le poste, leurs intervalles
apparaissent en lecture seule à côté des siens ; un commentaire sur un extrait de Léonard
arrive de la même façon (leur propre envoi sur cet extrait, avec son commentaire).

`annotateurs.txt` : la liste du menu « Annotateur » du poste.

**Reprendre l'écoute sur une autre machine** : `git pull`, puis `blanci import-labels` (relit
les trois CSV de ce dossier dans la base de la machine, sans doublon ; à relancer après chaque
`git pull`). Le poste d'annotation coche alors les candidats déjà écoutés.
