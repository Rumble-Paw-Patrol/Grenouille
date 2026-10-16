# Annotations de Léonard

Écrites par `blanci export-labels` à la fin de chaque session d'écoute, puis poussées sur GitHub.
Seulement les annotations de Léonard : rien d'importé (vérifications et scores BlanciNet de
Biophonia, qui restent dans la base, hors git).

- `extraits_annotes.csv` : extraits écoutés dans le poste d'annotation (files `lot1`, `lot2`…) ;
- `intervalles_annotes.csv` : un intervalle marqué (A. blanci, chœur, faux ami…) par ligne ;
- `fenetres_annotees.csv` : fenêtres annotées une à une.

Les chemins sont ceux des enregistrements sur le disque d'origine (`audio.renamed` dans
`config/default.yaml` donne leur nom sur les disques BLANCI_2023-24 et BLANCI_2025-26).

`files_ecoute/` : copie des files d'écoute de `paths.queues` (16/10/2026), pour les retrouver
sur une autre machine (y pointer `paths.queues` dans `config/local.yaml`). Des références
seulement (enregistrement, début, durée), pas d'audio.
