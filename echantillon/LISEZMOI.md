# Échantillon de test

66 clips d'environ 10 s (FLAC, 48 kHz stéréo), tirés des fenêtres étiquetées de `data/db/blanci.sqlite` :
10 au plus par classe, un enregistrement différent par clip autant que possible (tirage aléatoire, graine 0).
Chaque clip couvre la fenêtre étiquetée plus 3,5 s de contexte de part et d'autre.

`labels.csv` : label, qualité, espèce, site, micro, chemin source (relatif à la racine du disque),
début du clip dans la source, début et durée de la fenêtre étiquetée dans le clip.

Classes rares (moins de 10 étiquettes disponibles) : amphibian_contact_call 4, background 2,
orthoptera 3, rain 6, uncertain 1.
