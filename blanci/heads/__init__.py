"""Étape 3 — les têtes de détection : embedding (ou signal) → score.

Sur embeddings :
- `head` : registre des têtes (prototypes, kNN, logistique, LDA, cascade...).
- `pooling` : résumé des jetons d'un encodeur pour la sonde linéaire.
- `attentive`, `gated`, `proto_probe` : sondes sur les jetons ou les dimensions.
- `losses` : une même tête linéaire, plusieurs fonctions de perte.
- `dann` : apprentissage adverse contre le micro.
- `regularization/` : toutes les régularisations (R13 à R85) et leur assemblage.
- `cluster` : voie non supervisée (HDBSCAN).
- `finetune` : emplacement du fine-tuning (à venir).

Sans encodeur :
- `signal_processing` : traitement du signal et seuillage spectral (descripteurs du signal).
- `baselines` : seuillage spectral, onsets et rythme, template matching.
- `detectors/` : détecteurs audio → score (distillation, modèle maison : à venir).
"""
