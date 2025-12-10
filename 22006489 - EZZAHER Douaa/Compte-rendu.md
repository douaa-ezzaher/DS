# 📘 GRAND GUIDE : ANATOMIE D'UN PROJET DATA SCIENCE – HEART DISEASE (UCI)
## EZZAHER Douaa - S7 - CAC G2
---

## 1. Le Contexte Métier et la Mission

### Le Problème (Business Case)

Les maladies cardiovasculaires sont l’une des premières causes de mortalité dans le monde, et le diagnostic précoce repose sur de nombreux facteurs cliniques (âge, tension artérielle, cholestérol, douleur thoracique, etc.) difficiles à interpréter ensemble.  
Un médecin doit combiner ces informations pour décider si un patient présente une maladie coronarienne significative (rétrécissement des artères coronaires) ou non.

- **Objectif :** Construire un modèle de Machine Learning qui aide à prédire la présence de maladie cardiaque à partir de données cliniques simples (données de consultation et examens de base).
- **Enjeu critique :** Le coût des erreurs est très asymétrique.
  - Dire à un patient sain qu’il est malade (**Faux Positif**) entraîne du stress, des examens invasifs (comme la coronarographie) et des coûts inutiles.
  - Dire à un patient malade qu’il est sain (**Faux Négatif**) peut conduire à un infarctus non prévenu, des séquelles graves, voire la mort.

Dans ce contexte, **le modèle doit prioriser la sensibilité (Recall)** : mieux vaut détecter le maximum de patients réellement malades, quitte à accepter un certain nombre de faux positifs, surtout si le modèle est utilisé comme système d’alerte précoce ou de “second avis” pour le clinicien.

### Les Données (L’Input)

Nous utilisons le **Heart Disease Dataset – Cleveland** issu de l’UCI Machine Learning Repository.

- **X (Features)** :  
  13–14 variables cliniques et démographiques, par exemple :

  - `age` : âge du patient en années.  
  - `sex` : sexe (1 = homme, 0 = femme).  
  - `cp` : type de douleur thoracique (angine typique, angine atypique, douleur non angineuse, asymptomatique).  
  - `trestbps` : tension artérielle au repos (en mm Hg).  
  - `chol` : cholestérol sérique (mg/dl).  
  - `fbs` : glycémie à jeun > 120 mg/dl (1 = vrai, 0 = faux).  
  - `restecg` : résultat de l’ECG au repos (normal, anomalies ST-T, hypertrophie ventriculaire).  
  - `thalach` : fréquence cardiaque maximale atteinte.  
  - `exang` : angine (douleur) induite par l’effort (1 = oui, 0 = non).  
  - `oldpeak` : dépression du segment ST induite par l’effort par rapport au repos.  
  - `slope` : pente du segment ST au pic d’effort (ascendante, plate, descendante).  
  - `ca` : nombre de vaisseaux majeurs (0–3) colorés par fluoroscopie.  
  - `thal` : résultat du test au thallium (normal, défaut fixe, défaut réversible).

  Ce sont des **données tabulaires structurées** (valeurs numériques et catégorielles), pas des images ni des signaux bruts. Elles se prêtent bien à des modèles de classification supervisée “classiques” (logistic regression, arbres de décision, random forest, etc.).

- **y (Target)** :  
  - Variable `num` : diagnostic de maladie cardiaque mesuré par angiographie, codé de 0 à 4.  
    - `0` = aucune maladie coronarienne significative.  
    - `1`, `2`, `3`, `4` = présence et sévérité croissante de la maladie.  
  - Dans la plupart des projets de Machine Learning, on convertit cela en **problème binaire** :
    - `0` : pas de maladie (classe saine).  
    - `1` : maladie présente (regroupement des valeurs 1, 2, 3, 4).

En résumé, le projet peut être formulé ainsi :

> **Mission :** prédire automatiquement, à partir des caractéristiques cliniques (X), si un patient présente ou non une maladie cardiaque significative (y binaire), en optimisant notamment le **Recall** pour limiter au maximum les Faux Négatifs (patients malades classés comme sains).

