# 📘 GRAND GUIDE : ANATOMIE D'UN PROJET DATA SCIENCE
## EZZAHER Douaa - S7 - CAC G2
---

## 1. Le Contexte Métier et la Mission

### Le Problème (Business Case)
Dans le domaine universitaire, la pression académique, les difficultés financières et l'isolement social peuvent fortement impacter la santé mentale des étudiants et, par ricochet, leurs performances académiques. [web:2][web:9]

**Objectif :** Construire un modèle de type "Assistant IA" capable d'identifier les étudiants à risque de mauvaise santé mentale ou de chute de performance académique, à partir de leurs réponses à un questionnaire. [web:2][web:6]

**L'Enjeu critique :** Les coûts des erreurs de classification sont asymétriques.
*   Classer un étudiant en difficulté comme "sans problème" (**Faux Négatif**) peut conduire à un non-suivi, une aggravation des symptômes (anxiété, dépression) et un échec académique. [web:2][web:6]
*   Classer un étudiant sans difficulté majeure comme "à risque" (**Faux Positif**) peut générer une sur-sollicitation des services de soutien et une stigmatisation inutile, mais reste généralement moins grave qu'un Faux Négatif. [web:2][web:11]

**L'IA doit donc prioriser la sensibilité (Recall)** pour capter un maximum d'étudiants réellement en difficulté, tout en gardant une précision acceptable pour ne pas saturer les dispositifs de soutien. [web:11]

### Les Données (L'Input)
Nous utilisons le dataset **"Student Mental health"** publié sur Kaggle par **MD Shariful**. [web:2][web:17]
Ce jeu de données provient d'un questionnaire Google Forms rempli par des étudiants d'université, visant à analyser leur situation académique actuelle et leur santé mentale. [web:2][web:9]

*   **X (Features) :** Variables issues d'un questionnaire couvrant :
    * Informations démographiques (sexe, âge, niveau d'étude, université). [web:2][web:9]
    * Facteurs académiques (charge de travail, difficultés perçues, heures d'étude, CGPA/GPA). [web:2][web:3]
    * Indicateurs de santé mentale et mode de vie (stress académique, qualité du sommeil, anxiété, symptômes de dépression, soutien social, etc.). [web:2][web:6]

*   **y (Target) :** Binaire (selon formulation du problème).
    * `0` = **À risque** (stress élevé, dépression, faible performance académique)
    * `1` = **Pas à risque** (santé mentale stable, bonnes performances) [web:2][web:6][web:11]
