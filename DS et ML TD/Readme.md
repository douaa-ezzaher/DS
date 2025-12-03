# DS et ML TD  
Réalisé par : EZZAHER DOUAA – S7 CAC G2  

## Introduction 

Le jeu de données utilisé porte sur la santé mentale d’étudiants universitaires. Chaque ligne correspond à un étudiant et contient des informations simples sur son profil (âge, genre, études, CGPA, etc.) ainsi que des réponses indiquant s’il souffre de dépression, d’anxiété ou de panic attack. 
L’objectif est d’utiliser ces informations pour mieux comprendre les profils d’étudiants susceptibles de présenter des troubles psychologiques.

### Problématique et type de modèle

La question posée est : peut‑on prédire si un étudiant présente au moins un trouble de santé mentale à partir de ses caractéristiques personnelles et académiques ? 
Ce problème est traité comme une tâche de classification binaire supervisée: La variable cible vaut 1 si l’étudiant déclare au moins un trouble (dépression, anxiété ou panic attack) et 0 s’il ne déclare aucun de ces troubles.

### Dictionnaire 
Le dataset contient un nombre limité d’étudiants, chacun décrit par plusieurs variables qualitatives et quantitatives. On y trouve par exemple : genre (catégoriel), âge (numérique), filière et année d’étude (catégoriel), CGPA (numérique) et statut marital (catégoriel), ainsi que des colonnes booléennes indiquant la présence ou non de troubles mentaux. 
La target utilisée dans ce projet est une nouvelle variable binaire construite à partir de ces colonnes de troubles.

### Variables explicatives choisies

Les variables explicatives retenues sont :  
- Gender (genre)  
- Age (âge)  
- Course (filière / cursus)  
- Year of Study (année d’étude)  
- CGPA (moyenne générale)  
- Marital Status (statut marital)  

Ces variables servent d’entrées au modèle de classification pour prédire la présence d’au moins un trouble mental.

## Méthodologie

### Pré-traitement des données

Avant d’entraîner les modèles, le jeu de données a été nettoyé.  
Les doublons ont été supprimés et les noms de colonnes simplifiés pour être plus faciles à utiliser dans le code.  
Les valeurs manquantes ont été traitées de façon simple : l’âge a été remplacé par la médiane, et les variables qualitatives (genre, filière, année d’étude, CGPA, statut marital) par la valeur la plus fréquente.  
Ensuite, les variables catégorielles ont été transformées en variables numériques avec un encodage One-Hot, et la variable numérique `age` a été standardisée pour que son échelle soit comparable aux autres.

### Choix des algorithmes

Comme la variable cible est binaire (0 = pas de trouble, 1 = au moins un trouble), trois algorithmes de classification classiques ont été testés :

- Régression logistique : modèle simple de référence pour la classification binaire.
- Random Forest : modèle d’arbres de décision qui peut capturer des relations plus complexes.
- K-Nearest Neighbors (KNN) : modèle basé sur la similarité entre les étudiants.

Ces trois modèles ont été choisis car ils sont connus, présents dans les cours, et adaptés à un petit jeu de données.

### Validation et optimisation

Pour évaluer les modèles, une validation croisée (k-fold) a été utilisée sur l’échantillon d’entraînement.  
Cette méthode permet de calculer une accuracy moyenne plus fiable qu’un simple train/test.  
Après comparaison des trois modèles, la Random Forest a été sélectionnée comme modèle principal.  
Ses hyperparamètres (par exemple le nombre d’arbres et la profondeur maximale) ont ensuite été ajustés avec `GridSearchCV` afin d’essayer plusieurs combinaisons et de garder celle qui donne la meilleure performance en validation croisée.

## Résultats & discussion

### 1. Synthèse de l’analyse exploratoire (EDA)

L’histogramme de la distribution de l’âge montre deux groupes principaux d’étudiants, autour de 18–19 ans et de 23–24 ans, avec très peu d’observations entre 21 et 22 ans.  
![Distribution de l’âge](DS%20et%20ML%20TD/images/age_distribution.png)

Le graphique de répartition des catégories de CGPA indique que la plupart des étudiants se situent dans les tranches élevées (3.00–3.49 et 3.50–4.00), tandis que les catégories de CGPA plus faibles sont beaucoup moins représentées.  
![Répartition des catégories de CGPA](DS%20et%20ML%20TD/images/cgpa_distribution.png)

La répartition de la variable `mental_issue` montre qu’il y a plus d’étudiants qui déclarent au moins un trouble mental que d’étudiants sans trouble, ce qui confirme que les problèmes de santé mentale sont fréquents dans cet échantillon.  
![Présence d’au moins un trouble mental](DS%20et%20ML%20TD/images/mental_issue_dist.png)

Les graphiques “trouble mental selon le genre” et “trouble mental selon l’année d’étude” suggèrent que les troubles sont présents dans les deux genres et à différents niveaux d’étude, avec parfois un nombre plus élevé de cas déclarés chez les étudiantes ou dans certaines années.  

![Trouble mental selon le genre](DS%20et%20ML%20TD/images/mental_issue_gender.png)
![Trouble mental selon l'année d'étude](DS%20et%20ML%20TD/images/mental_issue_year.png)

Enfin, les heatmaps de corrélation (entre l’âge ou le CGPA numérique et `mental_issue`) montrent des corrélations faibles, ce qui laisse penser qu’aucune de ces variables numériques ne suffit à elle seule pour expliquer la présence d’un trouble mental.  
![Heatmap Age / trouble mental](DS%20et%20ML%20TD/images/heatmap_age_mental.png)
![Heatmap CGPA / trouble mental](DS%20et%20ML%20TD/images/heatmap_cgpa_mental.png)

### 2. Performances des modèles de classification

Trois modèles de classification ont été entraînés sur la variable cible `mental_issue` : régression logistique, Random Forest et K‑Nearest Neighbors (KNN).  
Pour chacun, une validation croisée (k‑fold) a été utilisée sur l’ensemble d’entraînement afin de calculer une accuracy moyenne plus stable. Les résultats obtenus montrent que la Random Forest est le modèle le plus performant, avec une accuracy d’environ **XX %** et un F1‑score de **YY** sur le jeu de test (valeurs à compléter depuis le notebook). Les modèles de régression logistique et KNN obtiennent des scores légèrement inférieurs, ce qui confirme que la Random Forest s’adapte mieux à la structure des données.

La matrice de confusion de la Random Forest permet d’analyser les erreurs plus en détail. On observe que la plupart des étudiants avec trouble mental (classe 1) sont correctement prédits, mais que le modèle a tendance à prédire un trouble mental pour certains étudiants qui n’en déclarent pas (faux positifs). Cela signifie que le modèle privilégie la détection des cas à risque, quitte à “sur‑détecter” quelques étudiants sans trouble.  
![Matrice de confusion Random Forest](DS%20et%20ML%20TD/images/confusion_rf.png)

Concernant la courbe ROC, la diagonale en pointillés correspond à un classifieur aléatoire. La courbe bleue se situe au‑dessus de cette diagonale, avec une AUC de 0,60 : le modèle fait mieux que le hasard, mais sa performance reste modérée, ce qui s’explique notamment par la petite taille du dataset et le faible nombre de variables.
![Courbe ROC du meilleur modèle](DS%20et%20ML%20TD/images/roc_best_model.png)

## Conclusion

Ce projet montre qu’à partir d’un jeu de données simple sur les étudiants, il est possible d’utiliser des modèles de Machine Learning pour prédire la présence d’au moins un trouble de santé mentale. Les graphes d’EDA ont mis en évidence la structure de la population (répartition des âges, des CGPA, des genres et des années d’étude) ainsi que la fréquence élevée des troubles déclarés.

Parmi les modèles testés, la Random Forest obtient les meilleures performances globales et constitue un bon compromis entre précision et simplicité. Cependant, plusieurs limites subsistent : le dataset est de petite taille, certaines classes sont peu représentées et les variables utilisées restent assez générales. Pour améliorer le modèle, il serait intéressant de disposer de plus de données, d’ajouter des variables plus détaillées sur le vécu psychologique des étudiants et de tester d’autres algorithmes ou techniques d’équilibrage des classes.
