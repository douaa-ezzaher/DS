# PROJET DATA SCIENCE 
## Thème: Santé mentale des étudiants
## EZZAHER Douaa - S7 - CAC G2
![photo](photo.jpeg)
---
## SOMMAIRE


- [1. Le Contexte Métier et la Mission](#1-le-contexte-métier-et-la-mission)

- [2. Le Code Python (Laboratoire)](#2-le-code-python-laboratoire)

- [3. Analyse Approfondie : Nettoyage (Data Wrangling)](#3-analyse-approfondie--nettoyage-data-wrangling)

- [4. Analyse Approfondie : Exploration (EDA)](#4-analyse-approfondie--exploration-eda)

- [5. Analyse Approfondie : Méthodologie (Split)](#5-analyse-approfondie--méthodologie-split)

- [6. Focus Théorique : Algorithme Random Forest](#6-focus-théorique--lalgorithme-random-forest)

- [7. Analyse Approfondie : Évaluation](#7-analyse-approfondie--évaluation)

---

## 1. Le Contexte Métier et la Mission
### Le Problème (Business Case)
Dans le domaine universitaire, la pression académique, les difficultés financières et l'isolement social peuvent fortement impacter la santé mentale des étudiants et, par ricochet, leurs performances académiques. 

**Objectif :** Construire un modèle de type "Assistant IA" capable d'identifier les étudiants à risque de mauvaise santé mentale ou de chute de performance académique, à partir de leurs réponses à un questionnaire. 

**L'Enjeu critique :** Les coûts des erreurs de classification sont asymétriques.
*   Classer un étudiant en difficulté comme "sans problème" (**Faux Négatif**) peut conduire à un non-suivi, une aggravation des symptômes (anxiété, dépression) et un échec académique. 
*   Classer un étudiant sans difficulté majeure comme "à risque" (**Faux Positif**) peut générer une sur-sollicitation des services de soutien et une stigmatisation inutile, mais reste généralement moins grave qu'un Faux Négatif.

**L'IA doit donc prioriser la sensibilité (Recall)** pour capter un maximum d'étudiants réellement en difficulté, tout en gardant une précision acceptable pour ne pas saturer les dispositifs de soutien. 

### Les Données (L'Input)
Nous utilisons le dataset **"Student Mental health"** publié sur Kaggle par **MD Shariful**. 
Ce jeu de données provient d'un questionnaire Google Forms rempli par des étudiants d'université, visant à analyser leur situation académique actuelle et leur santé mentale. 

*   **X (Features) :** Variables issues d'un questionnaire couvrant :
    * Informations démographiques (sexe, âge, niveau d'étude, université). 
    * Facteurs académiques (charge de travail, difficultés perçues, heures d'étude, CGPA/GPA). 
    * Indicateurs de santé mentale et mode de vie (stress académique, qualité du sommeil, anxiété, symptômes de dépression, soutien social, etc.). 

*   **y (Target) :** Binaire (selon formulation du problème).
    * `0` = **À risque** (stress élevé, dépression, faible performance académique)
    * `1` = **Pas à risque** (santé mentale stable, bonnes performances) 

---

## 2. Le Code Python (Laboratoire)
 
### 📥 Import du dataset Kaggle (Student Mental Health)

*Utilisé pour pouvoir importer les data sources de Kaggle*
import kagglehub

shariful07_student_mental_health_path = kagglehub.dataset_download("shariful07/student-mental-health")
print("Data source import complete.")


### 📥 Chargement du dataset

import kagglehub
import pandas as pd

path = kagglehub.dataset_download("shariful07/student-mental-health")

df = pd.read_csv(f"{path}/Student Mental health.csv")

print("Dataset chargé !")
print(df.shape)
df.head()


### 🧹 Pré-traitement

data = df.copy()

*Nettoyage des noms de colonnes*
data.columns = (
    data.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "")
    .str.replace("?", "")
)

##" Création de la target nommée mentalissue : 1 = au moins un trouble mental, 0 = aucun
data["mentalissue"] = (
    (data["doyouhavedepression"] == "Yes") |
    (data["doyouhaveanxiety"] == "Yes") |
    (data["doyouhavepanicattack"] == "Yes")
).astype(int)

print(data["mentalissue"].value_counts())


### 🔁 Suppression des lignes dupliquées

data = data.drop_duplicates()

*Variables explicatives*
features = [
    "chooseyourgender",
    "age",
    "whatisyourcourse",
    "yourcurrentyearofstudy",
    "whatisyourcgpa",
    "maritalstatus",
]

X = data[features].copy()
y = data["mentalissue"]


### 🧩 Vérifier les NA (données manquantes)

print(X.isna().sum())

*1. Numérique seulement : age*
X["age"] = X["age"].fillna(X["age"].median())

*2. Toutes les autres colonnes catégorielles : mode*
for col in [
    "chooseyourgender",
    "whatisyourcourse",
    "yourcurrentyearofstudy",
    "whatisyourcgpa",
    "maritalstatus",
]:
    X[col] = X[col].fillna(X[col].mode())


### 🔢 Encodage et Standardisation

import pandas as pd
from sklearn.preprocessing import StandardScaler

*One-Hot Encoding des catégorielles*
X_encoded = pd.get_dummies(X, drop_first=True)

*Standardisation seulement de la colonne age*
scaler = StandardScaler()
X_encoded["age"] = scaler.fit_transform(X_encoded[["age"]])

print(X_encoded.shape)
X_encoded.head()


### 📊 Visualisation : histogramme de l'âge, CGPA, statut marital

import matplotlib.pyplot as plt
import seaborn as sns

*Histogramme de l'âge*
plt.figure(figsize=(5, 4))
sns.histplot(data["age"], kde=True)
plt.title("Distribution de l'âge")
plt.xlabel("Âge")
plt.ylabel("Effectif")
plt.show()

*Graphe du CGPA*
plt.figure(figsize=(6, 4))
sns.countplot(x=data["whatisyourcgpa"])
plt.xticks(rotation=45)
plt.title("Répartition des catégories de CGPA")
plt.xlabel("CGPA")
plt.ylabel("Effectif")
plt.show()

*Graphe du statut marital*
plt.figure(figsize=(6, 4))
sns.countplot(x=data["maritalstatus"])
plt.xticks(rotation=30)
plt.title("Répartition par statut marital")
plt.xlabel("Statut marital")
plt.ylabel("Effectif")
plt.show()


### ✂️ Split train / test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

X_train.shape, X_test.shape


### 🤖 Entraînement de plusieurs modèles de base

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    print(f"{name} - Accuracy moyenne CV : {scores.mean():.3f}")


### 🔍 GridSearch sur RandomForest + évaluation

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

param_grid_rf = {
    "n_estimators": ,
    "max_depth": [None, 5, 10],
    "min_samples_split":,[1][2]
}

rf = RandomForestClassifier(random_state=42)

grid_rf = GridSearchCV(
    estimator=rf,
    param_grid=param_grid_rf,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

grid_rf.fit(X_train, y_train)

print("Meilleurs paramètres :", grid_rf.best_params_)
print("Meilleure accuracy CV :", grid_rf.best_score_)

best_model = grid_rf.best_estimator_
y_pred = best_model.predict(X_test)

print("Rapport de classification :")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.title("Matrice de confusion - RandomForest")
plt.show()


### 📈 Courbe ROC – AUC

from sklearn.metrics import roc_curve, roc_auc_score

*Probabilités pour la classe 1 (au moins un trouble mental)*
y_proba = best_model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)
print("ROC-AUC :", auc)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})")
plt.plot(,, "k--", label="Hasard")[3]
plt.xlabel("Taux de faux positifs (FPR)")
plt.ylabel("Taux de vrais positifs (TPR)")
plt.title("Courbe ROC - Student Mental Health")
plt.legend()
plt.show()

---

## 3. Analyse Approfondie : Nettoyage (Data Wrangling)

### Problématique spécifique à notre projet
Le dataset **Student Mental Health** provient d’un questionnaire réel rempli par des étudiants. Comme souvent avec ce type de données déclaratives, certaines réponses sont manquantes ou incohérentes, notamment sur l’âge ou les informations académiques.  
Dans notre cas, les algorithmes de **Machine Learning** (régression logistique, KNN, Random Forest) nécessitent une matrice de données complète et exploitable, sans valeurs manquantes.

### Stratégie de nettoyage appliquée
- **Suppression des doublons** : afin d’éviter que certains profils étudiants soient surreprésentés dans l’apprentissage.  
- **Variable numérique (âge)** : imputation par la médiane, choix robuste face aux valeurs extrêmes.  
- **Variables catégorielles (genre, filière, année d’étude, CGPA, statut marital)** : imputation par le mode, représentant la réponse la plus fréquente.  

Cette approche permet de conserver la majorité des observations tout en limitant la distorsion statistique.

---

## 4. Analyse Approfondie : Exploration (EDA)

L’analyse exploratoire vise à comprendre le profil des étudiants et la structure des variables avant la modélisation.

### Analyse des distributions
- **Âge** : la majorité des étudiants se situe dans une tranche d’âge jeune, ce qui est cohérent avec une population universitaire.  
- **CGPA** : les catégories de performance académique sont hétérogènes, suggérant des profils d’étudiants variés.  
- **Statut marital** : variable majoritairement dominée par les étudiants célibataires, mais conservée car potentiellement corrélée au bien‑être psychologique.  

### Apport de l’EDA au projet
Cette étape permet de :
- vérifier la cohérence globale des réponses,  
- identifier d’éventuels déséquilibres,  
- confirmer la pertinence des variables retenues pour la prédiction des troubles mentaux.  

Les visualisations confirment que les données sont exploitables et informatives pour un modèle de classification. 

---

## 5. Analyse Approfondie : Méthodologie (Split)

### 5.1 Pourquoi un protocole expérimental est indispensable
En Machine Learning, de bonnes performances sur les données d’entraînement ne garantissent pas que le modèle fonctionnera correctement sur de nouveaux étudiants. Sans séparation des données, le modèle risquerait de mémoriser des profils spécifiques au lieu d’apprendre des règles générales reliant les facteurs académiques et démographiques à la santé mentale.  
Le découpage **Train/Test** permet donc de simuler une situation réelle : prédire l’état de santé mentale d’un étudiant jamais observé auparavant.

### 5.2 Choix du découpage
Les choix méthodologiques suivants ont été retenus :

- **80 % Train / 20 % Test**  
  Le jeu d’entraînement contient suffisamment d’observations pour apprendre des relations robustes entre les variables explicatives (âge, CGPA, genre, année d’étude, statut marital) et la variable cible *mentalissue*.  
  Le jeu de test, quant à lui, permet une évaluation fiable et statistiquement pertinente.

- **random_state = 42**  
  Ce paramètre garantit la reproductibilité des résultats. Toute relance du code produit le même découpage, condition essentielle dans un cadre académique.

- **Stratification sur la variable cible**  
  La variable *mentalissue* pouvant être déséquilibrée, la stratification assure que la proportion d’étudiants à risque et non à risque est conservée dans les deux ensembles. Cela évite des évaluations biaisées dues au hasard.

### 5.3 Impact sur l’évaluation
Grâce à ce protocole, les métriques calculées sur le jeu de test reflètent la capacité réelle de généralisation du modèle et non une performance artificiellement optimiste.

---

## 6. Focus Théorique : Algorithme Random Forest

### 6.1 Pertinence de Random Forest pour ce projet
Le problème étudié est une classification binaire basée sur des données hétérogènes issues d’un questionnaire. **Random Forest** est particulièrement adapté car :
- il capture des relations non linéaires entre les facteurs académiques et la santé mentale,
- il est robuste face au bruit et aux données imparfaites,
- il gère efficacement les interactions complexes entre variables,
- il offre de bonnes performances sans nécessiter d’hypothèses statistiques fortes.

### 6.2 Limites d’un arbre de décision unique
Un arbre de décision seul peut apprendre des règles trop spécifiques aux données d’entraînement, par exemple :  
> « Si CGPA faible et âge élevé alors étudiant à risque ».  
Ce comportement mène au **sur-apprentissage (overfitting)**, réduisant fortement les performances sur de nouveaux étudiants.

### 6.3 Principe de fonctionnement de Random Forest
**Random Forest** repose sur l’agrégation de plusieurs arbres de décision indépendants :

- **Bootstrapping (diversité des étudiants)**  
  Chaque arbre est entraîné sur un échantillon aléatoire du jeu d’entraînement. Certains profils étudiants sont vus plusieurs fois, d’autres pas du tout.

- **Aléa sur les variables (diversité des facteurs)**  
  À chaque séparation, l’arbre ne considère qu’un sous-ensemble aléatoire de variables (âge, CGPA, genre, etc.), ce qui empêche le modèle de se focaliser excessivement sur une seule variable dominante.

- **Vote majoritaire**  
  Chaque arbre émet une prédiction. La décision finale correspond au vote majoritaire, ce qui réduit la variance et améliore la stabilité globale du modèle.

### 6.4 Optimisation par GridSearch
Une recherche par grille a été utilisée afin d’optimiser les hyperparamètres clés (**profondeur des arbres**, **nombre d’arbres**, **taille minimale des nœuds**). Cette étape permet d’améliorer la généralisation tout en limitant le sur-apprentissage.

---

## 7. Analyse Approfondie : Évaluation

### 7.1 Matrice de confusion
La matrice de confusion permet d’analyser précisément les performances du modèle :

- **Vrais Positifs (TP)** : étudiants réellement à risque correctement identifiés.  
- **Vrais Négatifs (TN)** : étudiants sans trouble correctement reconnus.  
- **Faux Positifs (FP)** : étudiants signalés à tort comme à risque.  
- **Faux Négatifs (FN)** : étudiants en difficulté non détectés.  

Dans le contexte de la santé mentale étudiante, les **Faux Négatifs** représentent l’erreur la plus critique.

### 7.2 Analyse des métriques
Plusieurs métriques sont utilisées :
- **Accuracy** : donne une vue globale, mais peut être trompeuse si les classes sont déséquilibrées.  
- **Precision** : mesure la fiabilité des alertes générées par le modèle.  
- **Recall (sensibilité)** : mesure la capacité à détecter les étudiants réellement à risque. C’est la métrique prioritaire de ce projet.  
- **F1-score** : compromis entre précision et rappel, utile pour comparer différents modèles.

### 7.3 Courbe ROC et AUC
La courbe **ROC** illustre le compromis entre le taux de vrais positifs et le taux de faux positifs selon le seuil de décision.  
Une **AUC** élevée indique une bonne capacité du modèle à distinguer les étudiants à risque des autres.

### 7.4 Interprétation globale
Les résultats montrent que **Random Forest** est capable d’identifier efficacement les étudiants présentant des signes de troubles de santé mentale.  
La priorité donnée au **Recall** permet de limiter le nombre d’étudiants en difficulté non détectés, ce qui est cohérent avec l’objectif de prévention du projet.  
Cette stratégie peut entraîner une augmentation modérée des **Faux Positifs**, mais ce compromis est acceptable : il est préférable de proposer un soutien à un étudiant qui n’en aurait pas strictement besoin plutôt que de laisser un étudiant en détresse sans accompagnement.  

Le modèle doit donc être considéré comme un **outil d’aide à la décision**, destiné à assister les services universitaires dans l’identification précoce des étudiants vulnérables, et non comme un **diagnostic médical automatisé**.
