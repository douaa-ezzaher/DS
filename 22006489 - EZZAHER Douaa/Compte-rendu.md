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


 ## 2. Le Code Python (Laboratoire)
 
# 📥 Import du dataset Kaggle (Student Mental Health)

# Utilisé pour pouvoir importer les data sources de Kaggle
import kagglehub

shariful07_student_mental_health_path = kagglehub.dataset_download("shariful07/student-mental-health")
print("Data source import complete.")


# 📥 Chargement du dataset

import kagglehub
import pandas as pd

path = kagglehub.dataset_download("shariful07/student-mental-health")

df = pd.read_csv(f"{path}/Student Mental health.csv")

print("Dataset chargé !")
print(df.shape)
df.head()


# 🧹 Pré-traitement

data = df.copy()

# Nettoyage des noms de colonnes
data.columns = (
    data.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "")
    .str.replace("?", "")
)

# Création de la target nommée mentalissue : 1 = au moins un trouble mental, 0 = aucun
data["mentalissue"] = (
    (data["doyouhavedepression"] == "Yes") |
    (data["doyouhaveanxiety"] == "Yes") |
    (data["doyouhavepanicattack"] == "Yes")
).astype(int)

print(data["mentalissue"].value_counts())


# 🔁 Suppression des lignes dupliquées

data = data.drop_duplicates()

# Variables explicatives
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


# 🧩 Vérifier les NA (données manquantes)

print(X.isna().sum())

# 1. Numérique seulement : age
X["age"] = X["age"].fillna(X["age"].median())

# 2. Toutes les autres colonnes catégorielles : mode
for col in [
    "chooseyourgender",
    "whatisyourcourse",
    "yourcurrentyearofstudy",
    "whatisyourcgpa",
    "maritalstatus",
]:
    X[col] = X[col].fillna(X[col].mode())


# 🔢 Encodage et Standardisation

import pandas as pd
from sklearn.preprocessing import StandardScaler

# One-Hot Encoding des catégorielles
X_encoded = pd.get_dummies(X, drop_first=True)

# Standardisation seulement de la colonne age
scaler = StandardScaler()
X_encoded["age"] = scaler.fit_transform(X_encoded[["age"]])

print(X_encoded.shape)
X_encoded.head()


# 📊 Visualisation : histogramme de l'âge, CGPA, statut marital

import matplotlib.pyplot as plt
import seaborn as sns

# Histogramme de l'âge
plt.figure(figsize=(5, 4))
sns.histplot(data["age"], kde=True)
plt.title("Distribution de l'âge")
plt.xlabel("Âge")
plt.ylabel("Effectif")
plt.show()

# Graphe du CGPA
plt.figure(figsize=(6, 4))
sns.countplot(x=data["whatisyourcgpa"])
plt.xticks(rotation=45)
plt.title("Répartition des catégories de CGPA")
plt.xlabel("CGPA")
plt.ylabel("Effectif")
plt.show()

# Graphe du statut marital
plt.figure(figsize=(6, 4))
sns.countplot(x=data["maritalstatus"])
plt.xticks(rotation=30)
plt.title("Répartition par statut marital")
plt.xlabel("Statut marital")
plt.ylabel("Effectif")
plt.show()


# ✂️ Split train / test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

X_train.shape, X_test.shape


# 🤖 Entraînement de plusieurs modèles de base

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


# 🔍 GridSearch sur RandomForest + évaluation

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


# 📈 Courbe ROC – AUC

from sklearn.metrics import roc_curve, roc_auc_score

# Probabilités pour la classe 1 (au moins un trouble mental)
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



