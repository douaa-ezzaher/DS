# PROJET DATA SCIENCE 
## Thème: Santé mentale des étudiants
## EZZAHER Douaa - S7 - CAC G2
![photo](photo.jpeg)
---
## SOMMAIRE
## Sommaire

1. Contexte métier et mission  
   1.1 Problème (Business Case)  
   1.2 Les données (Input)  

2. Code Python (Laboratoire)  
   2.1 Import et chargement du dataset Kaggle  
   2.2 Pré-traitement et création de la cible  
   2.3 Gestion des valeurs manquantes  
   2.4 Encodage et standardisation  
   2.5 Visualisations exploratoires (EDA)  
   2.6 Split train / test  
   2.7 Entraînement des modèles  
   2.8 GridSearch et évaluation du Random Forest  
   2.9 Courbe ROC – AUC  

3. Analyse approfondie : Nettoyage (Data Wrangling)  

4. Analyse approfondie : Exploration (EDA)  

5. Analyse approfondie : Méthodologie (Split)  

6. Focus théorique : Algorithme Random Forest  

7. Analyse approfondie : Évaluation (métriques et matrice de confusion)


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


## 3. Analyse Approfondie : Nettoyage (Data Wrangling)

### Le Problème Mathématique du "Vide"

Dans la base Heart Disease, certaines variables contiennent des valeurs manquantes (souvent encodées par `?`), notamment pour des colonnes comme `ca` ou `thal` dans certaines versions du dataset.  
Les algorithmes de machine learning basés sur l’algèbre linéaire (régression, SVM, réseaux, calculs de distances) ne peuvent pas traiter ces `NaN` et échouent dès qu’une seule valeur manquante apparaît dans la matrice de features.  

### La Mécanique de l’Imputation

Une stratégie simple consiste à remplacer les valeurs manquantes par une statistique calculée sur la colonne (moyenne, médiane ou mode), par exemple via `SimpleImputer`.  

1. **L’Apprentissage (`fit`) :**  
   L’imputer parcourt une colonne numérique comme `trestbps` (pression artérielle au repos) ou `chol` (cholestérol sérique) et calcule une statistique, par exemple la moyenne \(\mu\) de tous les patients observés.  
   Cette valeur (par exemple 132 mmHg pour `trestbps`) est mémorisée comme “valeur de remplacement” pour cette variable.  

2. **La Transformation (`transform`) :**  
   L’imputer repasse ensuite sur la colonne ; dès qu’il rencontre une valeur manquante, il la remplace par \(\mu\).  
   On obtient alors une matrice sans trous, compatible avec les algorithmes de classification comme la régression logistique ou Random Forest.  

### 💡 Le Coin de l’Expert (Data Leakage)

Dans un script pédagogique, on nettoie souvent les données **avant** de faire le split Train/Test, ce qui est plus simple mais théoriquement imparfait.  
En calculant la moyenne de `trestbps` ou `chol` sur tout le dataset, on utilise aussi les patients qui finiront dans le Test Set, ce qui introduit une **fuite d’information (Data Leakage)**.  

- **Pourquoi c’est un problème ?**  
  Les statistiques calculées sur tout le jeu de données incorporent des informations “du futur” (Test), ce qui peut rendre les performances du modèle trop optimistes.  

- **Bonne pratique en production :**  
  - Séparer d’abord en Train/Test.  
  - Ajuster l’imputer (`fit`) uniquement sur le Train.  
  - Appliquer ensuite cette imputation (`transform`) au Train **et** au Test avec les valeurs apprises sur le Train.  

---

## 4. Analyse Approfondie : Exploration (EDA)

C’est l’étape de profilage des patients et des variables cliniques du dataset Heart Disease.  

### Décrypter `.describe()`

L’appel `df[["age","trestbps","chol","thalach","oldpeak"]].describe()` fournit des statistiques descriptives sur des variables clés comme l’âge, la tension au repos, le cholestérol, la fréquence cardiaque maximale atteinte et la dépression ST.  
Comparer la **Mean** (moyenne) et le **50%** (médiane) permet de repérer les variables asymétriques : par exemple, un cholestérol moyen nettement plus élevé que la médiane indique une distribution tirée vers le haut par quelques hypercholestérolémies extrêmes.  

L’**écart-type (std)** renseigne sur la dispersion ; une variable avec `std` proche de 0 (quasi constante) apporte peu d’information discriminante au modèle et peut être candidate à la suppression ou à la mise de côté.  

### La Multicollinéarité (Redondance Clinique)

En examinant une matrice de corrélation, certaines variables apparaissent fortement corrélées, par exemple des combinaisons de paramètres de stress test comme `oldpeak` et `slope`, ou l’association entre certains marqueurs de risque (pression, cholestérol, fréquence cardiaque maximale).  
Cette redondance est peu gênante pour des modèles d’arbres (Random Forest, Gradient Boosting), mais peut rendre une régression logistique instable, car le modèle peine à attribuer clairement le “poids” de la décision à l’une ou l’autre variable fortement corrélée.  

---

## 5. Analyse Approfondie : Méthodologie (Split)

### Le Concept : Garantie de Généralisation

L’objectif n’est pas que le modèle mémorise la base Heart Disease, mais qu’il apprenne des règles générales valables pour de nouveaux patients jamais vus.  
Le découpage en Train/Test permet de simuler ce futur : le Train sert à l’apprentissage, le Test sert uniquement à mesurer la capacité de généralisation du modèle sur des données “neuves”.  

### Les Paramètres sous le capot

Avec `train_test_split(test_size=0.2, random_state=42, stratify=y)` :  

1. **Ratio 80/20 (principe de Pareto) :**  
   Environ 80 % des patients sont utilisés pour apprendre les relations entre les variables (âge, type de douleur thoracique `cp`, cholestérol `chol`, fréquence cardiaque `thalach`, etc.) et la présence de maladie cardiaque, tandis que 20 % sont réservés pour l’évaluation finale.  

2. **Reproductibilité (`random_state`) :**  
   Fixer `random_state=42` garantit que toute personne qui relance le notebook obtient exactement les mêmes patients en Train et Test, ce qui est indispensable pour comparer les résultats et valider un pipeline de manière scientifique.  

3. **Stratification (`stratify=y`) :**  
   La stratification conserve une proportion similaire de patients malades / non malades dans le Train et le Test, ce qui stabilise les métriques d’évaluation et évite des splits déséquilibrés par hasard.  

---

## 6. FOCUS THÉORIQUE : L’Algorithme Random Forest 🌲

Pourquoi Random Forest est-il un “couteau suisse” très apprécié pour la base Heart Disease ?  

### A. La Faiblesse de l’Arbre Unique

Un arbre de décision sur ce dataset peut construire des règles comme : “si `cp` = angine typique et `thalach` < 150 alors malade”, en enchaînant des seuils sur `age`, `chol`, `oldpeak`, etc.  
Seul, il a tendance à surapprendre des cas particuliers (par exemple un jeune patient très atypique), ce qui se traduit par une **variance élevée** et des performances instables sur de nouveaux patients.  

### B. La Force du Groupe (Bagging)

Random Forest construit de nombreux arbres en introduisant de l’aléa contrôlé.  

1. **Bootstrapping (échantillons patients) :**  
   Chaque arbre est entraîné sur un sous-ensemble tiré avec remise du jeu d’entraînement : certains patients sont vus plusieurs fois par un arbre, d’autres pas du tout pour cet arbre.  
   Chaque arbre développe ainsi sa propre “opinion clinique” basée sur une expérience légèrement différente.  

2. **Aléa sur les features (feature randomness) :**  
   À chaque split, l’arbre ne considère qu’un sous-ensemble aléatoire de variables (par exemple \(\sqrt{\text{nb\_features}}\)), ce qui l’oblige à tester aussi des colonnes moins évidentes (`restecg`, `exang`, `ca`, `thal`) au lieu de s’appuyer exclusivement sur les plus fortes (`cp`, `oldpeak`).  

### C. Le Consensus (Vote)

Lorsqu’un nouveau patient arrive :  

- Chaque arbre décide “malade” ou “non malade” en fonction de ses propres règles.  
- La forêt agrège ces avis par vote majoritaire (classification) ; les erreurs individuelles se compensent, tandis que le signal commun (les motifs cliniques robustes) domine la décision finale.  

---

## 7. Analyse Approfondie : Évaluation (L’Heure de Vérité)

### A. La Matrice de Confusion

Pour la classification “maladie cardiaque présente / absente”, la matrice de confusion se lit ainsi :  

- **Vrais Positifs (TP)** : Prédit “malade” | Réel “malade”. Patients cardiaques correctement identifiés.  
- **Vrais Négatifs (TN)** : Prédit “sain” | Réel “sain”. Pas de fausse alerte.  
- **Faux Positifs (FP – Erreur de Type I)** : Prédit “malade” | Réel “sain”.  
  - Impact : examens complémentaires inutiles, anxiété, surcoût, mais risque médical direct plus limité.  
- **Faux Négatifs (FN – Erreur de Type II)** : Prédit “sain” | Réel “malade”.  
  - Impact : risque majeur de retard diagnostique, d’infarctus ou de complications graves ; c’est le type d’erreur le plus critique à minimiser.  

### B. Les Métriques Avancées

Comme les classes peuvent être modérément déséquilibrées (répartition malades / non malades non parfaitement 50/50), l’**accuracy** seule peut être trompeuse.  

On surveille donc en priorité :  

1. **Précision (Precision)**  
   Elle mesure la proportion de patients prédits “malades” qui le sont réellement ; une précision faible signifie trop de fausses alertes pour les cardiologues.  

2. **Rappel (Recall / Sensibilité)**  
   Elle indique la capacité du modèle à détecter réellement les patients cardiaques ; un rappel faible signifie qu’on laisse passer trop de malades non détectés, ce qui est médicalement inacceptable.  

3. **F1-Score**  
   Le F1-score est la moyenne harmonique entre Précision et Recall ; c’est une note unique utile pour comparer plusieurs modèles lorsque l’équilibre entre faux positifs et faux négatifs est important.  

Dans le contexte de la maladie cardiaque, la priorité est généralement de **maximiser le Recall** (ne pas rater de patients malades), tout en gardant une précision raisonnable pour ne pas surcharger inutilement les examens et spécialistes.  
