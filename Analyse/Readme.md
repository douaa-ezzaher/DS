# Rapport d’analyse – Business_Sales(Dataset)2025

## Sommaire

- Introduction
- Description de la base de données
- Méthodologie générale
- Régression linéaire : prédiction du volume de ventes
- Régression logistique : probabilité de fortes ventes
- Principaux enseignements

---

## Introduction

Ce rapport présente une analyse de la base Business_Sales(Dataset)2025, avec un focus sur deux modèles prédictifs : une régression linéaire pour estimer le volume de ventes et une régression logistique pour modéliser la probabilité de fortes ventes à partir de certaines variables explicatives.

---

## Description de la base de données

La base Business_Sales(Dataset)2025 décrit l’activité de ventes d’une entreprise sur divers produits et périodes.  
Chaque enregistrement correspond à la performance d’un produit (ou d’un magasin) et contient notamment :

- Des informations produit : identifiant, catégorie, marque, etc.  
- Des variables de contexte : saison ou période, indicateurs de promotion, positionnement produit.  
- Des mesures de performance : volume de ventes, parfois chiffre d’affaires ou indicateurs de tendance.  

Grâce à cette structure multi‑variables, la base est adaptée à des analyses descriptives (tendances, performances) et à des modèles prédictifs comme la prévision de ventes ou l’étude de l’impact des promotions et du prix.  

---

## Méthodologie générale

L’analyse repose sur deux types de modèles :

- Régression linéaire pour prédire une variable continue : le volume de ventes.  
- Régression logistique pour prédire la probabilité d’appartenir à une catégorie binaire : “fortes ventes” vs “faibles ventes”.  

Les résultats sont visualisés à l’aide de graphes qui comparent valeurs réelles et valeurs prédites, ce qui permet d’évaluer la qualité d’ajustement des modèles et d’identifier des biais éventuels.  

---

## Régression linéaire : prédiction du volume de ventes

La régression linéaire est utilisée pour modéliser le volume de ventes à partir de plusieurs variables explicatives (par exemple le prix, les promotions, la saison, la catégorie, etc.).  
Dans le graphique “Linear Regression: Actual vs. Predicted Sales Volume”, chaque point représente une observation : l’axe horizontal correspond au volume de ventes réel et l’axe vertical au volume prédit par le modèle, tandis que la ligne rouge en pointillés représente la diagonale idéale où prédiction = valeur réelle.

Le nuage de points montre une tendance globale croissante : lorsque les ventes réelles augmentent, les ventes prédites augmentent également, ce qui indique que le modèle capte une partie de la relation entre les variables explicatives et le volume vendu.  
Cependant, les points sont souvent éloignés de la diagonale et forment visiblement deux “bandes” ou groupes, ce qui suggère des erreurs systématiques pour certains segments (types de produits ou conditions de vente) et la présence possible d’effets non linéaires ou de variables manquantes que le modèle linéaire simple ne parvient pas à représenter.  

---

## Régression logistique : probabilité de fortes ventes

La régression logistique sert à prédire la probabilité qu’une observation appartienne à la catégorie “fortes ventes” (par exemple Sales_Category = 1 si le volume dépasse un certain seuil, 0 sinon), en fonction d’une ou plusieurs variables explicatives, ici principalement le prix.  
Le modèle ne fournit pas un volume de ventes, mais une probabilité comprise entre 0 et 1, ensuite convertie en 0 ou 1 selon un seuil de décision (souvent 0,5) pour classer les observations en “High Sales” ou “Low Sales”.

Dans le graphique “Logistic Regression: Predicted Probability of High Sales vs. Price”, les points bleus et verts représentent les observations réelles (0 = faibles ventes, 1 = fortes ventes) le long de l’axe des prix, tandis que la courbe rouge montre la probabilité prédite de fortes ventes en fonction du prix.  
La courbe est décroissante : plus le prix augmente, plus la probabilité de fortes ventes diminue, ce qui traduit une sensibilité négative de la demande au prix ; cette visualisation permet ainsi d’identifier les niveaux de prix où la probabilité d’obtenir de “High Sales” devient très faible et d’orienter les décisions de tarification.  

---

## Principaux enseignements

- Le modèle de régression linéaire reproduit la tendance générale entre les facteurs explicatifs et le volume de ventes, mais les écarts à la diagonale parfaite et la présence de groupes distincts de points indiquent des limites et suggèrent de tester des modèles plus complexes ou des transformations de variables.  
- Le modèle de régression logistique met en évidence une relation négative entre le prix et la probabilité de fortes ventes, confirmant qu’une hausse du prix réduit fortement la probabilité d’appartenir à la classe “High Sales” au-delà de certains seuils.  
- Ces deux approches offrent une base pour des décisions business : ajustement du mix prix‑promotion, ciblage des produits à fort potentiel et identification des zones où les modèles doivent être améliorés avant un usage opérationnel.  
