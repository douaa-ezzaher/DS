# DS et ML TD  
Réalisé par : EZZAHER DOUAA – S7 CAC G2  

## Introduction de la base de données

Le jeu de données utilisé porte sur la santé mentale d’étudiants universitaires. Chaque ligne correspond à un étudiant et contient des informations simples sur son profil (âge, genre, études, CGPA, etc.) ainsi que des réponses indiquant s’il souffre de dépression, d’anxiété ou de panic attack. L’objectif est d’utiliser ces informations pour mieux comprendre les profils d’étudiants susceptibles de présenter des troubles psychologiques.

## Problématique et type de modèle

La question posée est : peut‑on prédire si un étudiant présente au moins un trouble de santé mentale à partir de ses caractéristiques personnelles et académiques ? 
Ce problème est traité comme une tâche de classification binaire supervisée: La variable cible vaut 1 si l’étudiant déclare au moins un trouble (dépression, anxiété ou panic attack) et 0 s’il ne déclare aucun de ces troubles.

## Dictionnaire 
Le dataset contient un nombre limité d’étudiants, chacun décrit par plusieurs variables qualitatives et quantitatives. On y trouve par exemple : genre (catégoriel), âge (numérique), filière et année d’étude (catégoriel), CGPA (numérique) et statut marital (catégoriel), ainsi que des colonnes booléennes indiquant la présence ou non de troubles mentaux. 
La target utilisée dans ce projet est une nouvelle variable binaire construite à partir de ces colonnes de troubles.

## Variables explicatives choisies

Les variables explicatives retenues sont :  
- Gender (genre)  
- Age (âge)  
- Course (filière / cursus)  
- Year of Study (année d’étude)  
- CGPA (moyenne générale)  
- Marital Status (statut marital)  
Ces variables servent d’entrées au modèle de classification pour prédire la présence d’au moins un trouble mental.
