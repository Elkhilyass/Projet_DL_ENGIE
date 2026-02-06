# Modélisation du Comportement Normal et Détection d'Anomalies pour Turbines Éoliennes

## Équipe
**DAHAOUI Ilyas** & **EL KHAZANE Ilyasse**

---

## Vue d'ensemble

Ce projet propose une solution complète d'analyse prédictive et de maintenance intelligente pour les turbines éoliennes, basée sur l'exploitation de données SCADA (Supervisory Control and Data Acquisition) à haute fréquence. L'approche combine apprentissage profond et analyse statistique pour construire un système de **Normal Behavior Modeling (NBM)** capable de :

1. **Prédire avec précision** la puissance active générée par une turbine éolienne à partir de ses paramètres opérationnels, sans recours aux mesures de vitesse du vent
2. **Détecter automatiquement** les anomalies de fonctionnement par analyse des résidus de prédiction
3. **Classifier intelligemment** les événements anormaux selon des catégories métier facilitant le diagnostic

Le système développé permet d'anticiper les défaillances, d'optimiser les opérations de maintenance et de maximiser la disponibilité des turbines.

---

## Contexte et Motivation

Dans le secteur de l'énergie éolienne, la détection précoce d'anomalies représente un enjeu économique majeur. Les systèmes SCADA génèrent des volumes importants de données opérationnelles (températures, vitesses, tensions, angles) échantillonnées toutes les 10 minutes. L'exploitation intelligente de ces données permet de :

- Réduire les temps d'arrêt imprévus et les pertes de production
- Optimiser les stratégies de maintenance (passage du correctif au prédictif)
- Prolonger la durée de vie des équipements
- Détecter des problèmes de performance avant qu'ils ne deviennent critiques

Le défi technique consiste à distinguer le comportement normal de la turbine, qui varie naturellement avec les conditions météorologiques, des déviations réellement problématiques nécessitant une intervention.

---

## Objectifs du Projet

### Objectif Principal
Développer un modèle de comportement normal (NBM) capable de prédire la puissance active d'une turbine éolienne à partir de variables opérationnelles SCADA, **sans utiliser les mesures directes de vitesse du vent**.

### Objectifs Secondaires
- Concevoir un système de détection d'anomalies multi-niveaux distinguant les écarts ponctuels des dérives persistantes
- Proposer une classification métier des anomalies pour faciliter le diagnostic opérationnel
- Comparer différentes approches d'apprentissage automatique (Deep Learning vs méthodes ensemblistes)
- Fournir des outils d'analyse et de visualisation pour l'aide à la décision

---

## Données

### Sources
Le projet s'appuie sur deux fichiers de données issus d'un parc éolien de quatre turbines :

- **`engie_X.csv`** : Variables explicatives SCADA (features)
- **`engie_Y.csv`** : Variable cible (puissance active générée)

### Résolution Temporelle
- Pas d'échantillonnage : **10 minutes**
- Période couverte : Plusieurs mois d'exploitation continue
- Turbine étudiée : **WT1** (Wind Turbine 1)

### Variables Clés

#### Identification
- `ID` : Identifiant unique de chaque enregistrement
- `MAC_CODE` : Code de la turbine (WT1, WT2, WT3, WT4)
- `Date_time` : Horodatage au format timestamp

#### Variables Opérationnelles (exemples)

**Cinématique**
- `Rotor_speed` : Vitesse de rotation du rotor (tours/min)
- `Generator_speed` : Vitesse de rotation du générateur (tours/min)

**Thermique**
- `Bearing_temp_front` : Température roulement avant (°C)
- `Bearing_temp_rear` : Température roulement arrière (°C)
- `Nacelle_temp` : Température nacelle (°C)
- `Gearbox_temp` : Température multiplicateur (°C)
- `Generator_temp` : Température générateur (°C)

**Électrique**
- `Grid_voltage` : Tension réseau (V)
- `Grid_current` : Courant réseau (A)

**Orientation et Contrôle**
- `Yaw_angle` : Angle de lacet (°)
- `Pitch_angle` : Angle de pitch des pales (°)

**Météorologique**
- `Ambient_temp` : Température ambiante (°C)
- `Atmospheric_pressure` : Pression atmosphérique (hPa)

**Variable Cible**
- `TARGET` : Puissance active générée (kW)

---

## Méthodologie

### 1. Partitionnement Temporel Strict

Le respect de l'ordre chronologique est essentiel pour simuler des conditions réalistes de déploiement :

- **TRAIN** : Ensemble d'entraînement (données les plus anciennes)
- **VAL** : Ensemble de validation (données intermédiaires)
- **TEST** : Ensemble de test (données les plus récentes, jamais vues pendant l'optimisation)

**Principe** : Aucun mélange aléatoire (shuffle) n'est appliqué pour préserver la structure temporelle.

### 2. Séparation des Régimes Opérationnels (Gating ON/OFF)

Une turbine éolienne alterne entre deux régimes de fonctionnement principaux :

- **Régime OFF** : Turbine à l'arrêt ou en veille
  - Critère : `Rotor_speed < 9 rpm`
  - Caractéristiques : Production nulle ou très faible, signaux SCADA peu informatifs

- **Régime ON** : Turbine en production active
  - Critère : `Rotor_speed >= 9 rpm`
  - Caractéristiques : Relations physiques exploitables entre variables et puissance

**Stratégie** : Le modèle NBM est entraîné **exclusivement sur les données TRAIN_ON** pour apprendre le comportement normal en production, évitant ainsi de modéliser les patterns d'arrêt non pertinents.

### 3. Prétraitement des Données

#### Nettoyage
- Suppression des doublons potentiels
- Tri chronologique strict par `Date_time`
- Vérification de la cohérence temporelle

#### Gestion des Valeurs Manquantes
- Création d'un indicateur binaire `Grid_voltage_isnan` pour capturer l'information de présence/absence
- Imputation par la médiane calculée sur l'ensemble TRAIN_ON (robuste aux outliers)
- Stratégie hybride (interpolation + imputation) pour les longues séquences manquantes si nécessaire

#### Ingénierie de Variables (Feature Engineering)

Plusieurs variables dérivées ont été créées pour enrichir la représentation physique :

- **`speed_ratio`** = Rotor_speed / Generator_speed
  - Capture la cohérence du couplage mécanique via le multiplicateur
  
- **`bearing_temp_diff`** = |Bearing_temp_front - Bearing_temp_rear|
  - Détecte les asymétries thermiques anormales des roulements
  
- **`Rotor_speed_cv`** = std(Rotor_speed) / mean(Rotor_speed)
  - Mesure la variabilité relative et la stabilité du régime
  
- **`power_control_proxy`**
  - Combinaison de signaux pour estimer l'intention de contrôle de puissance

Ces features permettent au modèle d'exploiter des relations physiques complexes non directement mesurées.

### 4. Architecture du Modèle Principal : DNN

Le réseau de neurones profond retenu présente l'architecture suivante :

**Structure**
- **Couches d'entrée** : Nombre de neurones = nombre de features SCADA
- **Couches cachées** : Architecture dense progressive (ex: 256 → 128 → 64 → 32)
- **Couche de sortie** : 1 neurone avec activation linéaire (régression)

**Techniques de Régularisation**
- **Batch Normalization** : Stabilise l'apprentissage et accélère la convergence
- **Dropout** : Taux variable par couche (0.2 à 0.3) pour prévenir le surapprentissage
- **Régularisation L2** : Pénalité sur les poids (λ ≈ 10⁻⁴)

**Fonction d'Activation**
- ReLU pour les couches cachées (non-linéarité et gradient stable)

**Optimisation**
- **Fonction de perte** : Mean Absolute Error (MAE)
  - Choix justifié par sa robustesse aux outliers
  - Interprétabilité directe en kW
  
- **Optimiseur** : Adam avec taux d'apprentissage adaptatif
  - Learning rate initial : 10⁻³
  - Décroissance programmée (ReduceLROnPlateau)
  
- **Stratégies d'entraînement**
  - Batch size : 128
  - Early stopping : Patience de 20 époques sur validation MAE
  - Nombre maximum d'époques : 200

### 5. Détection d'Anomalies par Analyse des Résidus

#### Calcul des Résidus

Pour chaque observation, le résidu quantifie l'écart entre prédiction et réalité :

```
résidu = TARGET_réel - TARGET_prédit
```

- **Résidu positif** : Surproduction par rapport au modèle
- **Résidu négatif** : Sous-production par rapport au modèle

#### Stratégie Multi-Niveaux

**Niveau 1 : Anomalies Ponctuelles**
- Seuil basé sur le quantile 99.5% des résidus absolus calculés sur TRAIN_ON
- Détecte les écarts instantanés importants (pics isolés)

**Niveau 2 : Anomalies d'État Persistantes**
- Application d'un lissage temporel (moyenne mobile ~1h) sur les résidus absolus
- Seuil basé sur le quantile 99% des résidus lissés
- Capture les dérives prolongées du comportement

#### Extraction d'Événements

Les anomalies d'état consécutives sont regroupées en événements caractérisés par :
- Instant de début et de fin
- Durée totale
- Gravité moyenne (amplitude moyenne du résidu)
- Nombre de points temporels affectés

### 6. Classification Métier des Anomalies

Chaque événement détecté est analysé par comparaison avec une fenêtre de référence (typiquement 1 heure avant le début de l'événement) pour proposer une hypothèse de diagnostic.

#### Catégories Définies

**CURTAILMENT_OR_PITCH_LIMITATION**
- Limitation intentionnelle de puissance
- Signature : Baisse de production + modification des angles de pitch
- Contexte : Contraintes réseau, protection équipement

**LOW_ROTOR_SPEED_OR_SHUTDOWN**
- Ralentissement ou arrêt du rotor
- Signature : Chute significative de la vitesse de rotation
- Contexte : Vents faibles, arrêt programmé, défaillance mécanique

**ELECTRICAL_OR_MEASUREMENT_ISSUE**
- Problème électrique ou capteur
- Signature : Anomalies de tension/courant réseau, incohérences mesures
- Contexte : Défaut réseau, panne capteur, problème de connexion

**OVERPERFORMANCE_OR_DISTRIBUTION_SHIFT**
- Surproduction par rapport au modèle
- Signature : Résidu moyen positif significatif
- Contexte : Changement de calibration, vent plus fort que prévu, dérive du modèle

**UNDERPERFORMANCE_UNSPECIFIED**
- Sous-performance sans cause identifiée
- Signature : Résidu négatif sans autre indicateur clair
- Contexte : Nécessite investigation approfondie

**Important** : Ces classifications sont des hypothèses explicables basées sur les données disponibles. Une validation avec les logs de maintenance réels est nécessaire pour confirmation.

---

## Modèles de Référence (Baselines)

Pour contextualiser les performances du DNN, trois algorithmes de Machine Learning classiques ont été implémentés :

### XGBoost (Extreme Gradient Boosting)
- Algorithme de boosting par gradient avec arbres de décision
- Excellente performance sur données tabulaires
- Offre une interprétabilité via l'importance des features

### Elastic Net
- Régression linéaire régularisée (combinaison L1 + L2)
- Test de l'hypothèse de linéarité des relations
- Baseline simple et rapide

### Histogram Gradient Boosting Regressor
- Variante optimisée du gradient boosting
- Efficace sur grands volumes de données
- Utilise des histogrammes pour la discrétisation

---

## Résultats

### Performances Prédictives (TEST_ON)

| Modèle | MAE (kW) | RMSE (kW) | R² |
|--------|----------|-----------|-----|
| **XGBoost** | **17.0** | 25.3 | 0.982 |
| **DNN** | **18.5** | 27.1 | 0.978 |
| Hist. Gradient Boosting | 19.2 | 28.4 | 0.975 |
| Elastic Net | 32.6 | 45.8 | 0.912 |

### Analyse des Résultats

- **XGBoost** obtient la meilleure performance brute (MAE ≈ 17 kW)
- **DNN** reste très compétitif avec une MAE de 18.5 kW et un R² de 0.978
- L'écart DNN/XGBoost (1.5 kW) est négligeable par rapport à la puissance nominale
- **Elastic Net** confirme la forte non-linéarité des relations (performances 2x inférieures)

### Choix du Modèle Final : DNN

Malgré des performances légèrement inférieures à XGBoost, le DNN a été retenu comme modèle principal pour :

1. **Adéquation à l'objectif NBM** : Meilleure intégration dans une approche de détection d'anomalies
2. **Flexibilité architecturale** : Possibilité d'extensions (LSTM, attention, multi-task learning)
3. **Contrôle de la régularisation** : Ajustement fin du compromis biais-variance pour la détection
4. **Capacité d'évolution** : Intégration future de données temporelles, séquentielles ou multimodales

---

## Structure du Projet

```
Projet_DL_ENGIE/
│
├── README.md                          # Ce fichier
├── Notebook.ipynb                     # Notebook principal d'analyse
├── requirements.txt                   # Dépendances Python
│
├── engie_Description.txt              # Description détaillée du challenge
│
├── wind_turbine_pipeline.keras        # Modèle DNN entraîné (format Keras)
└── wind_turbine_pipeline.pkl          # Pipeline complet (préprocessing + modèle)
```

---

## Installation et Utilisation

### Prérequis
```bash
Python >= 3.8
```

### Installation des Dépendances
```bash
pip install -r requirements.txt
```

### Packages Principaux
- `tensorflow` / `keras` : Construction et entraînement du DNN
- `scikit-learn` : Prétraitement, baselines ML, métriques
- `xgboost` : Modèle de référence
- `pandas` : Manipulation des données
- `numpy` : Calculs numériques
- `matplotlib` / `seaborn` : Visualisations

### Exécution
Ouvrez et exécutez le notebook principal :
```bash
jupyter notebook Notebook.ipynb
```

Le notebook contient :
1. Chargement et exploration des données
2. Prétraitement et feature engineering
3. Entraînement des modèles (DNN + baselines)
4. Évaluation et comparaison des performances
5. Détection d'anomalies et classification
6. Visualisations et analyses

---

## Perspectives d'Amélioration

### Court Terme
- Validation croisée des détections avec logs de maintenance réels
- Optimisation des seuils de détection par validation métier
- Enrichissement des règles de classification avec expertise terrain

### Moyen Terme
- Extension à l'ensemble des turbines du parc (WT1 à WT4)
- Intégration de features de flotte (médiane, écarts, corrélations inter-turbines)
- Architectures récurrentes (LSTM, GRU) pour exploiter les dépendances temporelles longues

### Long Terme
- Modèles multi-tâches (prédiction + détection + diagnostic simultanés)
- Apprentissage par transfert pour généralisation à d'autres parcs
- Intégration de données météorologiques externes (prévisions, réanalyses)
- Déploiement en production avec monitoring temps réel et alertes automatiques
- Techniques d'explicabilité (SHAP, LIME) pour interpréter les prédictions

---

## Contributions et Contact

Ce projet a été développé par :
- **DAHAOUI Ilyas**
- **EL KHAZANE Ilyasse**

Pour toute question, suggestion ou collaboration, n'hésitez pas à ouvrir une issue sur ce dépôt GitHub.

---

## Licence

Ce projet est fourni à des fins éducatives et de recherche. Les données ENGIE restent la propriété de leur détenteur.

---

## Références

- Schlechtingen, M., Santos, I. F. (2011). Comparative analysis of neural network and regression based condition monitoring approaches for wind turbine fault detection.
- Stetco, A. et al. (2019). Machine learning methods for wind turbine condition monitoring: A review.
- Tautz-Weinert, J., Watson, S. J. (2017). Using SCADA data for wind turbine condition monitoring.
