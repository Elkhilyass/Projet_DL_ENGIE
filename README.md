## Teaam
- DAHAOUI Ilyas & EL KHAZANE Ilyasse

# ENGIE Wind Turbine Power Prediction & NBM Anomaly Detection (WT1)

Ce dépôt contient un projet complet de modélisation de la production éolienne (puissance active) à partir de données SCADA (pas de 10 minutes), avec une approche **NBM (Normal Behavior Modeling)** :  
1) un modèle de prédiction du comportement normal (DNN),  
2) une détection d’anomalies basée sur l’analyse des **résidus** (erreur de prédiction) et leur persistance dans le temps,  
3) un étiquetage métier simple des événements détectés.  

Le projet est réalisé sur **une seule turbine (WT1)**, avec option d’utiliser les autres turbines pour calculer des **features de type "fleet median"** (effets communs) et des composantes **idiosyncratiques**.

---

## Objectifs

- **Prédire la puissance active** (TARGET) à partir de variables opérationnelles, **sans utiliser la vitesse du vent**.
- Construire un modèle **NBM** capable de :
  - apprendre le comportement normal en fonctionnement (**ON**),
  - détecter des **écarts anormaux persistants** via les résidus,
  - proposer une **interprétation** (hypothèses métier) des anomalies détectées.

---

## Données

- `engie_X.csv` : features SCADA (températures, vitesses de rotation, orientations, électrique, météo, etc.)
- `engie_Y.csv` : cible `TARGET` (puissance active)

Colonnes clés :
- `ID` : identifiant unique
- `MAC_CODE` : turbine (WT1, WT2, WT3, WT4)
- `Date_time` : index temporel (pas 10 min)

---

## Méthodologie (résumé)

### 1) Filtrage turbine + split temporel
- On travaille sur WT1.
- Split **chronologique** : Train / Val / Test (pas de shuffle).

### 2) Gating ON/OFF (séparation des régimes)
Le NBM doit modéliser le comportement **normal en production**, pas les périodes d’arrêt :
- OFF : turbine à l’arrêt → `TARGET ≈ 0`, signaux moins interprétables.
- ON  : turbine en production → relation physique SCADA → puissance exploitable.

 On utilise une règle simple et stable :
- `ON` si `TARGET >= 9 kW` (seuil opérationnel choisi)
- sinon `OFF`

Le modèle de régression (DNN et baselines) est entraîné **uniquement sur TRAIN_ON**.

### 3) Prétraitement
- Gestion des doublons et tri par `Date_time`
- Gestion des NaNs :
  - création d’un flag unique `Grid_voltage_isnan`
  - imputation (médiane fit sur TRAIN_ON ; + stratégie hybride selon gaps si utilisé)
- Feature engineering physique (exemples) :
  - `speed_ratio` (cohérence rotor ↔ générateur)
  - `bearing_temp_diff` (asymétrie thermique roulements)
  - `Rotor_speed_cv` (variabilité relative)
  - `power_control_proxy` (proxy commande puissance)

### 4) Modèle principal : DNN (NBM)
- DNN dense régularisé (BatchNorm / Dropout / L2)
- Optimisation MAE (métrique challenge)
- Sorties :
  - `y_hat` (puissance normale attendue)
  - `res = y - y_hat` (résidu signé)
  - `abs_res` et `abs_res_smooth` (lissé ~1h)

### 5) Détection d’anomalies via résidus
Deux niveaux :
- **Point anomalies** : dépassement d’un seuil `thr_point` (ex: quantile 0.995) sur `|res|`
- **State anomalies** : dépassement persistant de `thr_state` sur `abs_res_smooth` (ex: quantile 0.99)  
   On extrait des événements (start/end/durée/gravité).

### 6) Étiquetage métier (heuristiques)
Un événement est comparé à une fenêtre de référence avant l’événement (1h typiquement) pour proposer un label :
- `CURTAILMENT_OR_PITCH_LIMITATION`
- `LOW_ROTOR_SPEED_OR_SHUTDOWN`
- `ELECTRICAL_OR_MEASUREMENT_ISSUE`
- `OVERPERFORMANCE_OR_DISTRIBUTION_SHIFT`
- `UNDERPERFORMANCE_UNSPECIFIED`

 Ces labels sont des hypothèses explicables (pas des vérités sans logs de maintenance).

---

## Baselines ML (comparaison)
Pour benchmark :
- **XGBOOST**
- **Elastic Net**
- **HistGradientBoostingRegressor (HGBR)**

Résultat de comparaison :  
- Le modèle **le plus performant en MAE** est **XGBOOST** (MAE ≈ 17 sur TEST_ON).  
- Toutefois, l’objectif du projet étant de livrer une approche centrée **DNN + NBM**, le **modèle final retenu** pour la détection d’anomalies est le **DNN**.
