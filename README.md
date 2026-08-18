# CreditScore AI — Application Streamlit

Application web d'évaluation du risque de crédit, développée avec **Streamlit**, basée sur un modèle
**XGBoost** entraîné sur le *Credit Risk Dataset* (32 581 emprunteurs).

## 📁 Structure du projet

```
streamlit_app/
├── app.py                      # Application Streamlit principale
├── train_model.py              # Script de reproduction du pipeline (nettoyage + entraînement)
├── requirements.txt            # Dépendances Python
└── model_artifacts/            # Modèle et données pré-calculées
    ├── model.pkl                # Modèle XGBoost entraîné
    ├── scaler.pkl                # StandardScaler ajusté
    ├── feature_columns.json      # Liste ordonnée des variables du modèle
    ├── meta.json                  # Métadonnées (catégories, métriques)
    ├── test_predictions.csv       # Prédictions sur le jeu de test (page Performance)
    └── dataset_readable.csv       # Dataset nettoyé, lisible (page Exploration)
```

## 🚀 Installation et lancement

1. Créer un environnement virtuel (recommandé) :
```bash
python -m venv venv
source venv/bin/activate   # Windows : venv\Scripts\activate
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

3. Lancer l'application :
```bash
streamlit run app.py
```

L'application s'ouvre automatiquement dans votre navigateur à l'adresse `http://localhost:8501`.

## 🔄 Ré-entraîner le modèle

Si vous voulez régénérer les artefacts (`model.pkl`, `scaler.pkl`, etc.) à partir du dataset brut :

1. Placez `credit_risk_dataset.csv` dans le dossier du projet
2. Modifiez la variable `RAW_PATH` dans `train_model.py` si besoin
3. Exécutez :
```bash
python train_model.py
```

Cela régénère tous les fichiers dans `model_artifacts/`.

## 📄 Pages de l'application

- **🎯 Prédiction** : formulaire interactif pour estimer le risque de défaut d'un emprunteur (jauge de probabilité, niveau de risque)
- **📊 Exploration des données** : tableau de bord avec KPIs et graphiques sur le portefeuille de crédit
- **📈 Performance du modèle** : métriques, courbe ROC, matrice de confusion, importance des variables
- **ℹ️ À propos** : présentation du projet et de la méthodologie

## ⚠️ Avertissement

Ce projet est à but pédagogique. Les prédictions ne doivent pas être utilisées pour de véritables
décisions de crédit.
"# Prediction_risque_credit"  
