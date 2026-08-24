import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import joblib
import json

RAW_PATH = "/home/claude/credit_risk_dataset.csv"
OUT_DIR = "/home/claude/streamlit_app/model_artifacts"

# ---------- 1. Chargement et renommage ----------
df = pd.read_csv(RAW_PATH)

rename = {
    'person_age': 'age_personne',
    'person_income': 'revenu_annuel',
    'person_home_ownership': 'statut_logement',
    'person_emp_length': 'anciennete_emploi',
    'loan_intent': 'motif_pret',
    'loan_grade': 'note_pret',
    'loan_amnt': 'montant_pret',
    'loan_int_rate': 'taux_interet',
    'loan_status': 'statut_defaut',
    'loan_percent_income': 'pret_pourcentage_revenu',
    'cb_person_default_on_file': 'historique_defaut',
    'cb_person_cred_hist_length': 'anciennete_credit'
}
df = df.rename(columns=rename)

# ---------- 2. Nettoyage ----------
df = df.drop_duplicates().reset_index(drop=True)

def cap_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    borne_inf, borne_sup = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=borne_inf, upper=borne_sup)
    return df

for col in ['revenu_annuel', 'anciennete_emploi', 'age_personne']:
    df = cap_outliers_iqr(df, col)

mediane_emploi = df['anciennete_emploi'].median()
df['anciennete_emploi'] = df['anciennete_emploi'].fillna(mediane_emploi)
df['taux_interet'] = df.groupby('note_pret')['taux_interet'].transform(lambda x: x.fillna(x.median()))

# Version "lisible" (catégories en clair) pour le dashboard d'exploration de l'app
df_readable_dashboard = df.copy()

# ---------- 3. Transformation log ----------
df['revenu_annuel_log'] = np.log1p(df['revenu_annuel'])
df['montant_pret_log'] = np.log1p(df['montant_pret'])

# ---------- 4. Encodage ----------
ordre_notes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
df['note_pret_encodee'] = df['note_pret'].apply(lambda x: ordre_notes.index(x))
df['historique_defaut_encode'] = df['historique_defaut'].map({'N': 0, 'Y': 1})

statut_logement_categories = sorted(df['statut_logement'].unique().tolist())
motif_pret_categories = sorted(df['motif_pret'].unique().tolist())

df = pd.get_dummies(df, columns=['statut_logement', 'motif_pret'],
                     prefix=['logement', 'motif'], drop_first=True)
colonnes_bool = df.select_dtypes(include='bool').columns
df[colonnes_bool] = df[colonnes_bool].astype(int)

# ---------- 5. Suppression colonnes redondantes ----------
df_final = df.drop(columns=['note_pret', 'historique_defaut'])

# ---------- 6. Scaling ----------
num_cols_scaling = ['anciennete_emploi', 'taux_interet', 'pret_pourcentage_revenu',
                     'anciennete_credit', 'revenu_annuel_log', 'montant_pret_log',
                     'note_pret_encodee']

scaler = StandardScaler()
df_final[num_cols_scaling] = scaler.fit_transform(df_final[num_cols_scaling])

# ---------- 7. Split ----------
X = df_final.drop(columns=['statut_defaut'])
y = df_final['statut_defaut']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 8. Modèle final (hyperparamètres issus du RandomizedSearchCV du notebook) ----------
ratio_classes = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    n_estimators=448,
    max_depth=3,
    learning_rate=0.2248,
    subsample=0.8836,
    colsample_bytree=0.8073,
    min_child_weight=9,
    gamma=0.4387,
    scale_pos_weight=ratio_classes,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": round(accuracy_score(y_test, y_pred), 4),
    "precision": round(precision_score(y_test, y_pred), 4),
    "recall": round(recall_score(y_test, y_pred), 4),
    "f1": round(f1_score(y_test, y_pred), 4),
    "auc": round(roc_auc_score(y_test, y_proba), 4),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "taux_defaut": round(float(y.mean()), 4),
}

print("Metrics:", metrics)

# ---------- 9. Sauvegarde des artefacts ----------
joblib.dump(model, f"{OUT_DIR}/model.pkl")
joblib.dump(scaler, f"{OUT_DIR}/scaler.pkl")

feature_columns = X.columns.tolist()
with open(f"{OUT_DIR}/feature_columns.json", "w") as f:
    json.dump(feature_columns, f)

meta = {
    "num_cols_scaling": num_cols_scaling,
    "ordre_notes": ordre_notes,
    "statut_logement_categories": statut_logement_categories,
    "motif_pret_categories": motif_pret_categories,
    "metrics": metrics
}
with open(f"{OUT_DIR}/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

# Save test set + predictions for the "Performance" page (avoid recompute in app)
X_test_out = X_test.copy()
X_test_out['y_true'] = y_test.values
X_test_out['y_proba'] = y_proba
X_test_out['y_pred'] = y_pred
X_test_out.to_csv(f"{OUT_DIR}/test_predictions.csv", index=False)

df_readable_dashboard.to_csv(f"{OUT_DIR}/dataset_readable.csv", index=False)

print("Artefacts sauvegardés dans", OUT_DIR)
