import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

# ============================================================
# CONFIGURATION GÉNÉRALE
# ============================================================
st.set_page_config(
    page_title="CreditScore AI — Évaluation du risque de crédit",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

ART_DIR = "model_artifacts"

# ============================================================
# STYLE — CSS PERSONNALISÉ
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    :root {
        --navy: #10192e;
        --navy-light: #1a2540;
        --accent: #4C72B0;
        --accent2: #DD8452;
        --green: #2fb380;
        --red: #e05263;
        --gray: #8890a4;
        --card-bg: #151f38;
        --border: #253253;
    }

    .stApp {
        background: linear-gradient(180deg, #0b1120 0%, #0e1526 100%);
        color: #e8ecf5;
    }

    section[data-testid="stSidebar"] {
        background: #0b1120;
        border-right: 1px solid var(--border);
    }

    /* Hero header */
    .hero {
        padding: 28px 32px;
        border-radius: 18px;
        background: linear-gradient(135deg, #16214a 0%, #0d1530 100%);
        border: 1px solid var(--border);
        margin-bottom: 28px;
    }
    .hero h1 {
        font-size: 2rem;
        font-weight: 800;
        margin: 0 0 6px 0;
        color: #ffffff;
        letter-spacing: -0.5px;
    }
    .hero p {
        color: #9aa4bf;
        font-size: 1.02rem;
        margin: 0;
    }
    .badge {
        display: inline-block;
        background: rgba(76,114,176,0.18);
        color: #8fb2f0;
        border: 1px solid rgba(76,114,176,0.4);
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.4px;
        margin-bottom: 12px;
    }

    /* KPI cards */
    .kpi-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 18px 20px;
        text-align: left;
        transition: border-color 0.2s ease;
    }
    .kpi-card:hover { border-color: var(--accent); }
    .kpi-value { font-size: 1.65rem; font-weight: 800; color: #ffffff; margin: 4px 0 2px 0; }
    .kpi-label { font-size: 0.8rem; color: #8890a4; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;}
    .kpi-icon { font-size: 1.3rem; opacity: 0.85; }

    /* Section card */
    .section-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 24px 26px;
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-subtitle {
        font-size: 0.85rem;
        color: #8890a4;
        margin-bottom: 18px;
    }

    /* Result banner */
    .result-banner {
        border-radius: 16px;
        padding: 26px 28px;
        margin-top: 10px;
        border: 1px solid var(--border);
    }
    .result-banner.low { background: linear-gradient(135deg, rgba(47,179,128,0.16), rgba(47,179,128,0.04)); border-color: rgba(47,179,128,0.4);}
    .result-banner.medium { background: linear-gradient(135deg, rgba(221,132,82,0.16), rgba(221,132,82,0.04)); border-color: rgba(221,132,82,0.4);}
    .result-banner.high { background: linear-gradient(135deg, rgba(224,82,99,0.16), rgba(224,82,99,0.04)); border-color: rgba(224,82,99,0.4);}

    .result-title { font-size: 1.4rem; font-weight: 800; color: #fff; margin-bottom: 4px; }
    .result-sub { color: #b8c0d6; font-size: 0.92rem; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4C72B0, #3a5a8f);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1.4rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #5a80c0, #4C72B0);
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(76,114,176,0.35);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        background: var(--card-bg);
        border-radius: 10px 10px 0 0;
        border: 1px solid var(--border);
        border-bottom: none;
        color: #8890a4;
        font-weight: 600;
        padding: 10px 18px;
    }
    .stTabs [aria-selected="true"] {
        background: var(--accent) !important;
        color: white !important;
    }

    /* Inputs */
    .stSelectbox label, .stSlider label, .stNumberInput label, .stRadio label {
        color: #b8c0d6 !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
    }

    footer, #MainMenu { visibility: hidden; }

    hr { border-color: var(--border); }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CHARGEMENT DES ARTEFACTS (mis en cache)
# ============================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load(f"{ART_DIR}/model.pkl")
    scaler = joblib.load(f"{ART_DIR}/scaler.pkl")
    with open(f"{ART_DIR}/feature_columns.json") as f:
        feature_columns = json.load(f)
    with open(f"{ART_DIR}/meta.json") as f:
        meta = json.load(f)
    return model, scaler, feature_columns, meta

@st.cache_data
def load_test_predictions():
    return pd.read_csv(f"{ART_DIR}/test_predictions.csv")

@st.cache_data
def load_readable_dataset():
    return pd.read_csv(f"{ART_DIR}/dataset_readable.csv")

model, scaler, feature_columns, meta = load_artifacts()
test_df = load_test_predictions()
data_df = load_readable_dataset()

NUM_COLS_SCALING = meta["num_cols_scaling"]
ORDRE_NOTES = meta["ordre_notes"]
LOGEMENT_CATS = meta["statut_logement_categories"]
MOTIF_CATS = meta["motif_pret_categories"]
METRICS = meta["metrics"]

LABELS_MOTIF = {
    "DEBTCONSOLIDATION": "Consolidation de dettes",
    "EDUCATION": "Éducation",
    "HOMEIMPROVEMENT": "Amélioration du logement",
    "MEDICAL": "Frais médicaux",
    "PERSONAL": "Personnel",
    "VENTURE": "Projet entrepreneurial",
}
LABELS_LOGEMENT = {
    "RENT": "Locataire",
    "OWN": "Propriétaire",
    "MORTGAGE": "Hypothèque",
    "OTHER": "Autre",
}


# ============================================================
# FONCTION DE PRÉDICTION
# ============================================================
def preparer_features(age, revenu, statut_logement, anciennete_emploi, motif,
                       note, montant_pret, taux_interet, historique_defaut,
                       anciennete_credit):

    pret_pourcentage_revenu = montant_pret / revenu if revenu > 0 else 0
    revenu_log = np.log1p(revenu)
    montant_log = np.log1p(montant_pret)
    note_encodee = ORDRE_NOTES.index(note)
    defaut_encode = 1 if historique_defaut == "Oui" else 0

    row = {col: 0 for col in feature_columns}
    row["age_personne"] = age
    row["revenu_annuel"] = revenu
    row["anciennete_emploi"] = anciennete_emploi
    row["montant_pret"] = montant_pret
    row["taux_interet"] = taux_interet
    row["pret_pourcentage_revenu"] = pret_pourcentage_revenu
    row["anciennete_credit"] = anciennete_credit
    row["revenu_annuel_log"] = revenu_log
    row["montant_pret_log"] = montant_log
    row["note_pret_encodee"] = note_encodee
    row["historique_defaut_encode"] = defaut_encode

    logement_col = f"logement_{statut_logement}"
    if logement_col in row:
        row[logement_col] = 1
    motif_col = f"motif_{motif}"
    if motif_col in row:
        row[motif_col] = 1

    X = pd.DataFrame([row])[feature_columns]
    X[NUM_COLS_SCALING] = scaler.transform(X[NUM_COLS_SCALING])
    return X, pret_pourcentage_revenu


def predire(X):
    proba = model.predict_proba(X)[0, 1]
    pred = int(proba >= 0.5)
    return pred, proba


# ============================================================
# SIDEBAR — NAVIGATION
# ============================================================
PAGE_HOME = "🏠 Accueil"
PAGE_PREDICTION = "🎯 Prédiction"
PAGE_EXPLORATION = "📊 Exploration des données"
PAGE_PERFORMANCE = "📈 Performance du modèle"
PAGE_ABOUT = "ℹ️ À propos"

PAGES = [PAGE_HOME, PAGE_PREDICTION, PAGE_EXPLORATION, PAGE_PERFORMANCE, PAGE_ABOUT]

ALIASES = {
    "Accueil": PAGE_HOME,
    "Prédiction": PAGE_PREDICTION,
    "Exploration": PAGE_EXPLORATION,
    "Performance": PAGE_PERFORMANCE,
    "À propos": PAGE_ABOUT,
    "A propos": PAGE_ABOUT,
}

if "page_radio" not in st.session_state:
    st.session_state["page_radio"] = PAGES[0]

# Si une navigation programmatique a été demandée (clic sur un bouton),
# on applique le changement AVANT de créer le widget radio (obligatoire côté Streamlit).
if "nav_target" in st.session_state:
    st.session_state["page_radio"] = st.session_state.pop("nav_target")

def go_to(page_name):
    st.session_state["nav_target"] = ALIASES.get(page_name, page_name)

with st.sidebar:
    st.markdown("## 💳 Risque de crédit")
    st.caption("Système d'évaluation du risque de crédit")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        PAGES,
        label_visibility="collapsed",
        key="page_radio",
    )
    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:0.8rem; color:#8890a4; line-height:1.6;">
    <b>Modèle :</b> XGBoost optimisé<br>
    <b>AUC-ROC :</b> {METRICS['auc']}<br>
    <b>Données d'entraînement :</b> {METRICS['n_train']:,} emprunteurs
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE 0 — ACCUEIL
# ============================================================
if page == "🏠 Accueil":

    col_txt, col_img = st.columns([1.1, 1], gap="large")

    with col_txt:
        st.markdown("""
        <span class="badge">PROJET DE FIN D'ANNÉE · DATA SCIENCE & IA</span>
        <h1 style="font-size:2.6rem; font-weight:800; color:#ffffff; line-height:1.15; letter-spacing:-1px; margin-bottom:14px;">
            Évaluez le risque de crédit<br>en quelques secondes
        </h1>
        <p style="color:#9aa4bf; font-size:1.08rem; line-height:1.7; margin-bottom:24px;">
            La prédiction de risque de crédit combine exploration de données, prétraitement rigoureux et un modèle
            <b>XGBoost</b> optimisé pour estimer la probabilité de défaut de paiement d'un emprunteur,
            à partir de son profil socio-économique et des caractéristiques de son prêt.
        </p>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Tester une prédiction", use_container_width=True):
                go_to("Prédiction")
                st.rerun()
        with c2:
            if st.button("Voir la performance du modèle", use_container_width=True):
                go_to("Performance")
                st.rerun()

    with col_img:
        st.image("assets/hero.jpg", use_container_width=True,
                  caption="Analyse de données financières et prise de décision")

    st.markdown("<br>", unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    home_kpis = [
        ("👥", "Emprunteurs", f"{METRICS['n_train'] + METRICS['n_test']:,}"),
        ("📈", "AUC-ROC du modèle", f"{METRICS['auc']:.3f}"),
        ("🎯", "Précision", f"{METRICS['precision']*100:.1f}%"),
        ("📡", "Rappel", f"{METRICS['recall']*100:.1f}%"),
    ]
    for col, (icon, label, value) in zip([k1,k2,k3,k4], home_kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="font-size:1.2rem;">🧭 Explorer l\'application</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3, gap="medium")
    features = [
        (f1, "🎯", "Prédiction", "Renseignez un profil d'emprunteur et obtenez instantanément une estimation du risque de défaut.", "🎯 Prédiction"),
        (f2, "📊", "Exploration des données", "Visualisez les caractéristiques du portefeuille de 32 581 emprunteurs et les facteurs de risque.", "📊 Exploration des données"),
        (f3, "📈", "Performance du modèle", "Consultez les métriques détaillées, la courbe ROC et l'importance des variables du modèle.", "📈 Performance du modèle"),
    ]
    for col, icon, title, desc, target in features:
        with col:
            st.markdown(f"""
            <div class="section-card" style="min-height:200px;">
                <div style="font-size:1.8rem; margin-bottom:8px;">{icon}</div>
                <div class="section-title" style="font-size:1rem;">{title}</div>
                <div style="color:#8890a4; font-size:0.87rem; line-height:1.6;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Accéder →", key=f"nav_{target}", use_container_width=True):
                go_to(target)
                st.rerun()


# ============================================================
# PAGE 1 — PRÉDICTION
# ============================================================
elif page == "🎯 Prédiction":

    st.markdown("""
    <div class="hero">
        <span class="badge">MODÈLE XGBOOST · AUC 0.95</span>
        <h1>Évaluation du risque de défaut de crédit</h1>
        <p>Renseignez le profil de l'emprunteur pour estimer la probabilité de défaut de paiement.</p>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1.15, 1], gap="large")

    with col_form:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">👤 Profil de l\'emprunteur</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Informations personnelles et financières</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Âge", min_value=18, max_value=100, value=28, step=1)
            revenu = st.number_input("Revenu annuel ($)", min_value=1000, max_value=1000000, value=55000, step=1000)
            anciennete_emploi = st.number_input("Ancienneté d'emploi (années)", min_value=0.0, max_value=50.0, value=4.0, step=0.5)
        with c2:
            statut_logement_label = st.selectbox("Statut de logement", list(LABELS_LOGEMENT.values()))
            statut_logement = [k for k, v in LABELS_LOGEMENT.items() if v == statut_logement_label][0]
            anciennete_credit = st.number_input("Ancienneté du dossier de crédit (années)", min_value=0, max_value=40, value=5, step=1)
            historique_defaut = st.radio("Historique de défaut antérieur ?", ["Non", "Oui"], horizontal=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">🏦 Détails du prêt demandé</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Caractéristiques du crédit sollicité</div>', unsafe_allow_html=True)

        c3, c4 = st.columns(2)
        with c3:
            motif_label = st.selectbox("Motif du prêt", list(LABELS_MOTIF.values()))
            motif = [k for k, v in LABELS_MOTIF.items() if v == motif_label][0]
            montant_pret = st.number_input("Montant du prêt ($)", min_value=500, max_value=100000, value=10000, step=500)
        with c4:
            note = st.select_slider("Note de crédit du prêt", options=ORDRE_NOTES, value="B")
            taux_interet = st.slider("Taux d'intérêt (%)", min_value=5.0, max_value=25.0, value=11.0, step=0.1)

        st.markdown('</div>', unsafe_allow_html=True)

        predict_clicked = st.button("🔍 Évaluer le risque de crédit", use_container_width=True)

    with col_result:
        if predict_clicked:
            X, ratio_pret_revenu = preparer_features(
                age, revenu, statut_logement, anciennete_emploi, motif,
                note, montant_pret, taux_interet, historique_defaut, anciennete_credit
            )
            pred, proba = predire(X)

            if proba < 0.3:
                niveau, css_class, emoji, couleur = "Risque faible", "low", "✅", "#2fb380"
            elif proba < 0.6:
                niveau, css_class, emoji, couleur = "Risque modéré", "medium", "⚠️", "#DD8452"
            else:
                niveau, css_class, emoji, couleur = "Risque élevé", "high", "🚨", "#e05263"

            st.markdown(f"""
            <div class="result-banner {css_class}">
                <div style="font-size:2rem; margin-bottom:6px;">{emoji}</div>
                <div class="result-title">{niveau}</div>
                <div class="result-sub">Probabilité de défaut estimée : <b>{proba*100:.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True)

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba*100,
                number={'suffix': "%", 'font': {'size': 42, 'color': '#ffffff'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#8890a4'},
                    'bar': {'color': couleur},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(47,179,128,0.25)'},
                        {'range': [30, 60], 'color': 'rgba(221,132,82,0.25)'},
                        {'range': [60, 100], 'color': 'rgba(224,82,99,0.25)'}
                    ],
                }
            ))
            fig.update_layout(
                height=260, margin=dict(t=20, b=10, l=30, r=30),
                paper_bgcolor="rgba(0,0,0,0)", font={'color': "#e8ecf5"}
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📋 Résumé du profil</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <table style="width:100%; font-size:0.88rem; color:#b8c0d6; border-collapse:collapse;">
            <tr><td style="padding:6px 0;">Ratio prêt / revenu</td><td style="text-align:right; font-weight:600; color:#fff;">{ratio_pret_revenu:.2%}</td></tr>
            <tr><td style="padding:6px 0;">Note de crédit</td><td style="text-align:right; font-weight:600; color:#fff;">{note}</td></tr>
            <tr><td style="padding:6px 0;">Historique de défaut</td><td style="text-align:right; font-weight:600; color:#fff;">{historique_defaut}</td></tr>
            <tr><td style="padding:6px 0;">Prédiction du modèle</td><td style="text-align:right; font-weight:600; color:#fff;">{"Défaut probable" if pred==1 else "Remboursement probable"}</td></tr>
            </table>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="section-card" style="text-align:center; padding:60px 20px;">
                <div style="font-size:2.6rem; margin-bottom:10px;">🧮</div>
                <div style="color:#8890a4; font-size:0.95rem;">
                Remplissez le formulaire puis cliquez sur<br><b>« Évaluer le risque de crédit »</b><br>pour obtenir une estimation.
                </div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# PAGE 2 — EXPLORATION DES DONNÉES
# ============================================================
elif page == "📊 Exploration des données":

    st.markdown("""
    <div class="hero">
        <span class="badge">DATASET · 32 581 EMPRUNTEURS</span>
        <h1>Exploration du portefeuille de crédit</h1>
        <p>Vue d'ensemble des caractéristiques des emprunteurs et de leur risque de défaut.</p>
    </div>
    """, unsafe_allow_html=True)

    n = len(data_df)
    taux_defaut = data_df['statut_defaut'].mean()
    revenu_median = data_df['revenu_annuel'].median()
    montant_median = data_df['montant_pret'].median()

    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        ("👥", "Emprunteurs", f"{n:,}"),
        ("📉", "Taux de défaut", f"{taux_defaut*100:.1f}%"),
        ("💰", "Revenu médian", f"${revenu_median:,.0f}"),
        ("🏦", "Prêt médian", f"${montant_median:,.0f}"),
    ]
    for col, (icon, label, value) in zip([k1,k2,k3,k4], kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Vue d'ensemble", "Profil des emprunteurs", "Facteurs de risque"])

    plotly_template = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8ecf5"), colorway=["#4C72B0", "#DD8452", "#2fb380", "#e05263", "#8890a4"]
    )

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            counts = data_df['statut_defaut'].value_counts().sort_index()
            fig = px.bar(x=["Non défaut", "Défaut"], y=counts.values,
                         color=["Non défaut", "Défaut"],
                         color_discrete_map={"Non défaut": "#4C72B0", "Défaut": "#e05263"},
                         labels={'x': '', 'y': "Nombre d'emprunteurs"},
                         title="Distribution de la variable cible")
            fig.update_layout(**plotly_template, showlegend=False, height=380)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(data_df, x="revenu_annuel", nbins=40,
                               title="Distribution du revenu annuel",
                               labels={'revenu_annuel': 'Revenu annuel ($)'})
            fig.update_traces(marker_color="#4C72B0")
            fig.update_layout(**plotly_template, height=380)
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fig = px.histogram(data_df, x="age_personne", nbins=30,
                               title="Distribution de l'âge",
                               labels={'age_personne': 'Âge'})
            fig.update_traces(marker_color="#55A868")
            fig.update_layout(**plotly_template, height=350)
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            fig = px.histogram(data_df, x="montant_pret", nbins=30,
                               title="Distribution du montant emprunté",
                               labels={'montant_pret': 'Montant du prêt ($)'})
            fig.update_traces(marker_color="#DD8452")
            fig.update_layout(**plotly_template, height=350)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            vc = data_df['statut_logement'].value_counts()
            fig = px.pie(values=vc.values, names=[LABELS_LOGEMENT.get(x,x) for x in vc.index],
                        title="Répartition par statut de logement", hole=0.45)
            fig.update_layout(**plotly_template, height=380)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            vc = data_df['motif_pret'].value_counts()
            fig = px.pie(values=vc.values, names=[LABELS_MOTIF.get(x,x) for x in vc.index],
                        title="Répartition par motif de prêt", hole=0.45)
            fig.update_layout(**plotly_template, height=380)
            st.plotly_chart(fig, use_container_width=True)

        vc = data_df['note_pret'].value_counts().sort_index()
        fig = px.bar(x=vc.index, y=vc.values, title="Répartition par note de crédit",
                    labels={'x': 'Note de crédit', 'y': "Nombre d'emprunteurs"})
        fig.update_traces(marker_color="#4C72B0")
        fig.update_layout(**plotly_template, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.caption("Taux de défaut moyen selon les principales variables catégorielles")
        c1, c2 = st.columns(2)
        with c1:
            rate = data_df.groupby('statut_logement')['statut_defaut'].mean().sort_values(ascending=False)
            fig = px.bar(x=rate.values*100, y=[LABELS_LOGEMENT.get(x,x) for x in rate.index], orientation='h',
                        title="Taux de défaut par statut de logement",
                        labels={'x': 'Taux de défaut (%)', 'y': ''})
            fig.update_traces(marker_color="#e05263")
            fig.update_layout(**plotly_template, height=320)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            rate = data_df.groupby('note_pret')['statut_defaut'].mean().sort_index()
            fig = px.bar(x=rate.index, y=rate.values*100,
                        title="Taux de défaut par note de crédit",
                        labels={'x': 'Note de crédit', 'y': 'Taux de défaut (%)'})
            fig.update_traces(marker_color="#DD8452")
            fig.update_layout(**plotly_template, height=320)
            st.plotly_chart(fig, use_container_width=True)

        rate = data_df.groupby('motif_pret')['statut_defaut'].mean().sort_values(ascending=False)
        fig = px.bar(x=[LABELS_MOTIF.get(x,x) for x in rate.index], y=rate.values*100,
                    title="Taux de défaut par motif de prêt",
                    labels={'x': '', 'y': 'Taux de défaut (%)'})
        fig.update_traces(marker_color="#4C72B0")
        fig.update_layout(**plotly_template, height=350)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 3 — PERFORMANCE DU MODÈLE
# ============================================================
elif page == "📈 Performance du modèle":

    st.markdown("""
    <div class="hero">
        <span class="badge">XGBOOST OPTIMISÉ · VALIDÉ SUR 6 484 EMPRUNTEURS</span>
        <h1>Performance du modèle de scoring</h1>
        <p>Métriques d'évaluation calculées sur le jeu de test (20% des données, non vues à l'entraînement).</p>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        ("🎯", "Accuracy", f"{METRICS['accuracy']*100:.1f}%"),
        ("🔍", "Précision", f"{METRICS['precision']*100:.1f}%"),
        ("📡", "Rappel", f"{METRICS['recall']*100:.1f}%"),
        ("⚖️", "F1-score", f"{METRICS['f1']*100:.1f}%"),
        ("📈", "AUC-ROC", f"{METRICS['auc']:.3f}"),
    ]
    for col, (icon, label, value) in zip([k1,k2,k3,k4,k5], kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    plotly_template = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8ecf5")
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📉 Courbe ROC</div>', unsafe_allow_html=True)
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(test_df['y_true'], test_df['y_proba'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"XGBoost (AUC={METRICS['auc']})",
                                 line=dict(color="#4C72B0", width=3)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name="Modèle aléatoire",
                                 line=dict(color="#8890a4", width=1.5, dash='dash')))
        fig.update_layout(**plotly_template, height=380, xaxis_title="Taux de faux positifs",
                          yaxis_title="Taux de vrais positifs",
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧩 Matrice de confusion</div>', unsafe_allow_html=True)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(test_df['y_true'], test_df['y_pred'])
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["Non défaut", "Défaut"], y=["Non défaut", "Défaut"],
                        labels=dict(x="Prédiction", y="Réalité", color="Nombre"))
        fig.update_layout(**plotly_template, height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔑 Importance des variables</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Contribution de chaque variable aux prédictions du modèle XGBoost</div>', unsafe_allow_html=True)

    importances = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=True).tail(12)
    fig = px.bar(x=importances.values, y=importances.index, orientation='h',
                labels={'x': "Importance", 'y': ''})
    fig.update_traces(marker_color="#4C72B0")
    fig.update_layout(**plotly_template, height=430)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📖 Comment lire ces métriques ?</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#b8c0d6; font-size:0.9rem; line-height:1.8;">
    <b>Précision</b> : parmi les emprunteurs prédits en défaut, la proportion qui font effectivement défaut.<br>
    <b>Rappel</b> : parmi les emprunteurs réellement en défaut, la proportion correctement identifiée par le modèle.<br>
    <b>AUC-ROC</b> : capacité du modèle à distinguer les bons et mauvais payeurs, sur une échelle de 0.5 (aléatoire) à 1.0 (parfait).
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# PAGE 4 — À PROPOS
# ============================================================
else:
    st.markdown("""
    <div class="hero">
        <span class="badge">PROJET DE FIN D'ANNÉE</span>
        <h1>À propos de ce projet</h1>
        <p>Système de scoring crédit basé sur l'apprentissage automatique.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#b8c0d6; font-size:0.95rem; line-height:1.9;">
    Ce projet propose un système d'aide à la décision pour l'évaluation du risque de crédit, développé à partir
    du <b>Credit Risk Dataset</b> (32 581 emprunteurs). Il s'appuie sur une démarche complète de data science :

    <ul>
    <li><b>Exploration des données (EDA)</b> : analyse des distributions, détection des valeurs aberrantes et des corrélations</li>
    <li><b>Nettoyage et prétraitement</b> : traitement des valeurs manquantes, des outliers, encodage et normalisation des variables</li>
    <li><b>Modélisation</b> : comparaison de trois algorithmes (régression logistique, Random Forest, XGBoost)</li>
    <li><b>Optimisation</b> : réglage des hyperparamètres du modèle final par recherche aléatoire avec validation croisée</li>
    <li><b>Déploiement</b> : cette application Streamlit, qui permet de tester le modèle sur un profil personnalisé</li>
    </ul>

    <b>Modèle retenu :</b> XGBoost (AUC-ROC = 0.95), le plus performant parmi les modèles testés, conformément
    aux constats de la littérature scientifique sur le credit scoring (Lessmann et al. 2015 ; Baesens et al. 2003).
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚠️ Avertissement</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#8890a4; font-size:0.88rem; line-height:1.7;">
    Cette application est un projet académique à but pédagogique. Les prédictions générées ne doivent pas être
    utilisées pour de véritables décisions de crédit — elles reposent sur un modèle statistique entraîné sur un
    jeu de données historique et ne remplacent pas l'analyse d'un établissement financier.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
