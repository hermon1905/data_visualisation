import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Explorateur de Données CSV",
    page_icon="📊",
    layout="wide"
)

def detect_outliers_iqr(df, column):
    """Détecte les valeurs aberrantes avec la méthode IQR"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def detect_outliers_zscore(df, column, threshold=3):
    """Détecte les valeurs aberrantes avec le Z-score"""
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    outliers = df[z_scores > threshold]
    return outliers

def analyze_data(df):
    """Analyse exploratoire des données"""
    analysis = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        'outliers': {}
    }
    
    # Détection des valeurs aberrantes pour les colonnes numériques
    for col in analysis['numeric_columns']:
        outliers_iqr, _, _ = detect_outliers_iqr(df, col)
        analysis['outliers'][col] = len(outliers_iqr)
    
    return analysis

def clean_missing_values(df, strategy='auto'):
    """Nettoie les valeurs manquantes"""
    df_cleaned = df.copy()
    
    for column in df_cleaned.columns:
        missing_count = df_cleaned[column].isnull().sum()
        if missing_count > 0:
            if df_cleaned[column].dtype in ['int64', 'float64']:
                if strategy == 'mean':
                    df_cleaned[column].fillna(df_cleaned[column].mean(), inplace=True)
                elif strategy == 'median':
                    df_cleaned[column].fillna(df_cleaned[column].median(), inplace=True)
                else:  # auto
                    # Utiliser la médiane si la distribution est asymétrique
                    skewness = abs(df_cleaned[column].skew())
                    if skewness > 1:
                        df_cleaned[column].fillna(df_cleaned[column].median(), inplace=True)
                    else:
                        df_cleaned[column].fillna(df_cleaned[column].mean(), inplace=True)
            else:
                # Pour les variables catégorielles, utiliser le mode
                mode_value = df_cleaned[column].mode()
                if len(mode_value) > 0:
                    df_cleaned[column].fillna(mode_value[0], inplace=True)
    
    return df_cleaned

def remove_outliers(df, method='iqr', threshold=3):
    """Supprime les valeurs aberrantes"""
    df_cleaned = df.copy()
    outliers_removed = 0
    
    numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
    
    for column in numeric_columns:
        if method == 'iqr':
            Q1 = df_cleaned[column].quantile(0.25)
            Q3 = df_cleaned[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_mask = (df_cleaned[column] < lower_bound) | (df_cleaned[column] > upper_bound)
            outliers_removed += outliers_mask.sum()
            df_cleaned = df_cleaned[~outliers_mask]
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df_cleaned[column].dropna()))
            outliers_mask = z_scores > threshold
            outliers_removed += outliers_mask.sum()
            df_cleaned = df_cleaned[~outliers_mask]
    
    return df_cleaned, outliers_removed

# Interface principale
st.title("📊 Explorateur et Nettoyeur de Données CSV")
st.markdown("---")

# Upload du fichier
st.header("1. 📁 Upload de votre fichier CSV")
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type=['csv'])

if uploaded_file is not None:
    # Lecture du fichier
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Fichier chargé avec succès ! ({df.shape[0]} lignes, {df.shape[1]} colonnes)")
        
        # Aperçu des données
        st.subheader("Aperçu des données")
        st.dataframe(df.head())
        
        # Analyse exploratoire
        st.header("2. 🔍 Analyse Exploratoire")
        analysis = analyze_data(df)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Nombre de lignes", analysis['shape'][0])
        with col2:
            st.metric("Nombre de colonnes", analysis['shape'][1])
        with col3:
            st.metric("Valeurs dupliquées", analysis['duplicates'])
        with col4:
            total_missing = sum(analysis['missing_values'].values())
            st.metric("Valeurs manquantes", total_missing)
        
        # Détails des valeurs manquantes
        if total_missing > 0:
            st.subheader("📋 Détail des valeurs manquantes par colonne")
            missing_df = pd.DataFrame([
                {"Colonne": col, "Valeurs manquantes": count, "Pourcentage": f"{(count/len(df)*100):.2f}%"}
                for col, count in analysis['missing_values'].items() if count > 0
            ])
            st.dataframe(missing_df)
        
        # Valeurs aberrantes
        if analysis['outliers']:
            st.subheader("🎯 Valeurs aberrantes détectées (méthode IQR)")
            outliers_df = pd.DataFrame([
                {"Colonne": col, "Nombre de valeurs aberrantes": count}
                for col, count in analysis['outliers'].items() if count > 0
            ])
            if not outliers_df.empty:
                st.dataframe(outliers_df)
                
                # Visualisation des outliers
                if st.checkbox("Afficher les graphiques des valeurs aberrantes"):
                    for col in analysis['numeric_columns']:
                        if analysis['outliers'][col] > 0:
                            fig = px.box(df, y=col, title=f"Boîte à moustaches - {col}")
                            st.plotly_chart(fig)
        
        # Section de nettoyage
        st.header("3. 🧹 Nettoyage des Données")
        
        st.markdown("""
        **Explication du processus de nettoyage :**
        
        1. **Gestion des valeurs manquantes** :
           - Pour les variables numériques : utilisation de la moyenne (distribution normale) ou médiane (distribution asymétrique)
           - Pour les variables catégorielles : utilisation du mode (valeur la plus fréquente)
        
        2. **Suppression des doublons** :
           - Identification et suppression des lignes identiques
        
        3. **Traitement des valeurs aberrantes** :
           - Méthode IQR : suppression des valeurs en dehors de [Q1-1.5*IQR, Q3+1.5*IQR]
           - Méthode Z-score : suppression des valeurs avec |z-score| > seuil
        """)
        
        # Options de nettoyage
        st.subheader("Paramètres de nettoyage")
        
        col1, col2 = st.columns(2)
        
        with col1:
            handle_missing = st.selectbox(
                "Traitement des valeurs manquantes",
                ["auto", "mean", "median", "skip"],
                help="Auto : choix automatique selon la distribution"
            )
            
            remove_duplicates = st.checkbox("Supprimer les doublons", value=True)
        
        with col2:
            handle_outliers = st.selectbox(
                "Traitement des valeurs aberrantes",
                ["skip", "iqr", "zscore"]
            )
            
            if handle_outliers == "zscore":
                zscore_threshold = st.slider("Seuil Z-score", 2.0, 4.0, 3.0, 0.1)
        
        # Bouton de nettoyage
        if st.button("🚀 Lancer le nettoyage", type="primary"):
            df_cleaned = df.copy()
            cleaning_steps = []
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Étape 1: Valeurs manquantes
            if handle_missing != "skip" and total_missing > 0:
                status_text.text("Traitement des valeurs manquantes...")
                progress_bar.progress(25)
                
                df_cleaned = clean_missing_values(df_cleaned, handle_missing)
                cleaning_steps.append({
                    "Étape": "Valeurs manquantes",
                    "Action": f"Remplacement par {handle_missing}",
                    "Valeurs traitées": total_missing
                })
            
            # Étape 2: Doublons
            if remove_duplicates and analysis['duplicates'] > 0:
                status_text.text("Suppression des doublons...")
                progress_bar.progress(50)
                
                duplicates_before = len(df_cleaned)
                df_cleaned = df_cleaned.drop_duplicates()
                duplicates_removed = duplicates_before - len(df_cleaned)
                
                cleaning_steps.append({
                    "Étape": "Doublons",
                    "Action": "Suppression",
                    "Valeurs traitées": duplicates_removed
                })
            
            # Étape 3: Valeurs aberrantes
            if handle_outliers != "skip":
                status_text.text("Traitement des valeurs aberrantes...")
                progress_bar.progress(75)
                
                if handle_outliers == "iqr":
                    df_cleaned, outliers_removed = remove_outliers(df_cleaned, "iqr")
                elif handle_outliers == "zscore":
                    df_cleaned, outliers_removed = remove_outliers(df_cleaned, "zscore", zscore_threshold)
                
                if outliers_removed > 0:
                    cleaning_steps.append({
                        "Étape": "Valeurs aberrantes",
                        "Action": f"Suppression ({handle_outliers})",
                        "Valeurs traitées": outliers_removed
                    })
            
            progress_bar.progress(100)
            status_text.text("Nettoyage terminé !")
            
            # Résumé du nettoyage
            st.success("✅ Nettoyage terminé avec succès !")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Lignes avant nettoyage", len(df))
            with col2:
                st.metric("Lignes après nettoyage", len(df_cleaned))
            
            if cleaning_steps:
                st.subheader("📋 Résumé des actions effectuées")
                steps_df = pd.DataFrame(cleaning_steps)
                st.dataframe(steps_df)
            
            # Aperçu des données nettoyées
            st.subheader("Aperçu des données nettoyées")
            st.dataframe(df_cleaned.head())
            
            # Comparaison avant/après
            st.subheader("📊 Comparaison avant/après")
            
            comparison = pd.DataFrame({
                "Métrique": ["Nombre de lignes", "Valeurs manquantes", "Doublons"],
                "Avant": [len(df), df.isnull().sum().sum(), df.duplicated().sum()],
                "Après": [len(df_cleaned), df_cleaned.isnull().sum().sum(), df_cleaned.duplicated().sum()]
            })
            st.dataframe(comparison)
            
            # Téléchargement
            st.header("4. 💾 Téléchargement")
            
            csv_cleaned = df_cleaned.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Télécharger les données nettoyées",
                data=csv_cleaned,
                file_name=f"donnees_nettoyees_{uploaded_file.name}",
                mime="text/csv",
                type="primary"
            )
            
            # Statistiques finales
            st.info(f"""
            **Résumé final :**
            - Données originales : {len(df)} lignes, {df.shape[1]} colonnes
            - Données nettoyées : {len(df_cleaned)} lignes, {df_cleaned.shape[1]} colonnes
            - Réduction : {len(df) - len(df_cleaned)} lignes supprimées ({((len(df) - len(df_cleaned))/len(df)*100):.2f}%)
            """)
    
    except Exception as e:
        st.error(f"❌ Erreur lors de la lecture du fichier : {str(e)}")
        st.info("Assurez-vous que votre fichier est un CSV valide avec un encodage UTF-8.")

else:
    st.info("👆 Veuillez uploader un fichier CSV pour commencer l'analyse.")
    
    # Informations d'aide
    with st.expander("ℹ️ Aide et informations"):
        st.markdown("""
        **Comment utiliser cette application :**
        
        1. **Upload** : Sélectionnez votre fichier CSV
        2. **Analyse** : Consultez le rapport d'analyse automatique
        3. **Paramétrage** : Choisissez vos options de nettoyage
        4. **Nettoyage** : Lancez le processus de nettoyage
        5. **Téléchargement** : Récupérez vos données propres
        
        **Types de problèmes détectés :**
        - Valeurs manquantes (NaN, null, vides)
        - Doublons (lignes identiques)
        - Valeurs aberrantes (outliers)
        
        **Méthodes de nettoyage :**
        - Remplacement des valeurs manquantes par moyenne/médiane/mode
        - Suppression des doublons
        - Suppression des valeurs aberrantes (IQR ou Z-score)
        """)