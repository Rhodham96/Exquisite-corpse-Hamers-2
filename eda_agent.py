from .base_agent import BaseAgent
from ..utils.kaggle_downloader import KaggleDownloader
import pandas as pd
import os
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns
import numpy as np
from IPython.display import display
import sys
from contextlib import redirect_stdout, redirect_stderr
import tempfile
import traceback
from scipy import stats

class EDAAgent(BaseAgent):
    """Agent spécialisé dans l'analyse exploratoire de données."""
    
    def __init__(self):
        """Initialise l'agent EDA avec les outils spécifiques à l'analyse de données."""
        super().__init__()
        self.downloader = KaggleDownloader(check_credentials=False)
        self.figures = []
        self.output_text = ""
    
    def analyze_dataset(self, dataset_path_or_url: str) -> str:
        """Analyse un dataset local ou depuis Kaggle."""
        # Réinitialiser les figures et sorties
        self.figures = []
        self.output_text = ""
        
        try:
            # Vérifier si c'est un chemin local ou une URL Kaggle
            if os.path.exists(dataset_path_or_url):
                return self._analyze_local_dataset(dataset_path_or_url)
            elif 'kaggle.com' in dataset_path_or_url:
                # Déterminer si c'est une compétition ou un dataset
                if '/competitions/' in dataset_path_or_url or '/c/' in dataset_path_or_url:
                    return self._analyze_kaggle_competition(dataset_path_or_url)
                else:
                    return self._analyze_kaggle_dataset(dataset_path_or_url)
            elif '/' in dataset_path_or_url:
                # Format simple comme 'username/dataset' ou 'competition_id'
                # Essayer avec la méthode de compétition d'abord, puis dataset si cela échoue
                try:
                    return self._analyze_kaggle_competition(dataset_path_or_url)
                except Exception:
                    return self._analyze_kaggle_dataset(dataset_path_or_url)
            else:
                # Considérer comme un ID de compétition simple
                try:
                    return self._analyze_kaggle_competition(dataset_path_or_url)
                except Exception:
                    return f"Chemin ou URL invalide: {dataset_path_or_url}"
        except Exception as e:
            return f"Erreur lors de l'analyse: {str(e)}\n{traceback.format_exc()}"
    
    def execute_code(self, code_to_execute, local_vars=None):
        """Exécute un bloc de code Python et capture la sortie et les graphiques."""
        if local_vars is None:
            local_vars = {}
        
        # Capturer la sortie standard et d'erreur
        output_buffer = io.StringIO()
        
        # Créer un environnement d'exécution
        exec_globals = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            **local_vars
        }
        
        # Liste pour stocker les figures générées
        figures = []
        
        try:
            # Sauvegarder la configuration de matplotlib
            original_backend = plt.get_backend()
            plt.switch_backend('Agg')
            
            # Rediriger stdout et stderr
            with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
                # Exécuter le code
                exec(code_to_execute, exec_globals)
                
                # Capturer toutes les figures ouvertes
                for i in plt.get_fignums():
                    fig = plt.figure(i)
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    img_str = base64.b64encode(buf.read()).decode('utf-8')
                    figures.append(img_str)
                    buf.close()
                
                # Fermer toutes les figures pour éviter les fuites de mémoire
                plt.close('all')
            
            # Restaurer le backend
            plt.switch_backend(original_backend)
            
            return output_buffer.getvalue(), figures, exec_globals
            
        except Exception as e:
            error_msg = f"Erreur lors de l'exécution du code: {str(e)}\n"
            error_msg += traceback.format_exc()
            return error_msg, figures, exec_globals
        
    def _analyze_local_dataset(self, dataset_path: str) -> str:
        """Analyse un dataset local avec exécution de code et visualisations."""
        try:
            # Déterminer le format du fichier et le charger
            if dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
                file_format = 'csv'
            elif dataset_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(dataset_path)
                file_format = 'excel'
            else:
                return f"Format de fichier non supporté: {dataset_path}"
            
            # Générer un code d'analyse EDA avec visualisations
            analysis_code = self._generate_eda_code(df, os.path.basename(dataset_path), file_format)
            
            # Exécuter le code et capturer les résultats
            output, figures, _ = self.execute_code(analysis_code, {'df': df})
            
            # Stocker les résultats pour l'interface
            self.output_text = output
            self.figures = figures
            
            # Construire le résultat markdown
            result = f"# Analyse exploratoire du dataset: {os.path.basename(dataset_path)}\n\n"
            result += "## Code exécuté\n```python\n" + analysis_code + "\n```\n\n"
            result += "## Sortie\n```\n" + output + "\n```\n\n"
            
            # Ajouter les figures base64 (pour Gradio)
            if figures:
                result += "## Visualisations\n\n"
                for i, fig_base64 in enumerate(figures):
                    result += f"![Figure {i+1}](data:image/png;base64,{fig_base64})\n\n"
            
            return result
            
        except Exception as e:
            return f"Erreur lors de l'analyse du dataset: {str(e)}\n{traceback.format_exc()}"
    
    def _generate_eda_code(self, df, filename, file_format):
        """Génère du code d'analyse EDA plus sophistiqué."""
        # Déterminer le type de problème (classification, régression, etc.)
        code_lines = [
            f"# Analyse exploratoire complète du dataset: {filename}",
            "import pandas as pd",
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "from sklearn.preprocessing import StandardScaler",
            "",
            "# Configurer le style des graphiques",
            "sns.set(style='whitegrid')",
            "plt.rcParams['figure.figsize'] = (12, 8)",
            ""
        ]
        
        # Chargement des données (pour référence, df est déjà chargé)
        code_lines.extend([
            f"# Les données sont déjà chargées en mémoire depuis: {filename}",
            "",
            "# 1. Aperçu général des données",
            "print('Dimensions du dataset:', df.shape)",
            "print('\\nPremières lignes du dataset:')",
            "print(df.head())",
            "",
            "# 2. Information sur les types de données",
            "print('\\nTypes de données et valeurs non-nulles:')",
            "print(df.info())",
            "",
            "# 3. Statistiques descriptives",
            "print('\\nStatistiques descriptives:')",
            "print(df.describe())",
            "",
            "# 4. Vérification des valeurs manquantes",
            "missing_values = df.isnull().sum()",
            "print('\\nValeurs manquantes par colonne:')",
            "print(missing_values[missing_values > 0] if missing_values.sum() > 0 else 'Aucune valeur manquante')",
            ""
        ])
        
        # Détection de la variable cible potentielle
        target_col = None
        
        # Rechercher une variable 'target', 'label', 'class', etc.
        target_keywords = ['target', 'label', 'class', 'species', 'survived', 'churn', 'fraud', 'default']
        for keyword in target_keywords:
            matches = [col for col in df.columns if keyword.lower() in col.lower()]
            if matches:
                target_col = matches[0]
                break
        
        # Si pas trouvé, chercher une colonne binaire (0/1)
        if not target_col:
            for col in df.select_dtypes(include=['int64', 'float64']).columns:
                if df[col].nunique() == 2 and set(df[col].unique()).issubset({0, 1}):
                    target_col = col
                    break
        
        # Visualisations
        code_lines.extend([
            "# 5. Visualisations",
            "",
            "# 5.1 Distribution des variables numériques",
            "plt.figure(figsize=(15, 10))",
            "numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns[:6]  # Limiter à 6 colonnes",
            "for i, col in enumerate(numeric_cols):",
            "    plt.subplot(2, 3, i+1)",
            "    sns.histplot(df[col], kde=True)",
            "    plt.title(f'Distribution de {col}')",
            "plt.tight_layout()",
            "plt.show()",
            ""
        ])
        
        # Visualisation des corrélations
        if len(df.select_dtypes(include=['float64', 'int64']).columns) > 1:
            code_lines.extend([
                "# 5.2 Matrice de corrélation",
                "plt.figure(figsize=(12, 10))",
                "correlation_matrix = df.select_dtypes(include=['float64', 'int64']).corr()",
                "sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)",
                "plt.title('Matrice de corrélation')",
                "plt.show()",
                ""
            ])
        
        # Si une variable cible est identifiée, ajouter des visualisations spécifiques
        if target_col:
            code_lines.append(f"# 5.3 Analyses spécifiques à la variable cible: {target_col}")
            
            # Pour les cibles catégorielles
            if df[target_col].dtype == 'object' or df[target_col].nunique() < 10:
                code_lines.extend([
                    f"# Distribution de la variable cible",
                    f"plt.figure(figsize=(10, 6))",
                    f"sns.countplot(y=df['{target_col}'])",
                    f"plt.title('Distribution de {target_col}')",
                    f"plt.show()",
                    "",
                    f"# Relation entre variables numériques et la cible",
                    f"numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns[:4]",
                    f"if len(numeric_cols) > 0:",
                    f"    plt.figure(figsize=(14, 10))",
                    f"    for i, col in enumerate(numeric_cols):",
                    f"        if col != '{target_col}':",  # Éviter de comparer la cible avec elle-même
                    f"            plt.subplot(2, 2, i+1 if i < 4 else 4)",
                    f"            sns.boxplot(x='{target_col}', y=col, data=df)",
                    f"            plt.title(f'Distribution de {{col}} par {target_col}')",
                    f"    plt.tight_layout()",
                    f"    plt.show()",
                    "",
                    f"# Si possible, créer un pairplot pour les relations multivariées",
                    f"if len(df.columns) <= 10:",  # Limiter pour les datasets plus petits
                    f"    sns.pairplot(df, hue='{target_col}', diag_kind='kde', plot_kws={{'alpha': 0.6}})",
                    f"    plt.suptitle('Pairplot des variables avec coloration par {target_col}', y=1.02)",
                    f"    plt.show()",
                    ""
                ])
            # Pour les cibles numériques (régression)
            else:
                code_lines.extend([
                    f"# Distribution de la variable cible",
                    f"plt.figure(figsize=(10, 6))",
                    f"sns.histplot(df['{target_col}'], kde=True)",
                    f"plt.title('Distribution de {target_col}')",
                    f"plt.show()",
                    "",
                    f"# Relation entre variables numériques et la cible",
                    f"numeric_cols = [col for col in df.select_dtypes(include=['float64', 'int64']).columns if col != '{target_col}'][:4]",
                    f"if len(numeric_cols) > 0:",
                    f"    plt.figure(figsize=(14, 10))",
                    f"    for i, col in enumerate(numeric_cols[:4]):",
                    f"        plt.subplot(2, 2, i+1)",
                    f"        sns.regplot(x=col, y='{target_col}', data=df, scatter_kws={{'alpha': 0.5}})",
                    f"        plt.title(f'Relation entre {{col}} et {target_col}')",
                    f"    plt.tight_layout()",
                    f"    plt.show()",
                    ""
                ])
        
        # Variables catégorielles
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            code_lines.extend([
                "# 5.4 Analyse des variables catégorielles",
                "cat_cols = df.select_dtypes(include=['object', 'category']).columns",
                "if len(cat_cols) > 0:",
                "    for col in cat_cols[:3]:  # Limiter à 3 colonnes",
                "        plt.figure(figsize=(10, 6))",
                "        value_counts = df[col].value_counts().sort_values(ascending=False)",
                "        if len(value_counts) > 15:",  # Si trop de catégories, limiter l'affichage
                "            value_counts = value_counts.head(15)",
                "            plt.title(f'Top 15 valeurs pour {col}')",
                "        else:",
                "            plt.title(f'Distribution des valeurs pour {col}')",
                "        sns.barplot(x=value_counts.index, y=value_counts.values)",
                "        plt.xticks(rotation=45, ha='right')",
                "        plt.tight_layout()",
                "        plt.show()",
                ""
            ])
        
        # Analyse des valeurs aberrantes (outliers)
        code_lines.extend([
            "# 6. Détection des valeurs aberrantes",
            "numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns",
            "plt.figure(figsize=(15, 10))",
            "for i, col in enumerate(numeric_cols[:6]):  # Limiter à 6 colonnes",
            "    plt.subplot(2, 3, i+1)",
            "    sns.boxplot(y=df[col])",
            "    plt.title(f'Boxplot de {col}')",
            "plt.tight_layout()",
            "plt.show()",
            "",
            "# 7. Conclusion et insights",
            "print('\\nConclusion et insights:')",
            "print('- Dimensions du dataset:', df.shape)",
            "print('- Types de variables:', len(df.select_dtypes(include=['float64', 'int64']).columns), 'numériques,', len(df.select_dtypes(include=['object', 'category']).columns), 'catégorielles')",
            "missing_percentage = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100",
            "print(f'- Pourcentage de valeurs manquantes: {missing_percentage:.2f}%')",
            ""
        ])
        
        # Si une cible est identifiée, ajouter des statistiques supplémentaires
        if target_col:
            code_lines.append(f"print('- Variable cible identifiée: {target_col}')")
            
            # Pour la classification, ajouter la distribution des classes
            if df[target_col].dtype == 'object' or df[target_col].nunique() < 10:
                code_lines.extend([
                    f"class_distribution = df['{target_col}'].value_counts(normalize=True) * 100",
                    f"print('- Distribution des classes:')",
                    f"for cls, pct in class_distribution.items():",
                    f"    print(f'  * {{cls}}: {{pct:.2f}}%')",
                    ""
                ])
        
        # Ajouter des analyses avancées
        code_lines.extend([
            "# 8. Analyse temporelle",
            "date_cols = [col for col in df.columns if any(word in col.lower() for word in ['date', 'time', 'year', 'month', 'day'])]",
            "if date_cols:",
            "    for col in date_cols[:1]:  # Premier champ de date trouvé",
            "        try:",
            "            df[col] = pd.to_datetime(df[col])",
            "            plt.figure(figsize=(12, 6))",
            "            if isinstance(df.index, pd.DatetimeIndex) or col in df.columns:",
            "                time_col = df.index if isinstance(df.index, pd.DatetimeIndex) else df[col]",
            "                df_ts = df.select_dtypes(include=['float64', 'int64']).iloc[:, 0]  # Première colonne numérique",
            "                plt.plot(time_col, df_ts)",
            "                plt.title(f'Évolution de {df_ts.name} dans le temps')",
            "                plt.xticks(rotation=45)",
            "                plt.tight_layout()",
            "                plt.show()",
            "        except Exception as e:",
            "            print(f'Analyse temporelle impossible: {str(e)}')",
            "",
            
            "# 9. Analyse des formes de distribution et tests statistiques",
            "from scipy import stats",
            "numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns[:5]",
            "for col in numeric_cols:",
            "    data = df[col].dropna()",
            "    if len(data) > 8:",  # Suffisamment de données pour l'analyse",
            "        # Test de normalité",
            "        stat, p = stats.shapiro(data) if len(data) < 5000 else (0, 0)",
            "        plt.figure(figsize=(12, 5))",
            "        plt.subplot(1, 2, 1)",
            "        sns.histplot(data, kde=True)",
            "        plt.title(f'Distribution de {col}')",
            "        plt.subplot(1, 2, 2)",
            "        stats.probplot(data, plot=plt)",
            "        plt.title(f'Q-Q Plot de {col}')",
            "        plt.tight_layout()",
            "        plt.show()",
            "        if p > 0:",
            "            print(f'Test de normalité pour {col}: p-value={p:.4f} - ', 'Normal' if p > 0.05 else 'Non normal')",
            "",
            
            "# 10. Recommandations de preprocessing",
            "print('\\nRecommandations de preprocessing:')",
            "# Pour les variables numériques",
            "skewed_cols = []",
            "for col in df.select_dtypes(include=['float64', 'int64']).columns:",
            "    if df[col].skew() > 1 or df[col].skew() < -1:",
            "        skewed_cols.append((col, df[col].skew()))",
            "if skewed_cols:",
            "    print('- Variables avec forte asymétrie (candidats pour transformation log/sqrt):')",
            "    for col, skew in skewed_cols:",
            "        print(f'  * {col}: skew = {skew:.2f}')",
            "",
            "# Recommandations d'encodage pour variables catégorielles",
            "cat_cols = df.select_dtypes(include=['object', 'category']).columns",
            "if len(cat_cols) > 0:",
            "    print('- Variables catégorielles à encoder:')",
            "    for col in cat_cols:",
            "        n_unique = df[col].nunique()",
            "        if n_unique <= 5:",
            "            print(f'  * {col}: One-Hot Encoding recommandé ({n_unique} catégories)')",
            "        elif 5 < n_unique <= 15:",
            "            print(f'  * {col}: Target Encoding ou Label Encoding ({n_unique} catégories)')",
            "        else:",
            "            print(f'  * {col}: Embedding ou Feature Hashing recommandé ({n_unique} catégories)')",
            ""
        ])
        
        return "\n".join(code_lines)
    
    def _analyze_kaggle_dataset(self, dataset_url: str) -> str:
        """Analyse un dataset depuis Kaggle."""
        try:
            # Télécharger le dataset
            data_path = self.downloader.download_dataset(dataset_url)
            
            if isinstance(data_path, str) and data_path.startswith("Erreur"):
                return data_path
            
            # Trouver les fichiers de données
            data_files = self.downloader.get_dataset_files(data_path)
            
            if not data_files:
                return "Aucun fichier CSV ou Excel trouvé dans le dataset."
            
            # Analyser le premier fichier
            main_file = data_files[0]
            result = f"# Dataset Kaggle: {os.path.basename(dataset_url)}\n\n"
            result += self._analyze_local_dataset(main_file)
            
            # Mentionner les autres fichiers
            if len(data_files) > 1:
                result += "\n\n## Autres fichiers disponibles\n"
                for i, file in enumerate(data_files[1:], 1):
                    result += f"{i}. {os.path.basename(file)}\n"
            
            return result
            
        except Exception as e:
            return f"Erreur lors de l'analyse du dataset Kaggle: {str(e)}\n{traceback.format_exc()}"
    
    def _analyze_kaggle_competition(self, competition_url: str) -> str:
        """Analyse les données d'une compétition Kaggle."""
        try:
            # Télécharger les données de la compétition
            data_path = self.downloader.download_competition_data(competition_url)
            
            if isinstance(data_path, str) and data_path.startswith("Erreur"):
                return data_path
            
            # Trouver les fichiers de données
            data_files = self.downloader.get_dataset_files(data_path)
            
            if not data_files:
                return "Aucun fichier CSV ou Excel trouvé dans les données de la compétition."
            
            # Analyser le premier fichier (généralement train.csv)
            main_file = [f for f in data_files if "train" in f.lower()]
            if not main_file:
                main_file = data_files[0]  # Prendre le premier fichier si pas de train.csv
            else:
                main_file = main_file[0]  # Prendre le premier fichier train.csv
            
            # Obtenir les informations sur la compétition
            competition_id = self.downloader._extract_competition_id(competition_url)
            competition_info = self.downloader.get_competition_info(competition_id)
            
            result = f"# Compétition Kaggle: {competition_id}\n\n"
            
            # Ajouter les informations de la compétition si disponibles
            if isinstance(competition_info, dict):
                result += f"## Informations sur la compétition\n"
                result += f"- **Titre**: {competition_info.get('title', 'Non disponible')}\n"
                result += f"- **Description**: {competition_info.get('description', 'Non disponible')[:500]}...\n"
                result += f"- **Deadline**: {competition_info.get('deadline', 'Non disponible')}\n"
                result += f"- **Récompense**: {competition_info.get('reward', 'Non disponible')}\n\n"
            
            # Ajouter l'analyse du fichier principal
            result += f"## Analyse du fichier {os.path.basename(main_file)}\n\n"
            result += self._analyze_local_dataset(main_file)
            
            # Mentionner les autres fichiers
            other_files = [f for f in data_files if f != main_file]
            if other_files:
                result += "\n\n## Autres fichiers disponibles\n"
                for i, file in enumerate(other_files, 1):
                    result += f"{i}. {os.path.basename(file)}\n"
            
            return result
            
        except Exception as e:
            return f"Erreur lors de l'analyse de la compétition Kaggle: {str(e)}\n{traceback.format_exc()}" 