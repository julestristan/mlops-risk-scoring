# File that contains the trained model and saves it
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

model_dir = "../app/model"
model_name = "model.pkl"

# Load the dataset & Clean the data

df = pd.read_csv("../datasets/german_credit_data.csv",index_col=0)
df["checking_saving_accounts"] = df["Saving accounts"].combine_first(df["Checking account"])
mode = df["checking_saving_accounts"].mode()[0]
df['checking_saving_accounts'] = df['checking_saving_accounts'].fillna(mode)
df["Credit_amount"] = df["Credit amount"].copy()
df = df.drop('Credit amount',axis=1)
df = df.drop(["Saving accounts","Checking account"],axis=1)
df['Risk'] = df['Risk'].replace({'bad':1,'good':0})

# Split in features and target
X = df.drop('Risk',axis=1)
y = df.Risk

# On différencie selon le type de données de chaque feature
num_features = ['Age','Job','Credit_amount','Duration']
cat_features = ['Sex','Housing','Purpose','checking_saving_accounts']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Définition des pipelines de transformation
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combinaison des pré-traitements
preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)],
    remainder='drop') # Supprime toutes les colonnes non listées

# Construction du Pipeline complet
classifier = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.01,
    max_depth=4,
    random_state=42
)

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])

# Entraînement
print("Début de l'entraînement du pipeline...")
full_pipeline.fit(X_train, y_train)
print("Entraînement terminé.")

# Évaluation (pour information)
y_pred_proba = full_pipeline.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"Performance sur l'ensemble de test (ROC AUC) : {roc_auc:.4f}")

# Sauvegarde du Pipeline complet (le modèle et le pré-traitement)
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, model_name)
joblib.dump(full_pipeline, model_path)
print(f"Pipeline complet sauvegardé dans : {model_path}")