# File that contains the trained model and saves it
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

model_dir = "../app/model"
model_name = "model.pkl"

# Load the dataset & Clean the data

df = pd.read_csv("../datasets/german_credit_data.csv",index_col=0)
df["checking_saving_accounts"] = df["Saving accounts"].combine_first(df["Checking account"])
mode = df["checking_saving_accounts"].mode()[0]
df['checking_saving_accounts'] = df['checking_saving_accounts'].fillna(mode)
df = df.drop(["Saving accounts","Checking account"],axis=1)
df['Risk'] = df['Risk'].replace({'bad':1,'good':0})

# Split in features and target
X = df.drop('Risk',axis=1)
y = df.Risk

# Split in train / test data
