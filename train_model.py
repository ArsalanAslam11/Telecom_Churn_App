import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("churn_cleaned.csv")

#  SELECT ONLY FEATURES YOU USE IN STREAMLIT
features = ["tenure", "MonthlyCharges", "Contract"]
target = "Churn"

X = df[features]
y = df[target]

num_cols = ["tenure", "MonthlyCharges"]
cat_cols = ["Contract"]

preprocessor = ColumnTransformer([
    ("num", "passthrough", num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "rf_model.joblib")

print(" Model trained with selected features only")
