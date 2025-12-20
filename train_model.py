import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# ---------------------------
# Load dataset
# ---------------------------
df = pd.read_csv("churn_cleaned.csv")

# Target
y = df["Churn"]
X = df.drop("Churn", axis=1)

# Identify columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# ---------------------------
# Preprocessing
# ---------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

# ---------------------------
# Model
# ---------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# ---------------------------
# Pipeline
# ---------------------------
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# ---------------------------
# Train/Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Train
# ---------------------------
pipeline.fit(X_train, y_train)

# ---------------------------
# Evaluate
# ---------------------------
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ---------------------------
# Save Pipeline
# ---------------------------
joblib.dump(pipeline, "rf_model.joblib")

print("✅ Model + Preprocessing saved successfully")
