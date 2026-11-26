
import pandas as pd
import joblib
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("dataset/creditcard.csv")

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.to_list()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.to_list()

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42,
        max_depth=10,
        is_unbalance=True,
        verbosity=-1
    ))
])

model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]


thresholds = [0.30, 0.20, 0.17, 0.15, 0.10, 0.05, 0.02]

for threshold in thresholds:

    y_pred = (y_prob >= threshold).astype(int)

    print("\n================================")
    print("THRESHOLD:", threshold)
    print("================================")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

joblib.dump({
    "model": model,
    "threshold": 0.30
}, "credit_card.pkl")

print("\nModel saved successfully!")
