"""Titanic Survival Prediction using Logistic Regression."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)

# 1. Load data
df = pd.read_csv("data/titanic.csv")
print("Shape:", df.shape)
print(df.isnull().sum()[df.isnull().sum() > 0], "\n")

# 2. Clean data
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df = df.drop(columns=["Cabin", "PassengerId", "Name", "Ticket"])

# 3. Feature engineering / encoding
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

X = df.drop(columns=["Survived"])
y = df["Survived"]

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Scale numeric features (helps logistic regression converge)
num_cols = ["Age", "Fare", "FamilySize"]
mean, std = X_train[num_cols].mean(), X_train[num_cols].std()
X_train[num_cols] = (X_train[num_cols] - mean) / std
X_test[num_cols] = (X_test[num_cols] - mean) / std

# 6. Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 7. Evaluate
pred = model.predict(X_test)
print(f"Accuracy : {accuracy_score(y_test, pred):.3f}")
print(f"Precision: {precision_score(y_test, pred):.3f}")
print(f"Recall   : {recall_score(y_test, pred):.3f}")
print(f"F1-score : {f1_score(y_test, pred):.3f}\n")
print(classification_report(y_test, pred, target_names=["Died", "Survived"]))

# 8. Confusion matrix plot
ConfusionMatrixDisplay(confusion_matrix(y_test, pred),
                       display_labels=["Died", "Survived"]).plot(cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")

# 9. Feature coefficients
coef = pd.Series(model.coef_[0], index=X.columns).sort_values()
print("Coefficients:\n", coef.round(3))
plt.figure(figsize=(7, 4))
coef.plot(kind="barh", color="steelblue")
plt.title("Feature Coefficients")
plt.savefig("feature_coefficients.png", dpi=150, bbox_inches="tight")
