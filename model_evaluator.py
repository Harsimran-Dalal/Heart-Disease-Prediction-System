import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/heart.csv")
X = df.drop("target", axis=1)
y = df["target"]

scaler = joblib.load("scaler.joblib")
rf = joblib.load("rf_model.joblib")

X_scaled = scaler.transform(X)
y_pred = rf.predict(X_scaled)

print(classification_report(y, y_pred))

cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
