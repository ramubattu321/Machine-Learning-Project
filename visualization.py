import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create images folder
os.makedirs("images", exist_ok=True)

# Load your dataset (change file name if needed)
df = pd.read_csv("your_dataset.csv")

# -------------------------------
# 1. Correlation Heatmap
# -------------------------------
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="Blues")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("images/correlation_heatmap.png")
plt.close()

# -------------------------------
# 2. Model Comparison (MANUAL VALUES)
# Replace with your real results
# -------------------------------
models = ["Linear Regression", "SVM", "Random Forest", "Gradient Boosting"]
scores = [0.60, 0.68, 0.75, 0.80]  # replace with your actual scores

plt.figure(figsize=(8,5))
plt.bar(models, scores)
plt.title("Model Performance Comparison")
plt.xlabel("Models")
plt.ylabel("Score")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("images/model_comparison.png")
plt.close()

# -------------------------------
# 3. Actual vs Predicted (IF AVAILABLE)
# -------------------------------
# Uncomment only if you have these variables

# plt.figure(figsize=(6,6))
# plt.scatter(y_test, y_pred)
# plt.xlabel("Actual Values")
# plt.ylabel("Predicted Values")
# plt.title("Actual vs Predicted")
# plt.tight_layout()
# plt.savefig("images/actual_vs_predicted.png")
# plt.close()

# -------------------------------
# 4. Feature Importance (OPTIONAL)
# -------------------------------
# Uncomment if using Random Forest

# importance = model.feature_importances_
# features = X.columns

# plt.figure(figsize=(8,5))
# plt.barh(features, importance)
# plt.title("Feature Importance")
# plt.tight_layout()
# plt.savefig("images/feature_importance.png")
# plt.close()

print("Visualizations saved in 'images/' folder")
