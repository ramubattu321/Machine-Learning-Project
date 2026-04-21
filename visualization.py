import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("images", exist_ok=True)

df = pd.read_csv("production_data.csv")   # make sure file name matches

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="Blues")
plt.title("Correlation Heatmap")
plt.tight_layout()

plt.savefig("images/correlation_heatmap.png", dpi=300)
plt.close()

print("Saved images/correlation_heatmap.png")
