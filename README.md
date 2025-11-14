Task 1 : Unemployment Analysis with Python 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("unemployment.csv")

# Display first few rows
print(data.head())

# Check missing values
print(data.isnull().sum())

# Plot unemployment rate
plt.figure(figsize=(10,5))
sns.lineplot(data=data, x="Date", y="Unemployment_Rate")
plt.title("Unemployment Rate Over Time")
plt.xlabel("Year")
plt.ylabel("Unemployment Rate")
plt.show()

# Region-wise analysis
plt.figure(figsize=(10,6))
sns.barplot(data=data, x="Region", y="Unemployment_Rate")
plt.title("Region-wise Unemployment Analysis")
plt.xticks(rotation=45)
plt.show()
