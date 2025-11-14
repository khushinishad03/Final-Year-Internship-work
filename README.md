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




Task 2 : Email Spam Detection with Machine Learning 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("spam.csv")

# Labeling: spam=1, ham=0
data['label'] = data['Category'].map({'spam':1, 'ham':0})

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(data['Message'], data['label'], test_size=0.2)

# Vectorization
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Model training (Naive Bayes)
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Predictions
predictions = model.predict(X_test_vectorized)

# Evaluation
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))




Task 3 : Sales Prediction using Python 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("sales.csv")

# Display data
print(data.head())

# Features and target
X = data[['Marketing_Spend', 'Month', 'Store_Size']]
y = data['Sales']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("R² Score:", r2_score(y_test, predictions))
