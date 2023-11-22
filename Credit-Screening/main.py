import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

# Load the CSV file into a pandas dataframe
df = pd.read_csv(r"C:\Users\sinuk\OneDrive\Documents\AIML_SEM2\PR\german_credit_data.csv")
print(df.info())

# Preprocess the data
# Handle missing values
df.ffill(inplace=True)  # Use .ffill() to forward-fill missing values
df = df.drop(columns=['Index'])

# Encode categorical variables
df = pd.get_dummies(df, columns=['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose'])
print(df)
print(df.describe())

# Classify 'Job' into categories (0 = high risk, 1 = medium risk, 2 or 3 = low risk)
job_mapping = {
    0: 0, 1: 1, 2: 2, 3: 2  # Adjust the mapping as needed
}
df['Job_category'] = df['Job'].map(job_mapping)
print(df['Job_category'])

# Use 'Job_category' as the new target variable
x = df.drop(['Job_category', 'Job'], axis=1)
print(x)
y = df['Job_category']
print(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create a MinMaxScaler and fit it to the training data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Transform the testing data using the same scaler
X_test_scaled = scaler.transform(X_test)

# Create a random forest classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the scaled training data
rfc.fit(X_train_scaled, y_train)

# Evaluate the performance of the classifier on the scaled testing data
y_pred = rfc.predict(X_test_scaled)

# Calculate classification metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Calculate and print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

print('Accuracy:', accuracy)
print('Precision:','%.3f' %precision)
print('Recall:', recall)
print('F1 score:','%.3f' %f1)

# Create a Seaborn heatmap for the confusion matrix visualization
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['High Risk', 'Medium Risk', 'Low Risk'], yticklabels=['High Risk', 'Medium Risk', 'Low Risk'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
