import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import streamlit as st



# Load the dataset
df = pd.read_csv("D:\MDTE21\Employee Attrition\Employee-Attrition - Employee-Attrition.csv")

# Drop irrelevant columns (constant values)
df.drop(columns=['EmployeeCount','EmployeeNumber','StandardHours', 'Over18','DailyRate','HourlyRate','MonthlyRate','NumCompaniesWorked'], inplace=True)

    
# Encode categorical variables
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])


# Feature engineering
df['TenurePerJobLevel'] = df['YearsAtCompany'] / (df['JobLevel'] + 1)  # Avoid division by zero
df['PromotionLag'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
    

# 2. Exploratory Data Analysis (EDA)# Plot 1: Attrition rate by Department
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Department', hue='Attrition')
plt.title('Attrition by Department')
plt.savefig('attrition_by_department.png')
plt.close()
    
# Plot 2: Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()
    
# Plot 3: Job Satisfaction vs Attrition
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Attrition', y='JobSatisfaction')
plt.title('Job Satisfaction vs Attrition')
plt.savefig('job_satisfaction_vs_attrition.png')
plt.close()

# Save the preprocessed dataset
df.to_csv("preprocessed_employee_attrition.csv", index=False)
print("Preprocessing complete. The preprocessed dataset is saved as 'preprocessed_employee_attrition.csv'.")


# Split the dataset into features and target variable
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets and handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, random_state=42)

# Model 1: Logistic Regression
# Train the logistic regression model 
Lr_model = LogisticRegression()
Lr_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_lr = Lr_model.predict(X_test)

# Thresholding for probability predictions from Logistic Regression
y_proba_lr = Lr_model.predict_proba(X_test)[:, 1]
y_pred_lr = (y_proba_lr >= 0.5).astype(int)

# Evaluate the logistic regression model
accuracy = accuracy_score(y_test, y_pred_lr)
precision = precision_score(y_test, y_pred_lr)
recall = recall_score(y_test, y_pred_lr)
f1 = f1_score(y_test, y_pred_lr)
roc_auc = roc_auc_score(y_test, y_proba_lr)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_lr)
# Classification report
report = classification_report(y_test, y_pred_lr)

# Display
print("\nLogistic Regression Model Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")
print("\n📊 Confusion Matrix:")
print(cm)
print("\n📃 Classification Report:")
print(report)

# Model 2: Random Forest
# Train the random forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test)

# Thresholding for probability predictions from Random Forest
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]
y_pred_rf = (y_proba_rf >= 0.5).astype(int)

# Evaluate the random forest model
accuracy = accuracy_score(y_test, y_pred_rf)
precision = precision_score(y_test, y_pred_rf)
recall = recall_score(y_test, y_pred_rf)
f1 = f1_score(y_test, y_pred_rf)
roc_auc = roc_auc_score(y_test, y_proba_rf)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_rf)
# Classification report
report = classification_report(y_test, y_pred_rf)

# Display
print("\nRandom Forest Model Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")
print("\n📊 Confusion Matrix:")
print(cm)
print("\n📃 Classification Report:")
print(report)

# Save the models and scaler
joblib.dump(Lr_model, "logistic_regression_model.pkl")
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")


