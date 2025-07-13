import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the processed dataset
data = pd.read_csv('C:\Dev\heart-risk-ai\data\processed\heart_disease_engineered.csv')

# Load the trained model, feature names, and scaler
model_dict = joblib.load('C:\Dev\heart-risk-ai\models\heart_risk_ensemble_v3.pkl')
model = model_dict['ensemble_model']
feature_names = model_dict['feature_names']
scaler = model_dict['scaler']
optimal_threshold = model_dict.get('optimal_threshold', 0.5)

# Split the data into features (X) and target (y)
X = data[feature_names]
y = data['target']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the test data
X_test_scaled = scaler.transform(X_test)

# Make predictions on the test set
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_pred_proba >= optimal_threshold).astype(int)

# Calculate performance metrics
auc = roc_auc_score(y_test, y_pred_proba)
sensitivity = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Calculate specificity from the confusion matrix
cm = confusion_matrix(y_test, y_pred)
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

# Print metrics
print(f'AUC-ROC: {auc:.4f}')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall (Sensitivity): {sensitivity:.4f}')
print(f'F1-Score: {f1:.4f}')
print(f'Specificity: {specificity:.4f}')

# Print confusion matrix values
print('\nConfusion Matrix:')
print(cm)

# Generate and save the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.title('Confusion Matrix for Model v3')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('C:/Dev/heart-risk-ai/reports/confusion_matrix_analysis_v3.png')

print('\nConfusion matrix saved to reports/confusion_matrix_analysis_v3.png')