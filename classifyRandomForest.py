from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc, precision_recall_curve, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve

from joblib import dump,load
import gc


df = pd.read_csv('testData.csv', header=None)

# loading, setting and spliting the dataframe  
print("\nreading data ...")
df = pd.read_csv('final_data.csv', header=None)
print(df.head())

print("\nsplitting data ...")

train_df = df.sample(frac=0.8, random_state=42)
print(train_df.head())

test_df = df.drop(train_df.index)

X_train = train_df.iloc[:, 1:]
y_train = train_df.iloc[:, 0]
X_test = test_df.iloc[:, 1:]
y_test = test_df.iloc[:, 0]

del df
del test_df
del train_df

gc.collect()


# initialize the model / load a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
# model = load('models/LRClassifier-v001.joblib')

# Train the classifier on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred_proba = model.predict_proba(X_test)
y_pred_proba = y_pred_proba[:, 1]
y_pred = np.array([1 if x >= 0.5 else 0 for x in y_pred_proba])

# metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print('Classification Report:\n', class_report)

# plot

# conf matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# roc_auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - KNN')
plt.legend(loc='lower right')
plt.show()

# precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

plt.figure()
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Random Forest')
plt.show()