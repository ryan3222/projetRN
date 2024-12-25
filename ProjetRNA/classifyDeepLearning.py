from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_curve, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc

import gc

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

# setting the nn model
print("\nsetting model ...")
input_shape = [X_train.shape[1]]
model = tf.keras.Sequential([

    ## setup the model layers
    ### input layer with 64 nodes and relu activation
    tf.keras.layers.Dense(64, activation='relu', input_shape = (X_train.shape[1],)),
    ### hidden layer with 16 nodes and relu activation
    tf.keras.layers.Dense(16, activation='relu'),
    ### output layer with 1 node and sigmoid activation
    tf.keras.layers.Dense(1, activation='sigmoid')
])

print("\ncompiling model ...")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# loading the model / weights
# print("\nloading model ...")
# model = tf.keras.models.load_model('models/DLClassifier-v001')
# model.load_weights('path/to/dl_weights.h5')

# summery
model.summary()

# comment if model / weights loaded
print("\nfitting model ...")
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# save model / weights
model.save_weights('weights/dl_weights.h5')
model.save('models/DLClassifier-v001')

# evaluating
print("\nevaluating model ...")
val_loss, val_accuracy = model.evaluate(X_test, y_test)

# predicting
y_pred_proba = model.predict(X_test)
y_pred = np.array([1 if x >= 0.5 else 0 for x in y_pred_proba])


# metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

class_report = classification_report(y_test, y_pred)

print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print('Classification Report:\n', class_report)

# plots

# conf matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Deep Learning')
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
plt.title('ROC - Deep Learning')
plt.legend(loc='lower right')
plt.show()

# precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

plt.figure()
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Deep Learning')
plt.show()


# loss - accuracy curves
plt.figure(figsize=(12, 4))

# acc
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

## loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
