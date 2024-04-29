from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load your data from the CSV file (X should contain feature vectors, and y should contain labels)
# Replace 'your_data.csv' with the path to your CSV file
import pandas as pd
data = pd.read_csv('final_data.csv')

# Separate features (X) and labels (y)
X = data.iloc[:, 1:]  # Assuming features start from the 2nd column
y = data.iloc[:, 0]   # Assuming labels are in the 1st column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = rf_classifier.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)