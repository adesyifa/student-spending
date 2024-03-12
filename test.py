import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load dataset
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Prepare features and target variable for training
features = ["age", "gender", "year_in_school", "major"]
X_train = pd.get_dummies(train_data[features])
y_train = train_data["preferred_payment_method"]

# Prepare features for testing
X_test = pd.get_dummies(test_data[features])

# Initialize RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

# Train the model on the entire training set
model.fit(X_train, y_train)

# Predict on the test set
predictions = model.predict(X_test)

# Save predictions to a CSV file
output = pd.DataFrame({'id': test_data['id'], 'preferred_payment_method': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

