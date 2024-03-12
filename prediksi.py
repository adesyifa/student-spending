import pandas as pd

train_data = pd.read_csv("student_spending (1).csv")
train_data.head()
test_data = pd.read_csv("student_spending (1).csv")
test_data.head()
female_major_counts = train_data[train_data['gender'] == 'Female']['preferred_payment_method'].value_counts()
most_common_major_female = female_major_counts.idxmax()
rate_most_common_major_female = (female_major_counts.max() / len(train_data[train_data['gender'] == 'Female'])) * 100
print("Jenis transaksi yang paling sering dipakai perempuan:", most_common_major_female)
print("%siswa perempuan yang memakai jenis transaksi:", rate_most_common_major_female)
male_major_counts = train_data[train_data['gender'] == 'Male']['preferred_payment_method'].value_counts()
most_common_major_male = male_major_counts.idxmax()
rate_most_common_major_male = (male_major_counts.max() / len(train_data[train_data['gender'] == 'Male'])) * 100
print("Jenis transaksi yang paling sering dipakai perempuan:", most_common_major_male)
print("%siswa laki-laki yang memakai jenis transaksi:", rate_most_common_major_male)
from sklearn.ensemble import RandomForestClassifier
y = train_data["preferred_payment_method"]

features = ["age", "gender", "year_in_school", "major"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'Id': test_data['id'], 'preferred_payment_method': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")


features = ["age", "gender", "year_in_school", "major"]
X_train = pd.get_dummies(train_data[features])
y_train = train_data["preferred_payment_method"]


X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)


model.fit(X_train, y_train)


predictions = model.predict(X_test)

output = pd.DataFrame({'id': test_data['id'], 'preferred_payment_method': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

from sklearn.metrics import accuracy_score

# Load the ground truth (actual labels) for the test set if available
ground_truth = pd.read_csv("submission.csv")

# Assuming the ground truth labels are in the 'Survived' column
actual_labels = ground_truth['preferred_payment_method']

# Calculate accuracy based on the actual labels and predictions
accuracy = accuracy_score(actual_labels, predictions)
print(f"Accuracy on Test Set: {accuracy}")

from sklearn import tree
import matplotlib.pyplot as plt


tree_to_visualize = model.estimators_[0]


class_names = ['Credit/Debit Card', 'Cash', 'Mobile Payment App']  

plt.figure(figsize=(12, 12))
tree.plot_tree(tree_to_visualize,
               feature_names=X.columns,
               class_names=class_names,
               filled=True)
plt.savefig('random_forest_tree.png')
plt.show()


