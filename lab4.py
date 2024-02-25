from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

id3_classifier = DecisionTreeClassifier(criterion="entropy")
id3_classifier.fit(X_train, y_train)

predictions = id3_classifier.predict(X_test)

conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)


print("Classification Report:")
print(classification_report(y_test, predictions, target_names=iris.target_names))

plt.figure(figsize=(12, 8))
plot_tree(id3_classifier, filled=True, feature_names=iris.feature_names, class_names=[str(name) for name in iris.target_names])
plt.show()
