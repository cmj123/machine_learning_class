# Import Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import datasets

# Load and split data
dataset = datasets.load_iris()
# print(dataset)

features = dataset.data
targets = dataset.target
# print(features)

feature_train, feature_test, target_train, target_test = train_test_split(features, targets,test_size=.2)

# Model
model = RandomForestClassifier(n_estimators=1000, max_features='sqrt')
fitted_model = model.fit(feature_train, target_train)
predictions  = fitted_model.predict(feature_test)

# Evaluation 
print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))

# Cross validation
predicted = cross_val_score(model, features, targets,cv=10)
print(predicted)
