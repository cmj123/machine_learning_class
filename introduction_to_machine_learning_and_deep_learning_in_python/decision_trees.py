# Import libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import tree

# Get data
data = pd.read_csv("iris_data.csv")
# print(data.info())
# print(data.head())
# print(data.Class.unique())

# Split dataset
data.features = data[["SepalLength","SepalWidth","PetalLength","PetalWidth"]]
data.targets = data.Class
# print(data.features.head())
# print(data.targets.head())
feature_train, feature_test, target_train, target_test = train_test_split(data.features, data.targets,test_size=.2)

# DecisionTreeClassifier model - define, fit and predict
model = DecisionTreeClassifier(criterion='entropy')
model.fitted =model.fit(feature_train, target_train)
model.predictions = model.fitted.predict(feature_test)

# Evaluation
print(confusion_matrix(target_test, model.predictions))
print(accuracy_score(target_test, model.predictions))

# Cross validation
predicted = cross_val_score(model, data.features,data.targets,cv=10)
print(predicted)

# Visualise decision tree
# tree.export_graphviz(model.fitted)
