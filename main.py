import pandas as pd
import seaborn as sn

from sklearn.metrics import accuracy_score

# importing the dataset
dataset = pd.read_csv('data.csv')

X = dataset.iloc[:, 2:31].values
Y = dataset.iloc[:, 1].values

dataset.head()

print("Cancer dataset dimensions : {}".format(dataset.shape))

dataset.groupby('diagnosis').size()

# Visualization of data
dataset.groupby('diagnosis').hist(figsize=(12, 12))

dataset.isnull().sum()
dataset.isna().sum()

dataframe = pd.DataFrame(Y)

# Encoding categorical data values
from sklearn.preprocessing import LabelEncoder

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Logistic Regression Algorithm

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, Y_train)
Y_pred1 = classifier.predict(X_test)
print("Logistic Regression Algorithm")
print(accuracy_score(Y_test, Y_pred1))

confusion_matrix = pd.crosstab(Y_test, Y_pred1, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)
# 94.4 Accuracy

# K-NN Algorithm
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, Y_train)
Y_pred2 = classifier.predict(X_test)
print("KNN")
print(accuracy_score(Y_test, Y_pred2))

confusion_matrix = pd.crosstab(Y_test, Y_pred2, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)
# 95.8 Accuracy

# SVM
from sklearn.svm import SVC

classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, Y_train)
Y_pred3 = classifier.predict(X_test)
print("SVM")
print(accuracy_score(Y_test, Y_pred3))

confusion_matrix = pd.crosstab(Y_test, Y_pred3, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)
# 96.5 Accuracy

# K-SVM
from sklearn.svm import SVC

classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, Y_train)
Y_pred4 = classifier.predict(X_test)
print("K-SVM")
print(accuracy_score(Y_test, Y_pred4))

confusion_matrix = pd.crosstab(Y_test, Y_pred4, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)
# 96.5 Accuracy

# Naive Bayes
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train, Y_train)
Y_pred5 = classifier.predict(X_test)
print("Naive Bayes")
print(accuracy_score(Y_test, Y_pred5))

confusion_matrix = pd.crosstab(Y_test, Y_pred5, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)
# 92.3 Accuracy

# Decision Tree Algorithm
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, Y_train)
Y_pred6 = classifier.predict(X_test)
print("DT")
print(accuracy_score(Y_test, Y_pred6))

confusion_matrix = pd.crosstab(Y_test, Y_pred6, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)
# 95.1 Accuracy

# Random Forest Classification Algorithm
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train, Y_train)
Y_pred7 = classifier.predict(X_test)
print("Random Forests")
print(accuracy_score(Y_test, Y_pred7))

confusion_matrix = pd.crosstab(Y_test, Y_pred7, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)
# 96.5 Accuracy

# MLP
from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(solver='lbfgs', random_state=0, activation='logistic', hidden_layer_sizes=(15,))
classifier.fit(X_train, Y_train)
Y_pred8 = classifier.predict(X_test)
print("MLP")
print(accuracy_score(Y_test, Y_pred8))

confusion_matrix = pd.crosstab(Y_test, Y_pred8, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)
# 96.5 Accuracy