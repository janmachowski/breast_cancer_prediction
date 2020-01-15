import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, validation_curve
# importing the dataset
dataset = pd.read_csv('data.csv')
dataset.hist(bins = 10, figsize=(20, 15))
plt.show()
X = dataset.iloc[:, 2:].values
Y = dataset.iloc[:, 1].values
print("Dataset dimensions : {}".format(dataset.shape))

# print separated attributes and class labels
# print("Attributes", pd.DataFrame(X))
# print("Class Labels", pd.DataFrame(X))

# Encoding categorical data values
from sklearn.preprocessing import LabelEncoder

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Feature Scaling - Standarization
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_std = sc.fit_transform(X)
df = pd.DataFrame(X_std)
df.to_numpy()
df.hist(bins = 10, figsize=(20, 15))
plt.show()
# print("X_std")
# print(X_std)
# 5-fold validation
kfold = KFold(n_splits=5)

# Logistic Regression Algorithm
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
cv_results_LRA = cross_val_score(classifier, X_std, Y, cv=kfold)
Y_pred1 = cross_val_predict(classifier, X_std, Y, cv=kfold)
confusion_matrix = pd.crosstab(Y, Y_pred1, rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)
sn.heatmap(confusion_matrix, annot=True)
# plt.show()
print("Logistic Regression Classifier", "%.3f" % cv_results_LRA.mean())
# 97.71 Accuracy

# K-NN Algorithm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
# 5-fold validation
kfold = KFold(n_splits=5)
classifier = KNeighborsClassifier(n_neighbors=3, metric='manhattan', weights='uniform')
cv_results_KNN = cross_val_score(classifier, X_std, Y, cv=kfold)
Y_pred2 = cross_val_predict(classifier, X_std, Y, cv=kfold)
confusion_matrix = pd.crosstab(Y, Y_pred2, rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)
sn.heatmap(confusion_matrix, annot=True)
# plt.show()
print("3-Nearest Neighbors Classifier","%.3f" % cv_results_KNN.mean() )
# 96.48 Accuracy

# SVM
from sklearn.svm import SVC
# 5-fold validation
kfold = KFold(n_splits=5)
classifier = SVC(C=1, kernel='linear', random_state=0)
cv_results_SVC = cross_val_score(classifier, X_std, Y, cv=kfold)
Y_pred3 = cross_val_predict(classifier, X_std, Y, cv=kfold)
confusion_matrix = pd.crosstab(Y, Y_pred3, rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)
sn.heatmap(confusion_matrix, annot=True)
# plt.show()
print("Support Vector Classifier Linear Kernel","%.3f" % cv_results_SVC.mean())
# 97.18 Accuracy

# K-SVM
from sklearn.svm import SVC

classifier = SVC(C=1, kernel='rbf')
cv_results_SVC_RBF = cross_val_score(classifier, X_std, Y, cv=kfold)
Y_pred4 = cross_val_predict(classifier, X_std, Y, cv=kfold)
confusion_matrix = pd.crosstab(Y, Y_pred4, rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)
sn.heatmap(confusion_matrix, annot=True)
# plt.show()
print("Support Vector Classification RBF Kernel","%.3f" % cv_results_SVC_RBF.mean() )
# 97.01 Accuracy

# Naive Bayes
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
cv_results_GNB = cross_val_score(classifier, X_std, Y, cv=kfold)
Y_pred5 = cross_val_predict(classifier, X_std, Y, cv=kfold)
confusion_matrix = pd.crosstab(Y, Y_pred5, rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)
sn.heatmap(confusion_matrix, annot=True)
# plt.show()
print("Gaussian Naive Bayes Classification","%.3f" % cv_results_GNB.mean() )
# 92.68 Accuracy

# Decision Tree Algorithm
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
cv_results_DT = cross_val_score(classifier, X_std, Y, cv=kfold)
Y_pred6 = cross_val_predict(classifier, X_std, Y, cv=kfold)
confusion_matrix = pd.crosstab(Y, Y_pred6, rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)
sn.heatmap(confusion_matrix, annot=True)
# plt.show()
print("Decision Tree Classifier","%.3f" % cv_results_DT.mean() )
# 94.19 Accuracy

# Random Forest Classification Algorithm
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
cv_results_RF = cross_val_score(classifier, X_std, Y, cv=kfold)
Y_pred7 = cross_val_predict(classifier, X_std, Y, cv=kfold)
confusion_matrix = pd.crosstab(Y, Y_pred7, rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)
sn.heatmap(confusion_matrix, annot=True)
# plt.show()
print("Random Forest Classifier","%.3f" % cv_results_RF.mean() )
# 94.55 Accuracy

# MLP
from sklearn.neural_network import MLPClassifier

classifier = MLPClassifier(solver='lbfgs', random_state=0, activation='logistic', hidden_layer_sizes=(15,))
cv_results_MLP = cross_val_score(classifier, X_std, Y, cv=kfold)
Y_pred8 = cross_val_predict(classifier, X_std, Y, cv=kfold)
confusion_matrix = pd.crosstab(Y, Y_pred8, rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)
sn.heatmap(confusion_matrix, annot=True)
# plt.show()
print("MLP Classifier","%.3f" % cv_results_MLP.mean() )
# 96.13 Accuracy
