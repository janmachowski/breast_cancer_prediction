from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.base import clone
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.tree import DecisionTreeClassifier

alpha = .05
cancer = load_breast_cancer()
X_d = cancer.data
y = cancer.target
# Feature Scaling - Standarization
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X_d)
# print(X.shape, y.shape)


clfs = {
    "LR": LogisticRegression(),
    "KNN": KNeighborsClassifier(n_neighbors=3, metric='manhattan', weights='uniform'),
    "SVC": SVC(C=1, kernel='linear', random_state=0),
    "KSVC": SVC(C=1, kernel='rbf'),
    "GNB": GaussianNB(),
    "DT": DecisionTreeClassifier(criterion='entropy', random_state=0),
    "RF": RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0),
    "MLP": MLPClassifier(solver='lbfgs', random_state=0, activation='logistic', hidden_layer_sizes=(15,))
}

res = np.zeros((len(clfs), 5))
skf = StratifiedKFold(n_splits=5)
for i, (train, test) in enumerate(skf.split(X, y)):
    for j, clfn in enumerate(clfs):
        clf = clone(clfs[clfn])
        clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])

        score = accuracy_score(y[test], y_pred)

        res[j, i] = "%.3f" % score
mean_scores = np.mean(res, axis=1)
print(["%.3f" % i for i in mean_scores])

print(res,"\n")
results = np.full((len(res), len(res)),False,dtype=bool)
results_p = np.ones((len(res), len(res)))

for i in range(len(res)-1):
    for j in range(i+1,len(res)):
        print(i,j)
        p = wilcoxon(res[i],res[j]).pvalue
        print(p <= alpha and (mean_scores[i] < mean_scores[j]))
        results[i, j] =(p <= alpha and (mean_scores[i] < mean_scores[j]))
        # results[i,i] = False
        # results[j, i] = (p <= alpha and (mean_scores[i] < mean_scores[j]))

        results_p[i,j] = "%.3f" % p

print(results,"\n")
print(results_p)

# print(p <= alpha and (mean_scores[0] < mean_scores[1]))
