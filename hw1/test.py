from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


def test_accuracy(x_train, y_train, x_test, y_test):
    svm = SVC(gamma='auto')
    forest = RandomForestClassifier(n_estimators=10)
    knn = KNeighborsClassifier()
    perceptron = Perceptron()
    bn = GaussianNB()

    svm.fit(x_train, y_train)
    forest.fit(x_train, y_train)
    knn.fit(x_train, y_train)
    perceptron.fit(x_train, y_train)
    bn.fit(x_train, y_train)

    y1_hat = svm.predict(x_test)
    y2_hat = forest.predict(x_test)
    y3_hat = knn.predict(x_test)
    y4_hat = perceptron.predict(x_test)
    y5_hat = bn.predict(x_test)

    acc1 = metrics.accuracy_score(y_test, y1_hat)
    acc2 = metrics.accuracy_score(y_test, y2_hat)
    acc3 = metrics.accuracy_score(y_test, y3_hat)
    acc4 = metrics.accuracy_score(y_test, y4_hat)
    acc5 = metrics.accuracy_score(y_test, y5_hat)

    return {'svm': acc1, 'forest': acc2, 'knn': acc3, 'perceptron': acc4, 'bn': acc5}

