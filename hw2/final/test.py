from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


def test_accuracy(x_train, y_train, x_test, y_test):
    svm = SVC(gamma='auto', random_state=101)
    forest = RandomForestClassifier(n_estimators=10, random_state=101)
    knn = KNeighborsClassifier()

    svm.fit(x_train, y_train)
    forest.fit(x_train, y_train)
    knn.fit(x_train, y_train)

    y1_hat = svm.predict(x_test)
    y2_hat = forest.predict(x_test)
    y3_hat = knn.predict(x_test)

    acc1 = metrics.accuracy_score(y_test, y1_hat)
    acc2 = metrics.accuracy_score(y_test, y2_hat)
    acc3 = metrics.accuracy_score(y_test, y3_hat)

    return {'svm': round(acc1, 3), 'forest': round(acc2, 3), 'knn': round(acc3, 3)}

