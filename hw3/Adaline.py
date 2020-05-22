import numpy as np
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap


class Adaline:
    def __init__(self, lr=0.01, num_iter=100):
        self.lr = lr
        self.num_iter = num_iter
        self.losses = []
        self.weights = None

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1] + 1)
        X = bias_trick(X)
        for i in range(self.num_iter):
            output = X @ self.weights
            loss = 0.5 * ((y - output) ** 2).sum()
            grad = X.T @ (y - output)
            self.weights += self.lr * grad

            self.losses.append(loss)

    def predict(self, X):
        X = bias_trick(X)
        res = X @ self.weights
        res[res > 0] = 1
        res[res < 0] = -1
        return res


def bias_trick(X):
    X = np.concatenate([np.ones([*X.shape[:-1], 1], dtype=X.dtype), X], axis=1)
    return X


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, cmap=cmap(idx),
                    marker=markers[idx], label=cl)


if __name__ == "__main__":
    iris = datasets.load_iris()
    X_iris = iris.data
    y_iris = iris.target
    y_iris[y_iris == 2] = 1
    y_iris[y_iris == 0] = -1
    X_iris = X_iris[:, [0,2]]
    # X_iris[:, 0] = (X_iris[:,0] - X_iris[:,0].mean() )/X_iris[:,0].std()
    # X_iris[:, 1] = (X_iris[:, 1] - X_iris[:, 1].mean()) / X_iris[:, 1].std()

    digits = datasets.load_digits(2)
    X_digits = digits.data
    y_digits = digits.target
    y_digits[y_digits == 0] = -1
    X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.15, shuffle=True)

    clf = Adaline(lr=1e-4, num_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'iris accuracy : {accuracy}')

    perceptron = Perceptron()
    perceptron.fit(X_train, y_train)
    y_pred = perceptron.predict(X_test)
    perc_acc = accuracy_score(y_test, y_pred)
    print(f'perceptron accuracy : {perc_acc}')

    plot_decision_regions(X_iris, y_iris, classifier=clf)

    plt.title('Adaptive Linear Neuron - Gradient Descent')
    plt.xlabel('sepal length')
    plt.ylabel('petal length')
    plt.legend(loc='upper left')
    plt.show()

    plot_decision_regions(X_iris, y_iris, classifier=perceptron)

    plt.title('Perceptron - Gradient Descent')
    plt.xlabel('sepal length')
    plt.ylabel('petal length')
    plt.legend(loc='upper left')
    plt.show()

    plt.plot(range(1, len(clf.losses) + 1), clf.losses, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Sum-squared-error')
    plt.show()