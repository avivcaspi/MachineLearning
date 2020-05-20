import numpy as np
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt


class Adaline:
    def __init__(self, num_features, num_iter=100):
        self.weights = np.ones(num_features + 1)
        self.num_iter = num_iter

    def fit(self, X_train, y_train, lr=0.01):
        X_train = bias_trick(X_train)
        for i in range(self.num_iter):
            for x, y in zip(X_train, y_train):
                grad = self.grad(x, y)
                self.weights += lr * grad

    def grad(self, x, y):
        grad = (y - self.weights @ x.T) * x
        return grad

    def predict(self, X):
        X = bias_trick(X)
        res = X @ np.reshape(self.weights, (*self.weights.shape, 1))
        res[res > 0] = 1
        res[res < 0] = -1
        return res.reshape(-1)


def bias_trick(X):
    X = np.concatenate([np.ones([*X.shape[:-1], 1], dtype=X.dtype), X], axis=1)
    return X


if __name__ == "__main__":
    X1 = np.random.randint(0,5,(1000,1))
    y1 = np.random.randint(5,10,(1000,1))
    X2 = np.concatenate([X1, np.random.randint(5,10,(1000,1))],axis=0)
    y2 = np.concatenate([y1, np.random.randint(5, 10, (1000, 1))],axis=0)
    # X2 = np.concatenate([X,X1], axis=0)
    y = np.ones(2000)
    y[0:1000] = -1
    plt.figure()

    plt.scatter(X2,y2)
    plt.axes([0, 10, 0, 10])
    plt.show()
    clf = Adaline(X.shape[1], 1000)
    clf.fit(X,y, 0.01)
    pred = np.array([[1,5], [2,9], [6,1], [7,3]])
    y_pred = clf.predict(pred)
    print(y_pred)

    perceptron = Perceptron()
    perceptron.fit(X, y)
    print(perceptron.predict(pred))


