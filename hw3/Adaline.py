import numpy as np
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs


class Adaline:
    def __init__(self, lr=0.01, num_iter=100, tol=1e-3):
        self.lr = lr
        self.num_iter = num_iter
        self.losses = []
        self.train_accuracy = []
        self.weights = None
        self.n_iter_ = num_iter
        self.tol = tol

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1] + 1)
        X_with_bias = bias_trick(X)
        for i in range(self.num_iter):
            for x, y_ in zip(X_with_bias, y):
                output = x @ self.weights
                grad = x * (y_ - output)
                self.weights += self.lr * grad

            output = X_with_bias @ self.weights
            loss = 0.5 * ((y - output) ** 2).mean()
            self.losses.append(loss)
            train_accuracy = evaluate_performance(self, X, y)
            self.train_accuracy.append(train_accuracy)
            if len(self.losses) > 1:
                if loss > self.losses[-2] - self.tol:
                    self.n_iter_ = i
                    break

    def predict(self, X):
        X = bias_trick(X)
        res = X @ self.weights
        res[res > 0] = 1
        res[res < 0] = -1
        return res


def evaluate_performance(clf, X, y):
    y_pred_train = clf.predict(X)
    train_accuracy = accuracy_score(y, y_pred_train)
    return train_accuracy


def bias_trick(X):
    X = np.concatenate([np.ones([*X.shape[:-1], 1], dtype=X.dtype), X], axis=1)
    return X


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('blue', 'red', 'lightgreen', 'gray', 'cyan')
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


def load_iris_data_set():
    iris = datasets.load_iris()
    X_iris = iris.data
    y_iris = iris.target
    iris_classes = iris.target_names

    # Scaling
    iris_mean = np.mean(X_iris, axis=0)
    iris_std = np.std(X_iris, axis=0) + 1e-5
    X_iris = (X_iris - iris_mean) / iris_std

    return X_iris, y_iris, iris_classes


def load_digits_data_set():
    digits = datasets.load_digits()
    X_digits = digits.data
    y_digits = digits.target
    digits_classes = digits.target_names
    # Scaling
    digits_mean = np.mean(X_digits, axis=0)
    digits_std = np.std(X_digits, axis=0) + 1e-5
    X_digits = (X_digits - digits_mean) / digits_std

    return X_digits, y_digits, digits_classes


def one_vs_all(y, true_class):
    y_new = np.zeros_like(y)
    y_new[y != true_class] = -1
    y_new[y == true_class] = 1
    return y_new


def train_and_evaluate_adaline(X, y, lr, num_iter):
    num_of_classes = len(set(y))
    results = dict()
    for current_class in range(num_of_classes):
        if num_of_classes == 2 and current_class == 1:
            continue

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True)
        y_train = one_vs_all(y_train, current_class)
        y_test = one_vs_all(y_test, current_class)
        clf = Adaline(lr=lr, num_iter=num_iter)
        clf.fit(X_train, y_train)

        test_accuracy = evaluate_performance(clf, X_test, y_test)
        results[current_class] = {'train_accuracy': clf.train_accuracy, 'test_accuracy': test_accuracy, 'losses': clf.losses,
                                  'epochs': clf.n_iter_}
    return results


def train_and_evaluate_perceptron(X, y, lr, max_iter):
    num_of_classes = len(set(y))
    results = dict()
    for current_class in range(num_of_classes):
        if num_of_classes == 2 and current_class == 1:
            continue

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True)
        y_train = one_vs_all(y_train, current_class)
        y_test = one_vs_all(y_test, current_class)

        clf = Perceptron(alpha=lr, max_iter=max_iter, random_state=2)
        clf.fit(X_train, y_train)

        train_accuracy = evaluate_performance(clf, X_train, y_train)
        test_accuracy = evaluate_performance(clf, X_test, y_test)
        results[current_class] = {'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy,
                                    'epochs': clf.n_iter_}
    return results


def plot_graphs(results, dataset_name):
    for i in range(len(results)):
        plt.plot(range(1, len(results[i]['train_accuracy']) + 1), results[i]['train_accuracy'])
        plt.legend(['train'], loc='lower right')
        plt.xlabel('Epochs')
        plt.ylabel('accuracy')
        plt.title(f'{dataset_name} Dataset - type {i} vs all')
        plt.show()

        if len(results[i]['losses']) > 0:
            plt.plot(range(1, len(results[i]['losses']) + 1), results[i]['losses'], marker='o')
            plt.xlabel('Epochs')
            plt.ylabel('Sum-squared-error')
            plt.title(f'{dataset_name} Dataset - type {i} vs all')
            plt.show()


def compare_algorithms(X, y, classes_name, lr, num_iter, dataset_name, show_graphs):
    results = train_and_evaluate_adaline(X, y, lr, num_iter)
    num_of_classes = len(set(y))
    for i in range(num_of_classes):
        if num_of_classes == 2 and i == 1:
            continue
        print(f'{dataset_name} - Adaline type {classes_name[i]} vs all train accuracy = {results[i]["train_accuracy"][-1]},'
              f' test accuracy = {results[i]["test_accuracy"]} \nepochs = {results[i]["epochs"]}')

    if show_graphs:
        plot_graphs(results, dataset_name)

    results = train_and_evaluate_perceptron(X, y, lr=lr, max_iter=num_iter)
    for i in range(num_of_classes):
        if num_of_classes == 2 and i == 1:
            continue
        print(f'{dataset_name} - Perceptron type {classes_name[i]} vs all train accuracy = {results[i]["train_accuracy"]},'
              f' test accuracy = {results[i]["test_accuracy"]} \nepochs = {results[i]["epochs"]}')


def create_dataset_better_for_adaline():
    # Non-linear separable dataset
    X = np.random.randn(1000, 2)
    y = np.zeros(X.shape[0])
    y[X[:, 1] > ((X[:, 0]) ** 2)] = 1
    plt.title('Dataset separable by y=x^2')
    plt.plot(X[y == 0, 0], X[y == 0, 1], 'r.')
    plt.plot(X[y == 1, 0], X[y == 1, 1], 'b.')
    plt.show()
    return X, y


def create_dataset_better_for_perceptron():
    # linear separable dataset
    X = np.random.randn(1000, 2)
    y = np.zeros(X.shape[0])
    X[X[:, 1] > 0, 1] += 5
    y[X[:, 1] > 0] = 1
    plt.title('Dataset separable by y=2')
    plt.plot(X[y == 0, 0], X[y == 0, 1], 'r.')
    plt.plot(X[y == 1, 0], X[y == 1, 1], 'b.')
    plt.show()
    return X, y


def sklearn_comparison(X,y, lr, num_iter):
    heldout = [0.95, 0.90, 0.75, 0.50, 0.1]
    rounds = 20

    classifiers = [
        ("Adaline", Adaline(lr=lr, num_iter=num_iter)),
        ("Perceptron", Perceptron(alpha=lr, max_iter=num_iter)),
    ]

    xx = 1. - np.array(heldout)

    for name, clf in classifiers:
        print("training %s" % name)
        rng = np.random.RandomState(42)
        yy = []
        for i in heldout:
            yy_ = []
            for r in range(rounds):
                X_train, X_test, y_train, y_test = \
                    train_test_split(X, y, test_size=i, random_state=rng, stratify=y)
                y_train = one_vs_all(y_train, 5)
                y_test = one_vs_all(y_test, 5)

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                yy_.append(1 - np.mean(y_pred == y_test))
            yy.append(np.mean(yy_))
        plt.plot(xx, yy, label=name)

    plt.legend(loc="upper right")
    plt.xlabel("Proportion train")
    plt.ylabel("Test Error Rate")
    plt.show()


def main():

    X_iris, y_iris, iris_classes = load_iris_data_set()

    X_digits, y_digits, digits_classes = load_digits_data_set()

    show_graphs = False
    # --------- iris ----------
    compare_algorithms(X_iris, y_iris, iris_classes, 1e-4, 1000, 'iris', show_graphs)

    # ---------- digits ------------
    compare_algorithms(X_digits, y_digits, digits_classes, 1e-4, 1000, 'digits', show_graphs)

    X, y = create_dataset_better_for_adaline()
    compare_algorithms(X, y, [-1, 1], 1e-4, 1000, 'y=x^2', False)

    X, y = create_dataset_better_for_perceptron()
    compare_algorithms(X, y, [-1, 1], 1e-4, 1000, 'y=2', False)

    sklearn_comparison(X_digits, y_digits, 1e-4, 1000)


if __name__ == "__main__":
    main()
