import pandas as pd
import numpy as np
from data_preperation import split_label_from_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


def temp():
    """kf = KFold(n_splits=n_splits)

        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()
        # finding best k for KNN classifier and n_estimators for RandomForestClassifier
        knn_results = dict()
        forest_results = dict()
        for k, n_estimators in zip(range(1, 7), range(40, 160, 20)):
            knn_accuracy = 0
            forest_accuracy = 0
            for i, (train_index, test_index) in enumerate(kf.split(X_train)):
                KNN_classifier = KNeighborsClassifier(n_neighbors=k)
                KNN_classifier.fit(X_train[train_index], y_train[train_index])
                y_pred = KNN_classifier.predict(X_train[test_index])
                knn_accuracy += accuracy_score(y_train[test_index], y_pred)

                forest = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=20, random_state=1)
                forest.fit(X_train[train_index], y_train[train_index])
                y_pred = forest.predict(X_train[test_index])
                forest_accuracy += accuracy_score(y_train[test_index], y_pred)

            knn_accuracy /= n_splits
            knn_results[k] = knn_accuracy
            forest_accuracy /= n_splits
            forest_results[n_estimators] = forest_accuracy
            print(f'k = {k} knn accuracy = {knn_accuracy}')
            print(f'n_estimators = {n_estimators} forest accuracy = {forest_accuracy}')
        best_k = max(knn_results, key=knn_results.get)
        best_n_estimators = max(forest_results, key=forest_results.get)
        print(f'best k is {best_k}')
        print(f'best n_estimators is {best_n_estimators}')
        # TODO run CV to find min sample split
        best_knn = KNeighborsClassifier(best_k)
        best_forest = RandomForestClassifier(best_n_estimators, min_samples_split=20, random_state=1)
        best_knn.fit(X_train, y_train)
        best_forest.fit(X_train, y_train)
        return best_knn, best_forest"""


# Receive dict with classifiers and params with optional values, and finds the best combination of params values for
# each classifier, it assumes each classifier has at most 2 params
def find_best_params_CV(XY_train: pd.DataFrame, classifiers_params_dict: dict):
    X_train, y_train = split_label_from_data(XY_train)
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    n_splits = 5

    for classifier, params in classifiers_params_dict.items():
        random_state = False
        if 'random_state' in params.keys():
            random_state = params['random_state']
            params.pop('random_state')
        if len(params) == 2:
            param_name_1 = list(params.keys())[0]
            values_1 = params[param_name_1]
            param_name_2 = list(params.keys())[1]
            values_2 = params[param_name_2]
            best_score = 0
            best_values = None
            for value_1 in values_1:
                for value_2 in values_2:
                    params_dict = {param_name_1: value_1, param_name_2: value_2}
                    if random_state:
                        params_dict['random_state'] = random_state
                    clf = classifier(**params_dict)
                    score = np.mean(cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=n_splits))
                    if score > best_score:
                        best_values = {param_name_1: value_1, param_name_2: value_2}
                        best_score = score
        elif len(params) == 1:
            param_name_1 = list(params.keys())[0]
            values_1 = params[param_name_1]
            best_score = 0
            best_values = None
            for value_1 in values_1:
                params_dict = {param_name_1: value_1}
                if random_state:
                    params_dict['random_state'] = random_state
                clf = classifier(**params_dict)
                score = np.mean(cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=n_splits))
                if score > best_score:
                    best_values = {param_name_1: value_1}
                    best_score = score
        else:
            continue
        for param_name in params.keys():
            classifiers_params_dict[classifier][param_name] = best_values[param_name]
    return classifiers_params_dict


def train_and_evaluate(classifiers_params_dict: dict, XY_train: pd.DataFrame, XY_val: pd.DataFrame, score_function=accuracy_score):
    X_train, y_train = split_label_from_data(XY_train)
    X_val, y_val = split_label_from_data(XY_val)
    results = dict()
    for classifier, params in classifiers_params_dict.items():
        clf = classifier(**params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        score = score_function(y_val, y_pred)
        results[classifier] = score
    return results


def train_best_classifier(XY_train: pd.DataFrame, classifier, params):
    X_train, y_train = split_label_from_data(XY_train)
    clf = classifier(**params)
    clf.fit(X_train, y_train)
    return clf


def main():
    XY_train = pd.read_csv('train_transformed.csv', index_col=0, header=0)
    XY_val = pd.read_csv('val_transformed.csv', index_col=0, header=0)
    XY_test = pd.read_csv('test_transformed.csv', index_col=0, header=0)
    classifiers_params_dict = {RandomForestClassifier: {'n_estimators': list(range(60, 100, 20)),
                                                        'min_samples_split': list(range(2, 10, 4)),
                                                        'random_state': 2},
                               KNeighborsClassifier: {'n_neighbors': list(range(1, 7))},
                               SVC: {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']},
                               DecisionTreeClassifier: {'min_samples_split': list(range(2, 10, 4))},
                               SGDClassifier: {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']},
                               GaussianNB: {}}
    classifiers_params_dict = find_best_params_CV(XY_train, classifiers_params_dict)
    results = train_and_evaluate(classifiers_params_dict, XY_train, XY_val)
    # TODO merge train and val data sets and use train best classifier to train and then predict what they asked in section 6
    print(results, classifiers_params_dict)


if __name__ == '__main__':
    main()
