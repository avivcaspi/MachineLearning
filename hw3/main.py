import pandas as pd
import numpy as np
from data_preperation import split_label_from_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


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


def pick_best_classifier(results):
    best_clf = max(results, key=results.get)
    return best_clf


def main():
    import time

    start = time.time()
    automate_model_selection = False
    XY_train = pd.read_csv('train_transformed.csv', index_col=0, header=0)
    XY_val = pd.read_csv('val_transformed.csv', index_col=0, header=0)
    XY_test = pd.read_csv('test_transformed.csv', index_col=0, header=0)
    classifiers_params_dict = {RandomForestClassifier: {'n_estimators': list(range(60, 400, 30)),
                                                        'min_samples_split': list(range(2, 20, 2)),
                                                        'random_state': 2},
                               KNeighborsClassifier: {'n_neighbors': list(range(1, 10))},
                               SVC: {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']},
                               DecisionTreeClassifier: {'min_samples_split': list(range(2, 20, 2))},
                               GaussianNB: {}}
    classifiers_params_dict = find_best_params_CV(XY_train, classifiers_params_dict)
    print(f'Classifiers best params : \n{classifiers_params_dict}')

    results = train_and_evaluate(classifiers_params_dict, XY_train, XY_val)  # used to pick best model manually

    best_clf = pick_best_classifier(results) if automate_model_selection else RandomForestClassifier

    XY_train_new = pd.concat([XY_train, XY_val])
    clf = train_best_classifier(XY_train_new, best_clf, classifiers_params_dict[best_clf])

    # First prediction
    X_test, y_test = split_label_from_data(XY_test)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)

    print(f'Test Error : {1 - accuracy}')
    print(f'Confusion Matrix : \n{conf_mat}')

    division_of_voters = {party: list(y_pred).count(party) for party in set(y_pred)}
    party_with_majority = max(division_of_voters, key=division_of_voters.get)

    print(f'Party that will win majority of votes (in relation to Test set) : {party_with_majority}')

    n_voters = len(X_test)
    division_of_voters.update((key, round(value * 100 / n_voters, 3)) for key, value in division_of_voters.items())

    print(f'Division of voters : \n{division_of_voters}')
    bins = np.linspace(0, 12, 26)

    plt.hist([y_test, y_pred], bins, label=['Real votes', 'Prediction'])
    plt.xticks(range(0, 13, 1))
    plt.legend(loc='upper right')
    plt.show()

    end = time.time()
    print(f'Time it took : {end - start}')


if __name__ == '__main__':
    main()