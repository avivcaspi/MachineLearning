from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from main import find_best_params_CV, classes, train_and_evaluate, pick_best_classifier, train_best_classifier, createTransportationLists
from preprocessing import split_label_from_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TaskOrientedClf:
    def __init__(self, classifiers_params_dict=None):
        self.XY_train = None
        self.XY_val = None
        self.party_winner_clf = None
        self.division_of_voters_clf = None
        self.transportation_clf = None
        if classifiers_params_dict is None:
            self.classifiers_params_dict = {RandomForestClassifier: {'n_estimators': list(range(60, 100, 40)),
                                                                'min_samples_split': list(range(2, 10, 4)),
                                                                'random_state': 2},
                                       KNeighborsClassifier: {'n_neighbors': list(range(1, 7))},
                                       SVC: {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}}
        else:
            self.classifiers_params_dict =  classifiers_params_dict

    def find_best_task_clf(self, XY_train, XY_val):
        """
        Find best classifier for each task (classifiers are picked from classifiers_params_dict)
        :param XY_train: train data
        :param XY_val: validation data
        :return: dict of clf per task
        """
        self.XY_train = XY_train
        self.XY_val = XY_val
        params_task_dict = self._find_best_params()
        clf_scores = self._train_and_evaluate_each_task(params_task_dict)
        best_clf_per_task = {k: pick_best_classifier(scores) for k, scores in clf_scores.items()}
        trained_clf_per_trask = {k: train_best_classifier(pd.concat([XY_train, XY_val]), clf, params_task_dict[k][clf])
                                 for k, clf in best_clf_per_task.items()}
        self.party_winner_clf = trained_clf_per_trask['party_winner']
        self.division_of_voters_clf = trained_clf_per_trask['division_of_voters']
        self.transportation_clf = trained_clf_per_trask['transportation']
        return trained_clf_per_trask

    def _find_best_params(self):
        """
        find best params for each clf in classifiers_params_dict for each task
        :return: dict of params for each task
        """
        best_params = dict()
        best_params['party_winner'] = self._find_params_party_winner()
        best_params['division_of_voters'] = self._find_params_division_of_voters()
        best_params['transportation'] = self._find_params_transportation()
        return best_params

    def _find_params_party_winner(self):
        print('Finding best params for party winner')
        best_parmas = find_best_params_CV(self.XY_train, self.classifiers_params_dict, scoring=party_winner_scoring)
        print(f'Party winner params : {best_parmas}')
        return best_parmas

    def _find_params_division_of_voters(self):
        print('Finding best params for division')
        best_parmas = find_best_params_CV(self.XY_train, self.classifiers_params_dict, scoring=division_scoring)
        print(f'Division params : {best_parmas}')
        return best_parmas

    def _find_params_transportation(self):
        print('Finding best params for transportation')
        best_parmas = find_best_params_CV(self.XY_train, self.classifiers_params_dict, scoring=transportation_scoring)
        print(f'Transportation params : {best_parmas}')
        return best_parmas

    def _train_and_evaluate_each_task(self, params_task_dict):
        """
        train and evaluate each clf in params_task_dict according to the params in the dict for each task
        :param params_task_dict: dict of clf and params for each task
        :return: classifiers scores for each task and scoring function
        """
        clf_scoring = dict()
        clf_scoring['party_winner'] = train_and_evaluate(params_task_dict['party_winner'], self.XY_train, self.XY_val, party_winner_scoring)
        clf_scoring['division_of_voters'] = train_and_evaluate(params_task_dict['division_of_voters'], self.XY_train, self.XY_val, division_scoring)
        clf_scoring['transportation'] = train_and_evaluate(params_task_dict['transportation'], self.XY_train, self.XY_val, transportation_scoring)
        return clf_scoring


def party_winner_scoring(estimator, X, y):
    """
    scoring function for party winner task
    :param estimator: classifier
    :param X: data
    :param y: labels
    :return: score of the clf (higher the better)
    """
    true_division_of_voters = {party: list(y).count(classes.index(party)) for party in classes}
    true_winner = max(true_division_of_voters, key=true_division_of_voters.get)
    y_pred = estimator.predict(X)
    pred_division_of_voters = {party: list(y_pred).count(classes.index(party)) for party in classes}
    sorted_division = [(k,v) for k, v in sorted(pred_division_of_voters.items(), key=lambda item: item[1])]
    pred_winner_index = sorted_division.index((true_winner, pred_division_of_voters[true_winner]))
    return pred_winner_index


def division_scoring(estimator, X, y):
    """
    scoring function for division of voters task
    :param estimator: classifier
    :param X: data
    :param y: labels
    :return: score of the clf (higher the better)
    """
    true_division_of_voters = {party: list(y).count(classes.index(party)) for party in classes}
    y_pred = estimator.predict(X)
    pred_division_of_voters = {party: list(y_pred).count(classes.index(party)) for party in classes}
    division_dists = [abs(true_division_of_voters[party] - pred_division_of_voters[party]) for party in classes]
    total_dist = np.sum(division_dists)
    return len(y) - (total_dist + 1e-5)


def transportation_scoring(estimator, X, y):
    """
    scoring function for transportation services task
    :param estimator: classifier
    :param X: data
    :param y: labels
    :return: score of the clf (higher the better)
    """
    predict_proba_existance = getattr(estimator, "predict_proba", None)
    if callable(predict_proba_existance):
        y_pred = estimator.predict(X)
        accuracy = accuracy_score(y, y_pred)
        return accuracy
    else:
        return 0


def get_results_for_each_task(best_clf_trained, XY_test):
    """
    calculate each task test score for each clf found
    :param best_clf_trained: dict of best clf trained per task
    :param XY_test: test data
    :return: scores dict
    """
    accuracy_dict = dict()
    party_winner_dict = dict()
    division_dict = dict()
    transportation_dict = dict()
    X_test, y_test = split_label_from_data(XY_test)
    for k, clf in best_clf_trained.items():
        y_pred = clf.predict(X_test)
        accuracy_dict[k] = accuracy_score(y_test, y_pred)
        pred_division_of_voters = {party: list(y_pred).count(classes.index(party)) for party in classes}
        party_winner_dict[k] = max(pred_division_of_voters, key=pred_division_of_voters.get)
        division_dict[k] = pred_division_of_voters
        predict_proba_existance = getattr(clf, "predict_proba", None)
        if callable(predict_proba_existance):
            transportation_dict[k] = createTransportationLists(clf, X_test, 0.6)
    return accuracy_dict, party_winner_dict, division_dict, transportation_dict


def main():
    XY_train = pd.read_csv('train_transformed.csv', index_col=0, header=0)
    XY_val = pd.read_csv('val_transformed.csv', index_col=0, header=0)
    XY_test = pd.read_csv('test_transformed.csv', index_col=0, header=0)
    X_test, y_test = split_label_from_data(XY_test)
    task_clf = TaskOrientedClf()
    best_clf_trained = task_clf.find_best_task_clf(XY_train, XY_val)
    accuracy_dict, party_winner_dict, division_dict, transportation_dict = get_results_for_each_task(best_clf_trained, XY_test)
    for task, clf in best_clf_trained.items():
        print(f'Best classifier for {task} is {clf.__class__.__name__}')

    print(f'Accuracy score for clf of each task : {accuracy_dict}')

    y_pred = best_clf_trained['division_of_voters'].predict(X_test)
    bins = np.linspace(0, 12, 26)
    y_test = [classes[y] for y in y_test]
    y_pred = [classes[y] for y in y_pred]
    plt.hist([y_test, y_pred], bins, label=['Real votes', 'Prediction'])
    plt.xticks(range(0, 13, 1), rotation='vertical')
    plt.legend(loc='upper right')
    plt.title('Prediction - Real comparison')
    plt.show()

    explode = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0)
    plt.pie(division_dict['division_of_voters'].values(), explode=explode, labels=division_dict['division_of_voters'].keys(),
            autopct='%1.1f%%', shadow=True, startangle=0)
    plt.title('Division of voters')
    plt.axis('equal')

    plt.show()


if __name__ == '__main__':
    main()
