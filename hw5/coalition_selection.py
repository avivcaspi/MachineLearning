import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis
from data_preperation import split_label_from_data, insert_label_to_data
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

classes = ['Blues', 'Browns', 'Greens', 'Greys', 'Khakis', 'Oranges', 'Pinks', 'Purples', 'Reds', 'Turquoises',
           'Violets', 'Whites', 'Yellows']


def clustering_score(estimator, X, y):
    """
    scoring function for clusters
    :param estimator: classifier
    :param X: data
    :param y: labels
    :return: score of the clf (higher the better)
    """
    score = 0
    clusters_pred = estimator.predict(X)
    for cluster in set(clusters_pred):
        samples_party_in_cluster = y[clusters_pred == cluster]
        parties_in_cluster = set(samples_party_in_cluster)

        for party in parties_in_cluster:
            total_party_samples = np.sum(y == party)
            party_samples_in_cluster = np.sum(samples_party_in_cluster == party)
            percentage_in_cluster = party_samples_in_cluster / total_party_samples
            if percentage_in_cluster > 0.6:
                score += percentage_in_cluster
    return score


def find_best_cluster_model_params(XY_train, classifiers_params_dict):
    X_train, y_train = split_label_from_data(XY_train)
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    n_splits = 3
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)

    new_params_dict = dict()
    for classifier, params in classifiers_params_dict.items():
        new_params_dict[classifier] = dict()
        random_state = False
        if 'random_state' in params.keys():
            random_state = params['random_state']
            params.pop('random_state')
        if len(params) == 2:
            param_name_1 = list(params.keys())[0]
            values_1 = params[param_name_1]
            param_name_2 = list(params.keys())[1]
            values_2 = params[param_name_2]
            best_score = -1
            best_values = None
            for value_1 in values_1:
                for value_2 in values_2:
                    params_dict = {param_name_1: value_1, param_name_2: value_2}
                    if random_state:
                        params_dict['random_state'] = random_state
                    score = 0
                    for k, (train_index, val_index) in enumerate(kf.split(X_train)):
                        clf = classifier(**params_dict)
                        clf.fit(X_train[train_index], y_train[train_index])
                        # train_score = clustering_score(clf, X_train[train_index], y_train[train_index])
                        score += clustering_score(clf, X_train[val_index], y_train[val_index])
                    if score > best_score:
                        best_values = {param_name_1: value_1, param_name_2: value_2}
                        best_score = score
        elif len(params) == 1:
            param_name_1 = list(params.keys())[0]
            values_1 = params[param_name_1]
            best_score = -1
            best_values = None
            for value_1 in values_1:
                params_dict = {param_name_1: value_1}
                if random_state:
                    params_dict['random_state'] = random_state
                clf = classifier(**params_dict)
                score = np.mean(cross_val_score(clf, X_train, y_train, scoring=clustering_score, cv=n_splits))
                if score > best_score:
                    best_values = {param_name_1: value_1}
                    best_score = score
        else:
            continue
        for param_name in params.keys():
            new_params_dict[classifier][param_name] = best_values[param_name]
    return new_params_dict


def find_possible_coalitions(XY_train, XY_val, clusters_params):
    X_train, y_train = split_label_from_data(XY_train)
    X_val, y_val = split_label_from_data(XY_val)
    n_val_sample = len(y_val)
    possible_coalitions = set()
    for model, params in clusters_params.items():
        clf = model(**params)
        clf.fit(X_train)
        clusters_pred = clf.predict(X_val)
        for cluster in set(clusters_pred):
            samples_party_in_cluster = y_val[clusters_pred == cluster]
            parties_in_cluster = set(samples_party_in_cluster)
            coalition = []
            coalition_size = 0
            for party in parties_in_cluster:
                total_party_samples = np.sum(y_val == party)
                party_samples_in_cluster = np.sum(samples_party_in_cluster == party)
                percentage_in_cluster = party_samples_in_cluster / total_party_samples
                if percentage_in_cluster > 0.85:
                    coalition.append(party)
                    coalition_size += total_party_samples
                    if coalition_size / n_val_sample >= 0.51:
                        possible_coalitions.add(tuple(sorted(coalition)))
    return possible_coalitions


def calculate_variance(df, possible_coalitions):
    X, y = split_label_from_data(df)

    coalitions_vars = dict()
    for coalition in possible_coalitions:
        indices = y.isin(coalition)
        X_coalition = X.loc[indices, :]
        coalition_variance = X_coalition.var(axis=0)
        coalitions_vars[coalition] = coalition_variance.sum()
    return coalitions_vars


def calculate_oppo_coali_dist(df, possible_coalitions):
    X, y = split_label_from_data(df)

    coalitions_dists = dict()
    for coalition in possible_coalitions:
        coalition_indices = y.isin(coalition).to_numpy()
        opposition_indices = ~coalition_indices
        X_coalition = X.loc[coalition_indices, :]
        X_opposition = X.loc[opposition_indices, :]
        dist = np.linalg.norm((X_coalition.mean(axis=0) - X_opposition.mean(axis=0)), 1)
        coalitions_dists[coalition] = dist
    return coalitions_dists


def find_best_coalition_cluster(XY_train, XY_test_pred):
    classifiers_params_dict = {KMeans: {'n_clusters': list(range(2, 7)), 'max_iter': [3000]},
                               GaussianMixture: {'n_components': list(range(2, 7)), 'max_iter': [3000]}}
    best_clusters_params = find_best_cluster_model_params(XY_train, classifiers_params_dict)
    print(f'Best clusters params : {best_clusters_params}')

    possible_coalitions_test = find_possible_coalitions(XY_train, XY_test_pred, best_clusters_params)
    coalitions_variance = calculate_variance(XY_test_pred, possible_coalitions_test)
    coalitions_opposition_distance = calculate_oppo_coali_dist(XY_test_pred, possible_coalitions_test)
    sorted_variance = sorted(coalitions_variance.items(), key=lambda x: x[1])
    print(f'TEST SET - Sum variances of each coalition : {sorted_variance}')
    sorted_dists = sorted(coalitions_opposition_distance.items(), key=lambda x: x[1], reverse=True)
    print(f'TEST SET - Dists from opposition of each coalition : {sorted_dists}')

    plt.barh(range(len(coalitions_variance.keys())), coalitions_variance.values(), orientation='horizontal',
             align="center")
    plt.title('Variance of coalitions')
    plt.xlabel(f'Variance')
    plt.ylabel('Coalition index')
    plt.show()

    plt.barh(range(len(coalitions_opposition_distance.keys())), coalitions_opposition_distance.values(), orientation='horizontal',
             align="center")
    plt.title('Dist between coalition to opposition')
    plt.xlabel(f'Dist')
    plt.ylabel('Coalition index')
    plt.show()

    # Picked manually
    chosen_coalition = (2, 3, 4, 5, 6, 8, 9, 11, 12)

    return chosen_coalition


def find_possible_coalitions_generative(XY_train, XY_test, prob_matrix):
    X_train, y_train = split_label_from_data(XY_train)
    X_test, y_test = split_label_from_data(XY_test)
    labels = set(y_train)
    possible_coalitions = set()
    for label in labels:
        coalition = [label]
        coalition_size = 0

        while coalition_size < 0.51 or len(coalition) < len(labels):
            parties = np.zeros(len(labels))
            for party1 in coalition:
                for party2 in labels - set(coalition):
                    parties[party2] += prob_matrix[party1][party2] + prob_matrix[party2][party1]
            coalition.append(np.argmax(parties))
            coalition_size = X_test[y_test.isin(coalition)].shape[0] / X_test.shape[0]
            if coalition_size >= 0.51:
                possible_coalitions.add(tuple(sorted(coalition)))

    coalitions_variance = calculate_variance(XY_train, possible_coalitions)
    coalitions_opposition_distance = calculate_oppo_coali_dist(XY_train, possible_coalitions)

    sorted_variance = sorted(coalitions_variance.items(), key=lambda x: x[1])
    print(f'Sum variances of each coalition : {sorted_variance[:10]}')
    sorted_dists = sorted(coalitions_opposition_distance.items(), key=lambda x: x[1], reverse=True)
    print(f'Dists from opposition of each coalition : {sorted_dists[:10]}')

    # Picked manually
    best_coalition = (3, 4, 5, 6, 8, 9, 11, 12)
    return best_coalition


def calc_prob_matrix(XY_train, XY_trainVal, model):
    X_train, y_train = split_label_from_data(XY_train)
    model.fit(X_train, y_train)

    X_train, y_train = split_label_from_data(XY_trainVal)

    labels = set(y_train)
    label_sample = dict()
    for label in labels:
        label_sample[label] = X_train[y_train == label]

    prob_matrix = np.zeros((len(labels), len(labels)))

    for label1 in labels:
        pred_prob = model.predict_proba(label_sample[label1])
        for label2 in labels:
            prob_matrix[label1][label2] = sum(pred_prob[:, label2])
        prob_matrix[label1] = prob_matrix[label1] / len(label_sample[label1])

    return prob_matrix


def find_best_coalition_generative(XY_train, XY_val, XY_test_pred):
    classifiers_list = [GaussianNB(), LinearDiscriminantAnalysis()]

    XY_trainVal = pd.concat([XY_train, XY_val])

    X_trainVal, y_trainVal = split_label_from_data(XY_trainVal)
    X_train, y_train = split_label_from_data(XY_train)
    X_val, y_val = split_label_from_data(XY_val)

    best_model = None
    best_model_score = 0
    for model in classifiers_list:
        score = np.average(cross_val_score(model, X_trainVal, y_trainVal, cv=5))
        if score > best_model_score:
            best_model = model
            best_model_score = score

    print(f'best model: = {best_model}, with score: {best_model_score}')
    prob_matrix = calc_prob_matrix(XY_train, XY_trainVal, best_model)

    coalition = find_possible_coalitions_generative(XY_trainVal, XY_test_pred, prob_matrix)
    print(f'Generative model best Coalition : {sorted(list(coalition))}')
    return sorted(list(coalition))


def main():
    '''XY_train = pd.read_csv('train_transformed.csv', index_col=0, header=0)
    XY_val = pd.read_csv('val_transformed.csv', index_col=0, header=0)
    XY_test = pd.read_csv('test_transformed.csv', index_col=0, header=0)
    X_train, y_train = split_label_from_data(XY_train)
    X_test, y_test = split_label_from_data(XY_test)

    # Predicting model
    clf = RandomForestClassifier(n_estimators=220, min_samples_split=10)
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    XY_test_pred = insert_label_to_data(X_test, y_test_pred)

    # Finding best coalition using clustering models
    clustring_coalition = find_best_coalition_cluster(XY_train, XY_val, XY_test_pred)
    print(f'Cluster coalition is : {[classes[i] for i in clustring_coalition]}')
    # Finding best generative using clustering models
    generative_coalition = find_best_coalition_generative(XY_train, XY_val, XY_test_pred)
    print(f'Generative coalition is : {[classes[i] for i in generative_coalition]}')

    leading_features_for_each_party(pd.concat([XY_train, XY_val]))

    find_group_factors(pd.concat([XY_train, XY_val]), XY_test_pred, clustring_coalition)

    find_factors_to_change_results(XY_train, XY_val, XY_test_pred)'''


if __name__ == '__main__':
    main()
