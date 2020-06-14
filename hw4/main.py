import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from data_preperation import split_label_from_data
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from fourth_prediction import *

classes = ['Blues', 'Browns', 'Greens', 'Greys', 'Khakis', 'Oranges', 'Pinks', 'Purples', 'Reds', 'Turquoises', 'Violets', 'Whites', 'Yellows']


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


def cluster_score(estimator, X, y):
    score = silhouette_score(X, estimator.labels_)
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
                score = np.mean(cross_val_score(clf, X_train, y_train, scoring=cluster_score, cv=n_splits))
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


def find_best_coalition_cluster(XY_train, XY_val):
    classifiers_params_dict = {KMeans: {'n_clusters': list(range(2, 6)), 'max_iter': [3000]},
                               GaussianMixture: {'n_components': list(range(2, 6)), 'max_iter': [3000]}}
    best_clusters_params = find_best_cluster_model_params(XY_train, classifiers_params_dict)
    print(f'Best clusters params : {best_clusters_params}')

    possible_coalitions_val = find_possible_coalitions(XY_train, XY_val, best_clusters_params)
    possible_coalitions_train = find_possible_coalitions(XY_train, XY_train, best_clusters_params)
    possible_coalitions = possible_coalitions_train & possible_coalitions_val
    coalitions_variance = calculate_variance(pd.concat([XY_train, XY_val]), possible_coalitions)
    coalitions_opposition_distance = calculate_oppo_coali_dist(pd.concat([XY_train, XY_val]), possible_coalitions)
    sorted_variance = sorted(coalitions_variance.items(), key=lambda x: x[1])
    print(f'Sum variances of each coalition : {sorted_variance}')
    sorted_dists = sorted(coalitions_opposition_distance.items(), key=lambda x: x[1], reverse=True)
    print(f'Dists from opposition of each coalition : {sorted_dists}')
    chosen_coalition = (2, 3, 4, 5, 6, 8, 9, 11, 12)


    return chosen_coalition


def one_vs_all(y, true_class):
    true_indices = y == true_class
    y_new = pd.Series(data=-1, index=y.index)
    y_new[true_indices] = 1
    return y_new


def leading_features_for_each_party(df):
    X, y = split_label_from_data(df)
    num_of_classes = len(set(y))
    features = X.columns

    for current_class in range(num_of_classes):
        y_new = one_vs_all(y, current_class)
        clf = ExtraTreesClassifier(n_estimators=50)
        clf.fit(X, y_new)
        features_scores = clf.feature_importances_

        indices = np.argsort(features_scores)[::-1]

        # Print the feature ranking
        print("Feature ranking:")

        for f in range(X.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, indices[f], features_scores[indices[f]]))

        colors = ['r']*len(features)
        colors[indices[0]] = 'b'
        # Plot the feature importance of the forest
        plt.figure()
        plt.title(f"Feature importance of {classes[current_class]} vs all")
        plt.barh(range(X.shape[1]), features_scores, orientation='horizontal',
                color=colors, align="center", tick_label=features)
        plt.show()


def coalition_score(XY_train, coalition):
    coalition = tuple(coalition)
    var = calculate_variance(XY_train,[coalition])[coalition]
    dist = calculate_oppo_coali_dist(XY_train,[coalition])[coalition]
    return abs(dist)


def find_possible_coalitions_generative(XY_train, XY_test, prob_matrix):
    X_train, y_train = split_label_from_data(XY_train)
    X_test, y_test = split_label_from_data(XY_test)
    labels = set(y_train)
    best_coalition = None
    best_coalition_score = float('-inf')

    for label in labels:
        coalition = [label]
        coalition_size = 0

        while coalition_size < 0.51:
            parties = np.zeros(len(labels))
            for party1 in coalition:
                for party2 in labels - set(coalition):
                    parties[party2] += prob_matrix[party1][party2] + prob_matrix[party2][party1]
            coalition.append(np.argmax(parties))
            coalition_size = X_test[y_test.isin(coalition)].shape[0] / X_test.shape[0]

        score = coalition_score(XY_train, coalition)
        if score > best_coalition_score:
            best_coalition = tuple(coalition)
            best_coalition_score = score
        while len(coalition) < len(labels):
            parties = np.zeros(len(labels))
            for party1 in coalition:
                for party2 in labels - set(coalition):
                    parties[party2] += prob_matrix[party1][party2] + prob_matrix[party2][party1]
            coalition.append(np.argmax(parties))

            score = coalition_score(XY_train, coalition)
            if score > best_coalition_score:
                best_coalition = tuple(coalition)
                best_coalition_score = score

    return best_coalition,best_coalition_score


def calc_prob_matrix(XY_train,XY_trainVal, model):
    X_train, y_train = split_label_from_data(XY_train)
    model.fit(X_train,y_train)

    X_train, y_train = split_label_from_data(XY_trainVal)

    labels = set(y_train)
    label_sample = dict()
    for label in labels:
        label_sample[label] = X_train[y_train == label]

    prob_matrix = np.zeros((len(labels), len(labels)))

    for label1 in labels:
        pred_prob = model.predict_proba(label_sample[label1])
        for label2 in labels:
            prob_matrix[label1][label2] = sum(pred_prob[:,label2])
        prob_matrix[label1] = prob_matrix[label1] / len(label_sample[label1])

    plt.imshow(prob_matrix)
    plt.colorbar()
    plt.show()

    return prob_matrix


def find_best_coalition_generative(XY_train,XY_val,XY_test):

    classifiers_list = [GaussianNB(), LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis()]

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
    prob_matrix = calc_prob_matrix(XY_train, XY_trainVal,best_model)

    coalition ,score = find_possible_coalitions_generative(XY_trainVal,XY_test,prob_matrix)

    return sorted(list(coalition))



# dist  [3, 4, 5, 6, 8, 9, 11, 12]


def main():
    XY_train = pd.read_csv('train_transformed.csv', index_col=0, header=0)
    XY_val = pd.read_csv('val_transformed.csv', index_col=0, header=0)
    XY_test = pd.read_csv('test_transformed.csv', index_col=0, header=0)

    # Finding best coalition using clustering models
    #clustring_coalition = find_best_coalition_cluster(XY_train, XY_val)

    # Finding best generative using clustering models
    #generative_coalition = find_best_coalition_generative(XY_train, XY_val, XY_test)

    #leading_features_for_each_party(pd.concat([XY_train, XY_val]))

    tree = DecisionTreeClassifier(random_state=0, min_samples_split=3)
    x_train, y_train = split_label_from_data(pd.concat([XY_train, XY_val]))
    x_test, y_test = split_label_from_data(XY_test)
    tree.fit(x_train, y_train)
    export_graph_tree(tree)



if __name__ == '__main__':
    main()
