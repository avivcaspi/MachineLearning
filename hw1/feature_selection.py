import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier

def selectKBest_features_selection(x: pd.DataFrame, y: pd.DataFrame, k):
    selector = SelectKBest(mutual_info_classif, k=k)
    selector.fit(x, y)
    features = selector.get_support(indices=True)
    return x.columns.values[features]


def sklearn_ExtraTree_feature_selection(data_X, data_Y, k):
    clf = ExtraTreesClassifier(n_estimators=50)
    forest = clf.fit(data_X, data_Y)
    X = data_X.values
    y = data_Y.values
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()
    return indices[:k]


# Returns indices of features selected
def SFS(model, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series):
    features_name = x_train.columns
    x_train = x_train.values
    y_train = y_train.values
    x_test = x_test.values
    y_test = y_test.values
    features = []
    features_left = list(range(x_train.shape[1]))
    best_accuracy = 0
    best_feature = [0]  # random initialization

    while len(best_feature) and len(features_left) > 0:
        # Clone the model to avoid using fitted model again
        test_model = clone(model, safe=True)
        best_feature = []
        # going over all features left and finding best one
        for feature in features_left:
            current_features = sorted([*features, feature])
            # take only current features from the data
            x_train_converted = x_train[:, current_features]
            x_test_converted = x_test[:, current_features]
            # fit and predict
            test_model.fit(x_train_converted, y_train)
            y_pred = test_model.predict(x_test_converted)
            accuracy = metrics.accuracy_score(y_test, y_pred)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_feature = [feature]
            elif accuracy == best_accuracy:  # if there are multiple features with same accuracy we save them
                best_feature.append(feature)
        # In case no improvement best_feature will be empty
        if len(best_feature):
            chosen_feature = np.random.choice(best_feature)
            features.append(chosen_feature)
            features_left.remove(chosen_feature)
        print(features)


    chosen_features = [features_name[index] for index in sorted(features)]
    return chosen_features

