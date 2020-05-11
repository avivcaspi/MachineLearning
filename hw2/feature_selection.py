import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


