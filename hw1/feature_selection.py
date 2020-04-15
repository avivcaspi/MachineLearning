import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.base import clone
import random

"""
Returns indices of features selected
"""
def SFS(model, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
    features = []
    features_left = list(range(x_train.shape[1]))
    best_accuracy = 0
    best_feature = 0  # random initialization

    while best_feature is not None or len(features_left) == 0:
        test_model = clone(model, safe=True)
        best_feature = None
        # going over all features left and finding best one
        for feature in features_left:
            current_features = sorted([*features, feature])
            x_train_converted = x_train[:, current_features]
            x_test_converted = x_test[:, current_features]
            test_model.fit(x_train_converted, y_train)
            y_pred = test_model.predict(x_test_converted)
            accuracy = metrics.accuracy_score(y_test, y_pred)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_feature = feature
            elif accuracy == best_accuracy:
                if isinstance(best_feature, list):
                    best_feature.append(feature)
                elif best_feature is None:
                    best_feature = feature
                else:
                    best_feature = [feature, best_feature]
        if best_feature is not None:
            if isinstance(best_feature, list):
                best_feature = random.choice(best_feature)
            features.append(best_feature)
            features_left.remove(best_feature)
    return sorted(features)

