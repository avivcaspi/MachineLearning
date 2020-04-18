import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.base import clone


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

    chosen_features = [features_name[index] for index in sorted(features)]
    return chosen_features

