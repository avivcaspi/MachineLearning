from main import *
from sklearn.metrics import *


def automate_model_selection():
    XY_train = pd.read_csv('train_transformed.csv', index_col=0, header=0)
    XY_val = pd.read_csv('val_transformed.csv', index_col=0, header=0)

    classifiers_params_dict = {RandomForestClassifier: {'n_estimators': list(range(60, 80, 20)),
                                                            'min_samples_split': list(range(2, 4, 2)),
                                                            'random_state': 2},
                                   KNeighborsClassifier: {'n_neighbors': list(range(3, 6))}}

    classifiers_params_dict = find_best_params_CV(XY_train, classifiers_params_dict)

    classifiers_win_counter = {RandomForestClassifier: 0, KNeighborsClassifier: 0}

    score_func_dict = {accuracy_score: 1, balanced_accuracy_score: 0.8, cohen_kappa_score: 0.7, matthews_corrcoef: 0.8}

    # check for each score function which of the classifier get best score
    for score_func,weight in score_func_dict.items():
        results = train_and_evaluate(classifiers_params_dict, XY_train, XY_val,score_function=score_func)
        best_clf = max(results, key=results.get)
        classifiers_win_counter[best_clf] += weight

    # takes classifier with the most wins
    best_clf = max(classifiers_win_counter, key=classifiers_win_counter.get)
    print(f'Best classifier is : {best_clf.__name__}')
    return best_clf

automate_model_selection()
