from data_preperation import *
from model_selection import *
from coalition_selection import *
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier


def main():
    XY_train = pd.read_csv('train_transformed.csv', index_col=0, header=0)
    XY_val = pd.read_csv('val_transformed.csv', index_col=0, header=0)
    XY_test = pd.read_csv('test_transformed.csv', index_col=0, header=0)
    df_pred = pd.read_csv('ElectionsData_Pred_Features_transformed.csv', index_col=0, header=0)
    X_train, y_train = split_label_from_data(XY_train)
    X_val, y_val = split_label_from_data(XY_val)
    X_test, y_test = split_label_from_data(XY_test)
    find_best_params = False
    if find_best_params:
        classifiers_params_dict = {RandomForestClassifier: {'n_estimators': list(range(100, 250, 40)),
                                                            'min_samples_split': list(range(2, 11, 4)),
                                                            'random_state': 2},
                                   SVC: {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']},
                                   MLPClassifier: {'hidden_layer_sizes': [(100, 50,), (50, 100,), (100, 200,)],
                                                   'max_iter': [1000]}}
        classifiers_params_dict = find_best_params_CV(XY_train, classifiers_params_dict)
    else:
        classifiers_params_dict = {RandomForestClassifier: {'n_estimators': 220,
                                                            'min_samples_split': 6,
                                                            'random_state': 2},
                                   SVC: {'kernel': 'rbf'},
                                   MLPClassifier: {'hidden_layer_sizes': (150, 10,),
                                                   'max_iter': 1000}}

    print(f'Classifiers best params : \n{classifiers_params_dict}')
    # TODO add voting model with 3 models

    results = train_and_evaluate(classifiers_params_dict, XY_train, XY_val)  # used to pick best model manually
    print(results)
    best_clf = pick_best_classifier(results)

    XY_train_new = pd.concat([XY_train, XY_val])
    clf = train_best_classifier(XY_train_new, best_clf, classifiers_params_dict[best_clf])

    """Model Evaluation using test set"""
    X_train_new, y_train_new = split_label_from_data(XY_train_new)

    X_test, y_test = split_label_from_data(XY_test)
    y_pred = clf.predict(X_test)
    y_train_pred = clf.predict(X_train_new)

    accuracy = accuracy_score(y_test, y_pred)
    train_accuracy = accuracy_score(y_train_new, y_train_pred)

    print(f'Test Error : {1 - accuracy},  Train Error : {1 - train_accuracy}')

    """Prediction on new set"""
    y_pred = clf.predict(df_pred)
    y_pred_names = np.array([classes[i] for i in y_pred])
    a = {'IdentityCard_Num': list(range(1, len(y_pred_names) + 1)), 'PredictVote': y_pred_names}
    df_final_prediction = pd.DataFrame(a)
    df_final_prediction.to_csv('final_predict.csv')
    division_of_voters = {classes[party]: list(y_pred).count(party) for party in range(13)}
    print(division_of_voters)
    party_with_majority = max(division_of_voters, key=division_of_voters.get)

    print(f'Party that will win majority of votes (in relation to new set) : {party_with_majority}')

    n_voters = len(df_pred)
    division_of_voters.update((key, round(value * 100 / n_voters, 3)) for key, value in division_of_voters.items())
    print(f'Division of voters : \n{division_of_voters}')
    bins = np.linspace(0, 12, 26)
    y_pred = [classes[y] for y in y_pred]
    plt.hist(y_pred, bins, label='Prediction')
    plt.xticks(range(0, 13, 1), rotation='vertical')
    plt.legend(loc='upper right')
    plt.title('Prediction voters')
    plt.show()

    explode = np.zeros(13)
    explode[classes.index(party_with_majority)] = 0.1
    plt.pie(division_of_voters.values(), explode=explode, labels=division_of_voters.keys(),
            autopct='%1.1f%%', shadow=True, startangle=0)
    plt.title('Division of voters')
    plt.axis('equal')

    plt.show()


if __name__ == '__main__':
    main()
