from data_preperation import *
from feature_selection import *
from test import test_accuracy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from relief import *

pd.options.mode.chained_assignment = None

features_with_negative = ['Avg_monthly_expense_when_under_age_21', 'Avg_lottary_expanses', 'Avg_monthly_income_all_years']
onehot_nominal_features = ['Most_Important_Issue', 'Main_transportation', 'Occupation']
binary_features = ['Gender', 'Looking_at_poles_results', 'Married', 'Financial_agenda_matters', 'Voting_Time']
nominal_features = ['Will_vote_only_large_party', 'Age_group']

discrete_features = ['Occupation_Satisfaction', 'Last_school_grades', 'Number_of_differnt_parties_voted_for', 'Number_of_valued_Kneset_members', 'Num_of_kids_born_last_10_years']
continuous_features = ['Avg_monthly_expense_when_under_age_21', 'Avg_lottary_expanses', 'Avg_monthly_expense_on_pets_or_plants', 'Avg_environmental_importance',
                      'Financial_balance_score_(0-1)', '%Of_Household_Income', 'Yearly_IncomeK', 'Avg_size_per_room', 'Garden_sqr_meter_per_person_in_residancy_area', 'Avg_Residancy_Altitude',
                      'Yearly_ExpensesK', '%Time_invested_in_work', 'Avg_education_importance', 'Avg_Satisfaction_with_previous_vote',
                      'Avg_monthly_household_cost', 'Phone_minutes_10_years', 'Avg_government_satisfaction', 'Weighted_education_rank', '%_satisfaction_financial_policy',
                      'Avg_monthly_income_all_years', 'Political_interest_Total_Score', 'Overall_happiness_score']

uniform_features = ['Occupation_Satisfaction', 'Financial_balance_score_(0-1)',
                    '%Of_Household_Income', 'Yearly_IncomeK', 'Avg_government_satisfaction',
                    '%_satisfaction_financial_policy', 'Garden_sqr_meter_per_person_in_residancy_area',
                    'Yearly_ExpensesK', '%Time_invested_in_work']
normal_features = ['Number_of_differnt_parties_voted_for', 'Number_of_valued_Kneset_members',
                   'Avg_environmental_importance',
                   'Avg_education_importance', 'Avg_monthly_household_cost', 'Weighted_education_rank',
                   'Overall_happiness_score', 'Avg_size_per_room', 'Avg_Residancy_Altitude',
                   'Last_school_grades', 'Num_of_kids_born_last_10_years',
                   'Avg_monthly_expense_when_under_age_21', 'Avg_lottary_expanses',
                   'Avg_monthly_expense_on_pets_or_plants', 'Avg_Satisfaction_with_previous_vote',
                   'Phone_minutes_10_years', 'Avg_monthly_income_all_years', 'Political_interest_Total_Score']

numerical_features = discrete_features + continuous_features
total_nominal_features = nominal_features + binary_features

if __name__ == '__main__':
    run_feature_selection = False  # if load is False this will run feature selection as well as data transformation
    load = True     # Will load the transformed csv and won`t run feature selection

    if load:
        XY_train = pd.read_csv('train_transformed.csv', index_col=0, header=0)
        XY_val = pd.read_csv('val_transformed.csv', index_col=0, header=0)
        XY_test = pd.read_csv('test_transformed.csv', index_col=0, header=0)

    else:
        # Load data
        df = pd.read_csv('ElectionsData.csv', header=0)

        # Split data
        train_indices, val_indices, test_indices = split_data(df, test_size=0.15, val_size=0.15)
        df.loc[train_indices, :].to_csv('train_original.csv')
        df.loc[val_indices, :].to_csv('val_original.csv')
        df.loc[test_indices, :].to_csv('test_original.csv')

        # Convert nominal features to onehot
        df = convert_to_onehot(df, onehot_nominal_features)

        # Convert to categorical type
        df = convert_to_categorical(df)

        # Change negative values to positive
        df = abs_negative(df, features_with_negative)

        # Change binary values to -1 1 instead of 0 1
        df = change_binary_values(df, binary_features)

        # Fix ternary nominal features to -1 0 1
        df.loc[df['Will_vote_only_large_party'] == 1, 'Will_vote_only_large_party'] = -1
        df.loc[df['Will_vote_only_large_party'] == 2, 'Will_vote_only_large_party'] = 1
        df.loc[df['Age_group'] == 2, 'Age_group'] = -1

        # Create sets
        XY_train = df.loc[train_indices, :]
        XY_val = df.loc[val_indices, :]
        XY_test = df.loc[test_indices, :]

        # Remove outlier from train set
        print('Outlier removing')
        print(f'Number of nan before: {XY_train.isnull().values.sum()}')
        outlier = Outlier(XY_train, numerical_features)
        XY_train = outlier.remove_outlier(XY_train, 4.5)
        print(f'Number of nan after: {XY_train.isnull().values.sum()}')

        # Imputation
        print('Imputation')
        print(f'Number of nan before: {XY_train.isnull().values.sum()}')
        imputation = Imputation(XY_train, numerical_features, total_nominal_features)
        XY_train = imputation.impute_train(XY_train)
        print(f'Number of nan after: {XY_train.isnull().values.sum()}')

        print(f'Number of nan before: {XY_test.isnull().values.sum() + XY_val.isnull().values.sum()}')
        XY_test, XY_val = imputation.impute_test_val(XY_test, XY_val)
        print(f'Number of nan after: {XY_test.isnull().values.sum() + XY_val.isnull().values.sum()}')

        # Scaling
        XY_train, XY_val, XY_test = scale_all(XY_train, XY_val, XY_test, uniform_features, normal_features)

        XY_train.to_csv('train_transformed.csv')
        XY_val.to_csv('val_transformed.csv')
        XY_test.to_csv('test_transformed.csv')

    # feature selection
    X_train, y_train = split_label_from_data(XY_train)
    X_val, y_val = split_label_from_data(XY_val)
    X_test, y_test = split_label_from_data(XY_test)
    if run_feature_selection and not load:  # This will take long time to run
        # filter methods
        # ----------selectKBest------------
        # We found that 18 gives the best result
        features = selectKBest_features_selection(X_train, y_train, 18)
        # print(features)
        # print(test_accuracy(X_train[features], y_train, X_val[features], y_val))
        selectkbest_features = features

        # ----------relief------------
        # We run the it without the one hot features
        features = relief(XY_train.iloc[:, :35], 2000, 18)
        # print(features)
        # print(test_accuracy(X_train[features], y_train, X_val[features], y_val))
        relief_features = features

        # wrapper methods
        # ----------RFE with forest-------------
        estimator = RandomForestClassifier(random_state=101)
        selector = RFE(estimator, 18, step=1)
        selector = selector.fit(X_train, y_train)
        selected_features = selector.get_support(indices=True)
        # print(f'Features number: {18}, features: {selected_features}')
        # print(test_accuracy(X_train.iloc[:, selected_features], y_train, X_val.iloc[:, selected_features], y_val))
        # print(X_train.columns[selected_features])
        rfe_features = X_train.columns[selected_features]

        # ----------Features importance with ExtraTree------------
        features = sklearn_ExtraTree_feature_selection(X_train, y_train, 18)
        # print(X_train.columns.values[sorted(features)])
        # print(test_accuracy(X_train.iloc[:, features], y_train, X_val.iloc[:, features], y_val))
        features = list(X_train.columns.values[features])
        extratree_features = features

        # intersection
        intersection = set(rfe_features) & set(extratree_features) & set(selectkbest_features) & set(relief_features)
        union = set(rfe_features) | set(extratree_features) | set(selectkbest_features) | set(relief_features)
        inter_acc = test_accuracy(X_train[intersection], y_train, X_val[intersection], y_val)
        union_acc = test_accuracy(X_train[union], y_train, X_val[union], y_val)

    # final features we chose after evaluating the union/ intersection/ and each feature selection method results
    final_features = ['Yearly_IncomeK', 'Last_school_grades', 'Avg_education_importance', 'Avg_monthly_expense_on_pets_or_plants', 'Avg_Residancy_Altitude', 'Most_Important_Issue_Military', 'Avg_Satisfaction_with_previous_vote', 'Avg_size_per_room', 'Number_of_differnt_parties_voted_for', 'Avg_monthly_household_cost', 'Phone_minutes_10_years', 'Most_Important_Issue_Other', 'Avg_environmental_importance', 'Political_interest_Total_Score', 'Overall_happiness_score']
    print(test_accuracy(X_train[final_features], y_train, X_val[final_features], y_val))
    # XY_train[['Vote'] + final_features].to_csv('train_transformed.csv')
    # XY_val[['Vote'] + final_features].to_csv('val_transformed.csv')
    # XY_test[['Vote'] + final_features].to_csv('test_transformed.csv')





