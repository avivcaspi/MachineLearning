from data_preperation import *


selected_features = ['Yearly_IncomeK', 'Number_of_differnt_parties_voted_for', 'Political_interest_Total_Score',
                     'Avg_Satisfaction_with_previous_vote', 'Avg_monthly_income_all_years', 'Most_Important_Issue',
                     'Overall_happiness_score', 'Avg_size_per_room', 'Weighted_education_rank']
onehot_nominal_features = ['Most_Important_Issue']
features_with_negative = ['Avg_monthly_income_all_years']

uniform_features = ['Yearly_IncomeK']
normal_features = ['Number_of_differnt_parties_voted_for', 'Weighted_education_rank', 'Overall_happiness_score',
                   'Avg_size_per_room', 'Avg_Satisfaction_with_previous_vote', 'Avg_monthly_income_all_years',
                   'Political_interest_Total_Score']

discrete_features = ['Number_of_differnt_parties_voted_for']
continuous_features = ['Yearly_IncomeK', 'Avg_size_per_room', 'Avg_Satisfaction_with_previous_vote',
                       'Weighted_education_rank', 'Avg_monthly_income_all_years', 'Political_interest_Total_Score',
                       'Overall_happiness_score']
numerical_features = discrete_features + continuous_features

if __name__ == '__main__':
    load = True
    if load:
        XY_train = pd.read_csv('train_transformed.csv', index_col=0, header=0)
        XY_val = pd.read_csv('val_transformed.csv', index_col=0, header=0)
        XY_test = pd.read_csv('test_transformed.csv', index_col=0, header=0)

    else:
        # Load data
        df = pd.read_csv('ElectionsData.csv', header=0)
        df = df[['Vote'] + selected_features]

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
        imputation = Imputation(XY_train, numerical_features, [])
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

    X_train, y_train = split_label_from_data(XY_train)
    X_val, y_val = split_label_from_data(XY_val)
    X_test, y_test = split_label_from_data(XY_test)



