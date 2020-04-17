import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# split the data into train(70%), test(15%), val(15%)
def split_data(df: pd.DataFrame, test_size=0.15, val_size=0.15):
    X = df.loc[:, df.columns != 'Vote']
    y = df['Vote']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size/(1-test_size), stratify=y_train)
    return X_train.sort_index(), y_train.sort_index(), X_val.sort_index(), y_val.sort_index(), X_test.sort_index(), y_test.sort_index()


# convert all nominal unordered features to onehot
def convert_to_onehot(data: pd.DataFrame, onehot_nominal_features) -> pd.DataFrame:
    data_with_dummies = data
    for feature in onehot_nominal_features:
        dummies = pd.get_dummies(data_with_dummies[feature])
        data_with_dummies = pd.concat([data_with_dummies, dummies], axis=1)
        data_with_dummies.drop([feature], axis=1, inplace=True)
    return data_with_dummies


# convert all string object to categorical  and than to integer
def convert_to_categorical(data: pd.DataFrame) -> pd.DataFrame:
    ObjFeat = data.keys()[data.dtypes.map(lambda x: x == 'object')]  # picks all features type object
    for f in ObjFeat:
        data[f] = data[f].astype("category")
        data[f+"Int"] = data[f].cat.rename_categories(range(data[f].nunique())).astype('Int64')
        data.loc[data[f].isnull(), f+"Int"] = np.nan  # fix NaN conversion
        data[f] = data[f+"Int"]
        data = data.drop(f+"Int", axis=1)  # remove temporary columns
    return data


# removing garbage values
def remove_negative(data: pd.DataFrame) -> pd.DataFrame:
    numeric_feat = data.keys()[data.dtypes.map(lambda x: x == 'float64')]
    data[numeric_feat] = data[numeric_feat].mask(data[numeric_feat] < 0)
    return data


class Outlier:
    def __init__(self, xy_train: pd.DataFrame):
        self.numeric_feat = xy_train.keys()[xy_train.dtypes.map(lambda x: x == 'float64')]
        self.mean = xy_train.groupby('Vote').mean()[self.numeric_feat]
        self.std = xy_train.groupby('Vote').std()[self.numeric_feat]

    # Outlier removing
    def remove_outlier(self, df: pd.DataFrame, z_threshold) -> pd.DataFrame:
        for label in df['Vote'].unique():
            z_scores = (df.loc[df['Vote'] == label, self.numeric_feat] - self.mean.loc[label]) / self.std.loc[label]
            df.loc[df['Vote'] == label, self.numeric_feat] = df.loc[df['Vote'] == label, self.numeric_feat].mask(abs(z_scores) > z_threshold)
        return df


class Imputation:
    def __init__(self, xy_train: pd.DataFrame):
        self.numeric_feat = xy_train.keys()[xy_train.dtypes.map(lambda x: x == 'float64')]
        self.nominal_feat = xy_train.keys()[xy_train.dtypes.map(lambda x: x == 'Int64')]

        # self.median = xy_train.loc[xy_train['Vote'].unique(), self.numeric_feat].dropna().median()
        # self.mode = xy_train[self.nominal_feat].dropna().mode().iloc[0]

    # filling using median for numeric values and most common for nominal values
    def impute_train(self, data: pd.DataFrame) -> pd.DataFrame:
        for label in data['Vote'].unique():
            for feature in data.columns:
                missing = data.loc[data['Vote'] == label, feature].loc[data.loc[data['Vote'] == label, feature].isnull()]
                nnan = data.loc[data['Vote'] == label, feature].isnull().sum()
                values = np.random.choice(data.loc[data['Vote'] == label, feature].dropna(), nnan)
                missing.loc[:] = values

                data.loc[data['Vote'] == label, feature] = data.loc[data['Vote'] == label, feature].fillna(missing)

        return data
