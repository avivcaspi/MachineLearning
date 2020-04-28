import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# split the data into train(70%), test(15%), val(15%)
def split_data(df: pd.DataFrame, test_size=0.15, val_size=0.15):
    X = df.loc[:, df.columns != 'Vote']
    y = df['Vote']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size/(1-test_size), stratify=y_train)
    return X_train.sort_index(), y_train.sort_index().astype(int), X_val.sort_index(), y_val.sort_index().astype(int)\
        , X_test.sort_index(), y_test.sort_index().astype(int)


def split_label_from_data(df: pd.DataFrame) -> tuple:
    assert 'Vote' in df.columns
    x = df.loc[:, df.columns != 'Vote']
    y = df['Vote']
    return x, y.astype(int)


def insert_label_to_data(df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, 'Vote', y)
    return df


# convert all nominal unordered features to onehot
def convert_to_onehot(data: pd.DataFrame, onehot_nominal_features) -> pd.DataFrame:
    data_with_dummies = data
    for feature in onehot_nominal_features:
        dummies = pd.get_dummies(data_with_dummies[feature], prefix=feature)
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


def change_binary_values(df: pd.DataFrame, binary_features) -> pd.DataFrame:
    for feature in binary_features:
        df.loc[df[feature] == 0, feature] = -1
    return df


# TODO Check if maybe we should change the values to negative instead of removing them
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

        self.median = xy_train[self.numeric_feat].median(skipna=True)
        self.mode = xy_train[self.nominal_feat].mode(dropna=True).iloc[0]

    # filling missing values in train set by sampling from the set
    @staticmethod
    def impute_train(xy_train: pd.DataFrame) -> pd.DataFrame:
        assert 'Vote' in xy_train.columns  # assert label is in data frame
        for label in xy_train['Vote'].unique():
            for feature in xy_train.columns:
                missing = xy_train.loc[xy_train['Vote'] == label, feature].loc[xy_train.loc[xy_train['Vote'] == label, feature].isnull()]
                nnan = xy_train.loc[xy_train['Vote'] == label, feature].isnull().sum()
                values = np.random.choice(xy_train.loc[xy_train['Vote'] == label, feature].dropna(), nnan)
                missing.loc[:] = values

                xy_train.loc[xy_train['Vote'] == label, feature] = xy_train.loc[xy_train['Vote'] == label, feature].fillna(missing)

        return xy_train

    # filling missing values in test and val sets by using median and most frequent from train set
    def impute_test_val(self, x_test: pd.DataFrame, x_val: pd.DataFrame) -> tuple:
        x_test[self.numeric_feat] = x_test[self.numeric_feat].fillna(self.median)
        x_test[self.nominal_feat] = x_test[self.nominal_feat].fillna(self.mode)

        x_val[self.numeric_feat] = x_val[self.numeric_feat].fillna(self.median)
        x_val[self.nominal_feat] = x_val[self.nominal_feat].fillna(self.mode)
        return x_test, x_val


class Scaling:
    def __init__(self, xy_train: pd.DataFrame):
        self.min = xy_train.min(axis=0)
        self.max = xy_train.max(axis=0)
        self.mean = xy_train.mean(axis=0)
        self.std = xy_train.std(axis=0)

    def scale_min_max(self, df: pd.DataFrame, features, min_value, max_value) -> pd.DataFrame:
        df[features] = (df[features] - self.min[features]) / (self.max[features] - self.min[features]) * (max_value - min_value) + min_value
        return df

    def normalization(self, df: pd.DataFrame, features):
        df[features] = (df[features] - self.mean[features]) / self.std[features]
        return df
