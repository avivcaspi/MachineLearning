import pandas as pd
import numpy as np


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


# Outlier removing
def remove_outlier(data: pd.DataFrame, z_threshold) -> pd.DataFrame:
    numeric_feat = data.keys()[data.dtypes.map(lambda x: x == 'float64')]
    mean = data[numeric_feat].mean()
    std = data[numeric_feat].std()
    z_scores = (data[numeric_feat] - mean) / std
    data[numeric_feat] = data[numeric_feat].mask(abs(z_scores) > z_threshold)
    return data


# Imputation
# filling using median for numeric values and most common for nominal values
def imputation(data: pd.DataFrame) -> pd.DataFrame:
    # Median - numeric
    numeric_feat = data.keys()[data.dtypes.map(lambda x: x == 'float64')]
    data[numeric_feat] = data[numeric_feat].fillna(data[numeric_feat].dropna().median())

    # Most common - nominal
    nominal_feat = data.keys()[data.dtypes.map(lambda x: x == 'Int64')]
    data[nominal_feat] = data[nominal_feat].fillna(data[nominal_feat].dropna().mode().iloc[0])
    return data
