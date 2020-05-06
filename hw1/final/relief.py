
import pandas as pd
import numpy as np


def relief(data: pd.DataFrame, iterations, k):

    features = data.columns.values
    weights_array = np.zeros(features.shape[0])

    while iterations > 0:
        random_instance = data.sample()
        new_data = data.drop(random_instance.index)
        random_label = random_instance.iloc[0][0]
        df_miss = new_data[new_data['Vote'] != random_label]
        df_hit = new_data[new_data['Vote'] == random_label]
        min_miss = np.linalg.norm((df_miss.values[:, 1:] - random_instance.values[:, 1:]).astype(float), ord=2, axis=1).argmin()
        min_hit = np.linalg.norm((df_hit.values[:, 1:] - random_instance.values[:, 1:]).astype(float), ord=2, axis=1).argmin()
        for i in range(1, features.shape[0]):
            a = (random_instance.iloc[0][i]-df_miss.iloc[min_miss][i])**2
            b = (random_instance.iloc[0][i]-df_hit.iloc[min_hit][i])**2
            weights_array[i] += a-b
        iterations -= 1

    return features[weights_array.argsort()[:-k:-1]]


