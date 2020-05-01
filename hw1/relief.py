
import pandas as pd
import numpy as np
import math


def distance(x1, x2, len):
    return 10
    val = 0
    for i in range(1,len):
        val += (x1.iloc[i] - x2.iloc[i])**2
    return math.sqrt(val)

def relief(data:pd.DataFrame, iterations, threshold):

    features = data.columns.values
    weights_array = np.zeros(features.shape[0])

    while iterations > 0:
        random_instance = data.sample()
        new_data = data.drop(random_instance.index)
        random_label = random_instance.iloc[0][0]
        df_miss = new_data[new_data['Vote'] != random_label]
        df_hit = new_data[new_data['Vote'] == random_label]
        a=distance(random_instance.iloc[0],df_miss.iloc[2],features.shape[0])
        miss = df_miss.apply(lambda row: distance(row,random_instance.iloc[0],features.shape[0]),axis=1)#.idxmin()
        hit = df_hit.apply(lambda row: distance(row,random_instance.iloc[0],features.shape[0]),axis=1)#.idxmin()

        for i in range(1,features.shape[0]):
            weights_array[i] += (random_instance.iloc[0][i]-df_miss.iloc[miss][i])**2 -(random_instance.iloc[0][i]-df_hit.iloc[hit][i])**2

        iterations-=1

    return features[weights_array > threshold]


