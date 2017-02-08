import pandas as pd
import numpy as np
import random

def create_layer(numNeurons, numInputs):
    return [[random.uniform(-1.0, 1.0) for x in range(numInputs + 1)] for x in range(numNeurons)]
def calc_threshold(input_vector, input_weights):
    return sum([x * y for x, y in zip([-1] + input_vector, input_weights)]) > 0
input_vector = [2, 3, 4]
print map(lambda x: calc_threshold(input_vector, x), create_layer(len(input_vector), 3))

df = pd.read_csv("iris.data", header = None, names = ["sepal_length",
                                                            "sepal_width",
                                                            "petal_length",
                                                            "petal_width", "class"])
df_input = df.drop("class", axis=1)
df_norm = (df_input - df_input.mean()) / (df_input.max() - df_input.min())
print df_norm

df = pd.read_csv("diabetes.data", header = None)
df_input = df.drop(list(df)[-1], axis=1)
df_norm = (df_input - df_input.mean()) / (df_input.max() - df_input.min())
print df_norm
