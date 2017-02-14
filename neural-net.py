import pandas as pd
import numpy as np
import random

def create_layer(numNeurons, numInputs):
    return [[random.uniform(-1.0, 1.0) for x in range(numInputs + 1)] for x in range(numNeurons)]
def calc_threshold(input_vector, input_weights):
    return sum([x * y for x, y in zip([-1] + input_vector, input_weights)])
def calc_activation(threshold):
    return 1/(1 + 2.71828**-threshold)
def calc_error_output(activation, target):
    return activation * (1 - activation) * (activation - target)
def calc_error_hidden(activation, weights, error):
    return activation(1 - activation) * sum (map(lambda x: x*error, weights))
def calc_weight(current_weight, learning_rate, error, activation):
    return current_weight - (learning_rate * error * activation)
def create_classifier(numInputs, *layers):
    layers_list = list(layers)
    layer = layers_list.pop(0)
    if len(layers_list) == 0:
        return [create_layer(layer, numInputs)]
    else:
        return [create_layer(layer, numInputs)] + create_classifier(layer, *layers_list)
def classify(inputVector, imm_neural_net):
    neural_net = list(imm_neural_net)
    current_output = map(lambda neuron_weights:calc_activation(calc_threshold(inputVector, neuron_weights)), neural_net.pop(0)) 
    if len(neural_net) == 0:
        return current_output
    else:
        return classify(current_output, neural_net)
def accuracy(output, target):
    truePositive = 0
    falsePositive = 0
    for i in range(len(output)):
        if output[i] == target[i]:
            truePositive = truePositive + 1
        else:
            falsePositive = falsePositive + 1
    return float(truePositive) / len(output)


df = pd.read_csv("iris.data", header = None, names = ["sepal_length",
                                                            "sepal_width",
                                                            "petal_length",
                                                            "petal_width", "class"])
df_input = df.drop("class", axis=1)
df_norm = (df_input - df_input.mean()) / (df_input.max() - df_input.min())
classifier = create_classifier(4, 4, 3)
output = map(np.argmax, [classify(row, classifier) for index, row in df_norm.iterrows()])
classes = list(set(df["class"]))
print accuracy(output, [classes.index(x) for x in df["class"]])

df = pd.read_csv("diabetes.data", header = None)
df_input = df.drop(list(df)[-1], axis=1)
df_norm = (df_input - df_input.mean()) / (df_input.max() - df_input.min())
df_norm
