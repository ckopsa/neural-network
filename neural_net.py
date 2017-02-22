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
    return activation * (1 - activation) * sum(map(lambda x,y:x*y, weights, error)) 
def calc_weight(current_weight, learning_rate, error, activation):
    new_weight = current_weight - (learning_rate * error * activation)
    #print current_weight, new_weight
    return new_weight
def create_classifier(numInputs, *layers):
    layer = layers[0]
    if len(layers) == 1:
        return [create_layer(layer, numInputs)]
    else:
        return [create_layer(layer, numInputs)] + create_classifier(layer, *layers[1:])
def train(inputVector, targetVector, neural_net, learningRate):
    j_activation = map(lambda neuron_weights:
                         calc_activation(calc_threshold(inputVector, neuron_weights)),
                         neural_net[0])
    if len(neural_net) == 1:
        j_error = map(calc_error_output, j_activation, targetVector)
        ij_weights = [map(lambda x:map(lambda y:calc_weight(y[0], learningRate, x[1], y[1]), zip(x[0], [-1] + inputVector)), zip(neural_net[0], j_error))]
        return ij_weights, j_error
    else:
        jk_weights, k_error = train(j_activation, targetVector, neural_net[1:], learningRate)
        j_error = map(lambda x,y:calc_error_hidden(x, y, k_error), j_activation, zip(*jk_weights[0])[1:])
        inputVector =  list(inputVector)
        ij_weights = [map(lambda x:map(lambda y:calc_weight(y[0], learningRate, x[1], y[1]), zip(x[0], [-1] + inputVector)), zip(neural_net[0], j_error))]
        return ij_weights + jk_weights, j_error
def classify(inputVector, neural_net):
    j_activation = map(lambda neuron_weights:
                         calc_activation(calc_threshold(inputVector, neuron_weights)),
                         neural_net[0])
    if len(neural_net) == 1:
        return j_activation
    else:
        return classify(j_activation, neural_net[1:])
def accuracy(output, target):
    truePositive = 0
    falsePositive = 0
    for i in range(len(output)):
        if output[i] == target[i]:
            truePositive = truePositive + 1
        else:
            falsePositive = falsePositive + 1
    return float(truePositive) / len(output)
def outputize(num, length):
    output = [0] * length
    output[num-1] = 1
    return output

test_input = [-.9, 0.1, .03, .05]
test_weights = [-.3, .5, -.5, .8, -.6]
test_weights = [[test_weights] * 4] + [[test_weights] * 3]
print len(test_weights)
output = map(calc_activation, map(lambda x:calc_threshold(test_input, x),test_weights[1]))
print output
errors = map(calc_error_output, output, [0, 0, 1])
print calc_weight(test_weights[1][0][1], 5, errors[0], output[0]), test_weights[1][0][1]
print calc_weight(test_weights[1][0][3], 5, errors[2], output[2]), test_weights[1][0][3]
output = map(calc_activation, map(lambda x:calc_threshold(test_input, x),test_weights[0]))
print len(output), len(test_weights[1]), len(errors)
#print calc_error_hidden(output,test_weights[1][0], errors)
df = pd.read_csv("iris.data", header = None, names = ["sepal_length",
                                                      "sepal_width",
                                                      "petal_length",
                                                      "petal_width", "class"])
df = df.sample(frac=1)
df_input = df.drop("class", axis=1)
df_norm = (df_input - df_input.mean()) / (df_input.max() - df_input.min())

classifier = create_classifier(4, 4, 3)
classes = list(set(df["class"]))
targetVector = [classes.index(x) for x in df["class"]]
output = map(np.argmax, [classify(row, classifier) for index, row in df_norm.iterrows()])
for i in range(100):
    #for index, row in df_norm.iterrows():
    classifier = train(df_norm.loc[0], outputize(1, 3), classifier, .1)[:-1][0]
        #classifier = train(row, outputize(targetVector[index], 3), classifier, .10)[:-1][0]
output = map(np.argmax, [classify(row, classifier) for index, row in df_norm.iterrows()])
accuracy(output, targetVector)

df = pd.read_csv("diabetes.data", header = None)
df_input = df.drop(list(df)[-1], axis=1)
df_norm = (df_input - df_input.mean()) / (df_input.max() - df_input.min())
df_norm
