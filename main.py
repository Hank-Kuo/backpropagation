from Backpropagation import NeuralNetwork
import numpy as np
import pandas as pd 


if __name__ == "__main__":
    testing_data = pd.read_csv('iris_testing_data.txt', sep=' ', names=[
                                "sepal_length", "sepal_width", "petal_length","petal_width", "class"])

    training_data = pd.read_csv('iris_training_data.txt', sep=' ', names=[
                                "sepal_length", "sepal_width", "petal_length","petal_width", "class"])

    label = ["setosa","versicolor","virginica"]

    
    bpn_1 = NeuralNetwork(
        layers=[1,3],
        lr=1,
        epochs=100,
        components=["sepal_length", "sepal_width", "petal_length","petal_width"],
        label = label)

    print("Number of hidden neurons = 1")
    print("Learning rates = 1")
    bpn_1.train(training_data)
    print("training accuracies = %r %% " %(bpn_1.acc ))
    bpn_1.predict_batch(testing_data)
    print("testing  accuracie = %r %% " %(bpn_1.acc))
    print("epochs = %r" %(bpn_1.epochs_count))
    
    print("--------------------------------------------")
    bpn_2 = NeuralNetwork(
        layers=[1,3],
        lr=1,
        epochs=100,
        components=["sepal_length", "sepal_width", "petal_length","petal_width"],
        label = label)

    print("Number of hidden neurons = 2")
    print("Learning rates = 1")
    bpn_2.train(training_data)
    print("training accuracies = %r %% " %(bpn_2.acc ))
    bpn_2.predict_batch(testing_data)
    print("testing  accuracie = %r %% " %(bpn_2.acc))
    print("epochs = %r " %(bpn_2.epochs_count))

    