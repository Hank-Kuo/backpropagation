from Backpropagation import NeuralNetwork
import numpy as np
import pandas as pd 


if __name__ == "__main__":
    testing_data = pd.read_csv('iris_testing_data.txt', sep=' ', names=[
                                "sepal_length", "sepal_width", "petal_length","petal_width", "class"])

    training_data = pd.read_csv('iris_training_data.txt', sep=' ', names=[
                                "sepal_length", "sepal_width", "petal_length","petal_width", "class"])

    label = ["setosa","versicolor","virginica"]

    for i in np.arange(0.1, 1.1, 0.1):
        for j in np.arange(1, 10, 1):
            bpn = NeuralNetwork(
                layers=[j,3],
                lr=i,
                epochs=100,
                components=["sepal_length", "sepal_width", "petal_length","petal_width"],
                label = label)

            print("------train ------------")
            print("lr = ",i)
            print("Hidden layer neuron = ", j)
            bpn.train(training_data)
            print("Train_accuracy = %r %% " %(bpn.acc))
            print("epochs = %r " %(bpn.epochs_count))
            print("------test ------------")
            bpn.predict_batch(testing_data)
            print("Test_accuracy  = %r %% " %(bpn.acc))

    