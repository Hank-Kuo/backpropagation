import numpy as np
import pandas as pd


class NeuralNetwork:
    def __init__(self, layers, lr, epochs, components,label):
        self.layers = layers
        self.layers.insert(0,len(components))
        self.lr = lr 
        self.epochs = epochs
        self.components = components
        self.label = label
        self.setup()
    
    def setup(self):
        self.setupVariable()
        self.targetToList()

    def targetToList(self):
        self.target = np.identity(self.layers[-1])

    def setupVariable(self):
        self.acc = 0
        self.acc_count = 0 
        self.epochs_count = 0 
        self.Weight1 = np.random.rand(self.layers[1],self.layers[0])
        self.Weight2 = np.random.rand(self.layers[2],self.layers[1])
        self.Bias1 = np.random.rand(self.layers[1],1)
        self.Bias2 = np.random.rand(self.layers[2],1)
    
    def transfer_target(self, data_target):
        position = self.label.index(data_target)
        return np.array([self.target[position]]).T
    
    def activation_fn(self,net_input):
        output = 1/(1 + np.exp(-net_input))
        return output
    
    def accuracies(self, data_target, output,index,length):
        target = self.transfer_target(data_target)
        output_target = np.where(output > 0.5 , 1 , 0)
        error_target = (target - output_target).T
        for i in error_target:
            if (i.sum() == 0):
                self.acc_count += 1
        if(index == length-1):
            self.acc = self.acc_count / length *100
            self.acc_count = 0 

    
    # output neuron 
    def backend(self, data_target, output,input_data):
        errors = []
        target = self.transfer_target(data_target)
        output_errors = (target - output[1] ) * (output[1]*(1-output[1]))
        hidden_errors = np.dot(output_errors.T, self.Weight2).T*(output[0]*(1-output[0]))
        errors.append(hidden_errors)
        errors.append(output_errors)

        self.Weight2 = self.Weight2 + self.lr * 2 * np.dot(output_errors,output[0].T)
        self.Bias2 = self.Bias2 + self.lr * 2 * output_errors
        self.Weight1 = self.Weight1+self.lr * 2 * np.dot(hidden_errors,input_data.T)
        self.Bias1 = self.Bias1 + self.lr * 2 * hidden_errors
        return self

    def feedforward(self, data):
        output = []
        hidden_net_input = np.dot(self.Weight1,data) + self.Bias1
        hidden_output = self.activation_fn(hidden_net_input)
        output.append(hidden_output)
        output_net_input = np.dot(self.Weight2, hidden_output) + self.Bias2
        output_output = self.activation_fn(output_net_input)
        output.append(output_output)
        return output

    
    def train(self, train_data):
        for j in range(self.epochs):
            for i, data in train_data.iterrows():
                input_data = np.array([data[self.components].values],dtype=float).T
                
                # feedforward 
                output = self.feedforward(input_data)
                # Backend 
                self.backend(data["class"],output,input_data)
                
                self.accuracies(data["class"],output[1],i,len(train_data))
            
            self.epochs_count+=1
            if(self.acc >= 99):
                break

    
    def predict(self,data):
        data = data.T
        output_matrix = self.feedforward(data)
        output = np.where(output_matrix[-1] > 0.5 , 1 , 0)
        output_class = "No"
        for i in range(len(output)):
            if(output[i]==1):
                output_class = self.label[i]
                break
        return output_class
    
    def predict_batch(self,test_data):
        self.acc = 0 
        for i, data in test_data.iterrows():
            input_data = np.array([data[self.components].values],dtype=float).T
            # feedforward 
            output_matrix = self.feedforward(input_data)
            output = np.where(output_matrix[-1] > 0.5 , 1 , 0)
            output_class = "No"
            for j in range(len(output)):
                if(output[j]==1):
                    output_class = self.label[j]
                    # print(output_class)
                    self.accuracies(data["class"],output_matrix[-1], i, len(test_data))
                    break


                

        

    




