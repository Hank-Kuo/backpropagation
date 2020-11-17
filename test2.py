import numpy as np
import pandas as pd




date_class = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

testing_data = pd.read_csv('iris_testing_data.txt', sep=' ', names=[
                                "sepal_length", "sepal_width", "petal_length","petal_width"])

training_data = pd.read_csv('iris_training_data.txt', sep=' ', names=[
                                "sepal_length", "sepal_width", "petal_length","petal_width", "class"])

label = ["setosa","versicolor","virginica"]

class NeuralNetwork:
    def __init__(self, layers, lr, epochs, components,label):
        self.layers = layers
        self.Weight = []
        self.Bias = []
        self.lr = lr 
        self.epochs = epochs
        self.components = components
        self.label = label
        self.target = np.identity(self.layers[-1])
        self.acc = 0 
        
    
    def initWeight(self):
        for i in range(len(self.layers)-1):
            weight = np.random.rand(self.layers[i+1],self.layers[i])
            self.Weight.append(weight)

    def initBais(self):
        for i in range(len(self.layers)-1):
            bias = np.random.rand(self.layers[i+1],1)
            self.Bias.append(bias)
    
    def transfer_target(self, data_target):
        position = self.label.index(data_target)
        return np.array([self.target[position]]).T
    
    def activation_fn(self,net_input):
        output = 1 / (1 + np.exp(net_input))
        return output
    
    def accuracies(self, data_target, output, data_length):
        target = self.transfer_target(data_target)
        output_target = np.where(output > 0.5 , 1 , 0)
        error_target = target - output_target
        length = data_length[0]
        for i in error_target:
            if (i.sum() == 0):
                self.acc += 1
        self.acc = self.acc / length * 100.0
        print(self.acc)

    
    # output neuron 
    def granddecent_output(self, data_target, output):
        one = np.ones(np.shape(output))
        target = self.transfer_target(data_target)
        output_error = ( target - output )* (output*(one-output))
        return output_error
    
    def granddecent_hidden(self,weight,error,output):
        one = np.ones(np.shape(output))
        output_error = np.dot(error.T,weight).T*(output*(one-output))
        return output_error
    
    def updateWeight(self,weight,output,error):
        weight = weight + 2 * self.lr * np.dot(error ,output.T )
        return weight
    
    def updateBias(self,bias,error):
        bias = bias + 2 * self.lr * error
        return bias


    def feedforward(self, weight,bias,origin_data):
        net_input = np.array(np.dot(weight,origin_data),dtype=np.float32)+ bias
        output = self.activation_fn(net_input)
        return output

    
    def train(self, train_data):
        output_ = []
        for i, data in train_data.iterrows():
            output_matrix = []
            input_data = np.array([data[self.components].values]).T
            output_matrix.append(input_data)
            
            ## feedforward
            for layer in range(len(self.layers)-1):
                output = self.feedforward(self.Weight[layer],self.Bias[layer],output_matrix[layer])
                output_matrix.append(output)
            

            error_matrix = []
            ## Calculate the error
            for layer in range(len(self.layers)-1,0,-1):
                if(layer == len(self.layers)-1):
                    # output error 
                    error = self.granddecent_output(data["class"],output_matrix[layer])
                    error_matrix.append(error)
                else:
                    error = self.granddecent_hidden(self.Weight[layer],error_matrix[-1],output_matrix[layer])
                    error_matrix.append(error)
            
            # update Weight and Bias 
            index = 0 
            for layer in range(len(self.layers)-1,0,-1):
                weight = self.updateWeight(self.Weight[layer-1],output_matrix[layer-1],error_matrix[index])
                self.Weight[layer-1] = weight 
                bias = self.updateBias(self.Bias[layer-1],error_matrix[index])
                self.Bias[layer-1] = bias
                index+=1
            
            output_ += output_matrix[-1]
            if(i==0):
                break
        self.accuracies(data["class"],output_,training_data.shape)

    def predict(self,data):
        output_matrix = []
        output_matrix.append(data.T)
        for layer in range(len(self.layers)-1):
            output = self.feedforward(self.Weight[layer],self.Bias[layer],output_matrix[layer])
            output_matrix.append(output) 
        return output_matrix[-1] 




a = NeuralNetwork(
    layers=[4,1,3],
    lr=1,
    epochs=50,
    components=["sepal_length", "sepal_width", "petal_length","petal_width"],
    label = label)

a.initWeight()

a.initBais()

print(a.Weight)
print(a.Bias)

a.train(training_data)
print("-------new weigtht and bias -----------")
print(a.Weight)
print(a.Bias)



print(a.acc)

# print(a.predict(np.array([[1,1,1,1]])))
