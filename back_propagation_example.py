from numpy import *
  
class NeuralNetwork(object): 
    def __init__(self): 
        random.seed(1) 
        self.synaptic_weights = 2 * random.random((3, 1)) - 1
        
    def __sigmoid(self, x): 
        return 1 / (1 + exp(-x))
      
    def __sigmoid_derivative(self, x): 
        return x * (1 - x) 
      
    def train(self, inputs, outputs, training_iterations): 
        for iteration in range(training_iterations): 
            output = self.learn(inputs) 
            error = outputs - output 
            factor = dot(inputs.T, error * self.__sigmoid_derivative(output)) 
            self.synaptic_weights += factor 
            
    def learn(self, inputs): 
        return self.__sigmoid(dot(inputs, self.synaptic_weights)) 
  
if __name__ == "__main__": 
    neuralnet = NeuralNetwork() 
    inputs = array([[0, 0, 1], [1, 0, 1], [1, 0, 0]]) 
    outputs = array([[0, 1, 1]]).T 
    neuralnet.train(inputs, outputs, 10000) 
    xyz=neuralnet.learn(array([1, 0, 1]))
    print(xyz)
