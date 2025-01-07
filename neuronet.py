import numpy as np
class NeuralNetwork:
    def __init__(self, input, labels, layers, normalization, ratio = 0.8, random_seed=72):
        np.random.seed(random_seed)

        self.layers = layers
    
        self.input = input
        self.labels = labels

        self.bias = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.normal(0, np.sqrt(2 / y), size=(y, x))
                        for x, y in zip(layers[:-1], layers[1:])]
        
        self.activation = None
        self.weighted_sum = None


        num_train = int(len(input) * ratio)

        #training data - 80 percent default
        self.train_X = self.input[:num_train]
        self.train_y = self.labels[:num_train]
        self.train_X = self.train_X / normalization
        
        #testing data - 20 percenet default
        self.test_X = self.input[num_train:]
        self.test_y = self.labels[num_train:]
        self.test_X = self.test_X / normalization

        if not (0 < ratio < 1):
            raise ValueError("Ratio must be between 0 and 1.")

        if len(self.train_X) == 0 or len(self.test_X) == 0:
            raise ValueError("Insufficient data for training or testing. Check the ratio or dataset size.")

    def train(self, batch_size=5, epochs=30, learning_rate=0.012):
        for _ in range(epochs):
            average_weights = [np.zeros_like(w) for w in self.weights]
            average_bias = [np.zeros_like(b) for b in self.bias]
            average_cost = 0
            train_X, train_y = self.randomize_training_data()
            #len(train_X)
            for image in range(len(train_X)):
                errors = []
 
                #intial forward prop
                output = self.forward(train_X[image])
                average_cost += self.cost(output, train_y[image])
                
                #calculate inital error for output layer
                label_matrix = np.zeros((self.layers[-1], 1))
                label_matrix[train_y[image], 0] = 1  
                z_L = (self.weights[-1] @ self.activation[-2]) + self.bias[-1]
                errors.append((output - label_matrix) * self.derivative_relu(z_L))
                
                #caluclate error for all other layers
                errors = self.calculate_errors(errors)
                #update the changes in weights
                activation_index = len(self.activation) - 2
                for i in range(len(errors) - 1, -1, -1):
                    
                    average_weights[i] += errors[i] @ self.activation[activation_index].T
                    average_bias[i] += errors[i]
                    activation_index -= 1


                #apply gradient per batch size
                if(image + 1) % batch_size == 0:
                    for i in range(len(average_weights)):
                        average_weights[i] /= batch_size
                        average_bias[i] /= batch_size

                    self.apply_gradient(average_weights, average_bias, learning_rate)
                    average_weights = [np.zeros_like(w) for w in self.weights]
                    average_bias = [np.zeros_like(b) for b in self.bias]  
            
            
            self.evaluate(_)
            #print(f"Cost Per Epoch: {average_cost/len(train_X)}")

  
    def calculate_errors(self, errors):
        index = len(self.activation)-3
        for i in range(len(self.weights)-2, -1, -1):
            z_L = (self.weights[i] @ self.weighted_sum[index]) + self.bias[i]
            error = (self.weights[i+1].T @ errors[-1]) * (self.derivative_relu(z_L))
            errors.append(error)
            index -= 1
           
        
        return errors[::-1]


    
    #does the matrix multiplication, taking the aN*nW, takes input images and calculate output based on weights
    def forward(self, image):
        activations = [image.flatten().reshape(-1, 1)]
        weighted_sum = [image.flatten().reshape(-1, 1)]
        for weight, bias in zip(self.weights, self.bias):
            z_L = (weight @ activations[-1]) + bias
            activations.append(self.relu(z_L))
            weighted_sum.append(z_L)
        
        self.activation = activations
        self.weighted_sum = weighted_sum
        return self.activation[-1]


    def randomize_training_data(self):
        indices = np.random.permutation(len(self.train_X))
        train_X = self.train_X[indices]
        train_y = self.train_y[indices]
        return(train_X, train_y)
    

    def cost(self, output, image_label):  
        label = self.train_y[image_label]
        label_matrix = np.zeros((self.layers[-1], 1))
        label_matrix[label, 0] = 1
        
        cost = np.mean(np.square(output - label_matrix))
        return cost
    
    #not using for now
    def softmax(self, res):
        exp_res = np.exp(res - np.max(res))  # for numerical stability
        output = exp_res / np.sum(exp_res)  # softmax probabilities
        return output
    
    def derivative_relu(self, x, alpha=0.01):
        grad = np.ones_like(x)
        grad[x <= 0] = alpha
        return grad
    
    def relu(self, x, alpha=0.01):
        return np.maximum(alpha * x, x)

    #applies gradient to networks weight and bias
    def apply_gradient(self, average_weights, average_bias, learning_rate):
      for i in range(len(self.weights)):
          self.weights[i] -= (learning_rate * average_weights[i])
          self.bias[i] -= (learning_rate * average_bias[i])

    #use test data to evaluate model performance
    def evaluate(self, _):
        correct = 0
        for image in range(len(self.test_X)):
            output = self.forward(self.test_X[image])
            ret = np.argmax(output)
            if ret == self.test_y[image]:
                correct += 1
        print(f"Epoch {_+1}: {correct}/{len(self.test_X)} - Accuracy: {(correct/len(self.test_X))*100:.2f}") 
        return correct/len(self.test_X)