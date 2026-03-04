import numpy as np
import random, time

'''
    ### DNN Tester ###

    Trains the 4-layer DNN on differently sized, randomly sampled training sets. Prints accuracy and time taken for each run.

'''

training_data, validation_data, test_data = mnistloader.load_data_wrapper()
random.shuffle(training_data)
random.shuffle(test_data)

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

class Network:

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(neurons, 1) for neurons in sizes[1:]]
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]
        
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.matmul(w, a)+b)
        return a
    
    def evaluate(self, test_data):
        results = [(np.argmax(self.feedforward(x)), y) for x, y in test_data] 
        return sum(int(x == y) for x, y in results)
    
    def stoch_grad(self, training_data, epochs, minibatch_size, learning_rate, test_data=None):
        n = len(training_data)
        for i in range(epochs):
            random.shuffle(training_data)
            minibatches = [training_data[k:k+minibatch_size] for k in range(0, n, minibatch_size)]
            for minibatch in minibatches:
                self.update(minibatch, learning_rate)
            
            if test_data:
                if i == epochs - 1:
                    total = self.evaluate(test_data)
                    l = len(test_data)
                    print(f'Epoch {i}: {total} / {l} --> {round(total * 100 / l, 5)}')
                    return (total, round(total * 100/l, 5))
                else:
                    print(f'Epoch {i} complete')

            else:
                print(f'Epoch {i} complete')

    
    def update(self, minibatch, learning_rate):
        bias_sum = [np.zeros(b.shape) for b in self.biases]
        weight_sum = [np.zeros(w.shape) for w in self.weights]

        for x, y in minibatch:
            partial_bias, partial_weight = self.backpropagation(x, y)
            bias_sum = [b + pb for b, pb in zip(bias_sum, partial_bias)]
            weight_sum = [w + pw for w, pw in zip(weight_sum, partial_weight)]

        self.weights = [w - (learning_rate/len(minibatch)) * sw for w, sw in zip(self.weights, weight_sum)]
        self.biases = [b - (learning_rate/len(minibatch)) * sb for b, sb in zip(self.biases, bias_sum)]
    
    def get_activation(self, input_x):
        activation = input_x
        weightedinput = []
        activations = [input_x]
        for b, w in zip(self.biases, self.weights):
            z = np.matmul(w, activation) + b
            weightedinput.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        return weightedinput, activations
            
    def backpropagation(self, input_x, y):
        partial_bias = [np.zeros(b.shape) for b in self.biases]
        partial_weight = [np.zeros(w.shape) for w in self.weights]

        weightedinput, activation = self.get_activation(input_x)

        delta = np.multiply(activation[-1] - y, sigmoid_prime(weightedinput[-1]))
        partial_bias[-1] = delta
        partial_weight[-1] = np.matmul(delta, np.transpose(activation[-2]))

        for i in range(2, self.num_layers):
            wT = np.transpose(self.weights[-i+1])
            delta = np.multiply(np.matmul(wT, delta), sigmoid_prime(weightedinput[-i]))
            partial_bias[-i] = delta
            partial_weight[-i] = np.matmul(delta, np.transpose(activation[-i-1]))

        return partial_bias, partial_weight
    
def timedrun(training_size, network):
    print(f'Size: {training_size}')
    random.shuffle(training_data)
    random.shuffle(test_data)
    start = time.time()
    info = network.stoch_grad(training_data[:training_size], 30, 10, 3, test_data)
    time_taken = round(time.time() - start, 3)
    return training_size, *info, time_taken

def main():
    n = Network([784, 30, 30, 10])
    info = [timedrun(50, n), timedrun(150, n), timedrun(500, n), timedrun(1500, n), timedrun(5000, n), timedrun(15000, n), timedrun(50000, n)]
    
    for i in info:
        print(i)

if __name__ == '__main__':
    main()