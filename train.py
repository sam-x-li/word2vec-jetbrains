import numpy as np
import random, time


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

class Word2Vec:

    def __init__(self, words, d):
        self.wordToIndex, self.indexToWord = self.getOneHotEncoding(words)
        self.d = d 

    def getOneHotEncoding(self, words: str) -> dict:
        if words is str:
            newWords = set(words.lower().split())
        elif words is list: 
            newWords = set([word.lower() for word in words])
        wordToIndex = {w: i for i, w in enumerate(newWords)}
        indexToWord = {i: w for i, w in enumerate(newWords)}
        return (wordToIndex, indexToWord)
    
    def indexWords(self, words: list) -> np.array:
        return np.array([self.oneHotEncoding[word] for word in words])
    
    def initialiseWeights
    
    #pairs of indices, derived from the word pair
    def generatePairs(self, words: list) -> list(int, int):
        pass

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
    
def main():
    pass

if __name__ == '__main__':
    main()