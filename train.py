import numpy as np
import random, time
from math import log


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(v: np.array):
    expVector = np.exp(v - np.max(v)) #safe softmax
    return expVector / np.sum(expVector)

class Word2Vec:

    '''
        General method:
        1. Get one-hot encoding of all the words
        2. Intialise window 
        
        ---forward---
        3. Multiply by W_in, to get embedding for centre word
        4. Multiply by W_out, to get score for all words in vocab
        5. Use softmax to get pd vector
        6. Calculate loss
        -------------

        7. Backprop
        8. Slide window by 1, repeat
        Do this first, then implement neg sampling. 
    '''

    def __init__(self, words, d: int = 5, r: int = 2, lr: float = 0.01):
        self.words = words.lower().split()
        self.wordToIndex, self.indexToWord = self.getOneHotEncoding()
        self.V = len(self.wordToIndex) #number of unique words
        self.d = d #number of context dimensions
        self.r = r #window size
        self.lr = lr #learning rate 
        self.W_in = self.initialiseWeights() #centre word embeddings (V, d)
        self.W_out = self.initialiseWeights() #context word embeddings (V, d)

    #takes in corpus, removes copies, returns mappings to indices
    def getOneHotEncoding(self) -> dict:
        if isinstance(self.words, str):
            newWords = sorted(set(self.words.lower().split()))
        elif isinstance(self.words, list): 
            newWords = sorted(set([word.lower() for word in self.words]))
            #wordlists are sorted for consistency
        wordToIndex = {w: i for i, w in enumerate(newWords)}
        indexToWord = {i: w for i, w in enumerate(newWords)}
        return (wordToIndex, indexToWord)

    #converts list of words into np array of indices 
    def indexWords(self, words: list) -> np.array:
        return np.array([self.wordToIndex[word] for word in words])
    
    #initialises weights randomly, for a (V x d) matrix
    def initialiseWeights(self) -> np.array:
        return np.random.rand(self.V, self.d)

    #saves weights of matrices in a file
    def storeModel(self):
        pass

    def forward(self, centreIndex: int) -> (np.array, np.array):
        centreContext = self.W_in[centreIndex] #one-hot encoding, which is equivalent to a lookup
        scores = self.W_out @ centreContext    
        probDistVector = softmax(scores)
        return (centreContext, probDistVector)
    
    def crossEntropyLoss(self, pd, targetIndex):
        #target is one-hot, so all other terms are 0
        return -log(pd[targetIndex])
    
    #abstracted out as this will change when swapping to negative sampling
    def calculateGradients(self, probDistVector: np.array, targetEncoding: np.array, centreContext: np.array):
        error = probDistVector - targetEncoding 
        gradOut = np.outer(error, centreContext)
        gradIn = self.W_out.T @ error
        return (gradOut, gradIn)
    
    def updateMatrices(self, centreIndex, gradOut, gradIn, lr):
        self.W_out -= lr * gradOut
        self.W_in[centreIndex] -= self.lr * gradIn
    
    def backprop(self, centreIndex: int, targetIndex: int, centreContext: np.array, probDistVector: np.array):
        targetEncoding = np.zeros(self.V)
        targetEncoding[targetIndex] = 1 

        gradOut, gradIn = self.calculateGradients(probDistVector, targetEncoding, centreContext)

        self.updateMatrices(centreIndex, gradOut, gradIn, self.lr)

        loss = self.crossEntropyLoss(probDistVector, targetIndex)

        return loss
    
    #single epoch
    def trainingPass(self):
        r = self.r 
        netLoss = 0
        indexVector = self.indexWords(self.words)
        for i in range(r, self.V - r):
            window = indexVector[i-r:i+r+1]
            centreIndex = window[r]
            centreContext, probDistVector = self.forward(centreIndex)
            for j, targetIndex in enumerate(window):
                if j == r:
                    continue
                netLoss += self.backprop(centreIndex, targetIndex, centreContext, probDistVector)
        return netLoss
    


def main():
    corpus = "the quick brown fox jumps over the lazy dog"
    model = Word2Vec(corpus)
    for epochs in range(25):
        print(f"Loss: {model.trainingPass()}")

    for word, idx in model.wordToIndex.items():
        print(word, model.W_in[idx])

if __name__ == '__main__':
    main()