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

    def __init__(self, d: int = 50, r: int = 5, k: int = 5, lr: float = 0.02):
        self.corpus = []
        self.wordToIndex = {}
        self.indexToWord = {}

        self.d = d # embedding size
        self.r = r # window size
        self.lr = lr # learning rate 
        self.k = k # number of negative samples

        self.V = 0 #number of unique words
        self.W_in = self.initialiseWeights() #centre word embeddings (V, d)
        self.W_out = self.initialiseWeights() #context word embeddings (V, d)

    def setup(self, corpus):
        self.setCorpus(corpus)
        self.setOneHotEncoding()

        self.V = len(self.wordToIndex)
        self.W_in = self.initialiseWeights()
        self.W_out = self.initialiseWeights()

        self.indexCorpus = self.indexWords(self.corpus) 

    def setCorpus(self, corpus: str):
        self.corpus = corpus.lower().split()

    #takes in corpus, removes copies, returns mappings to indices
    def setOneHotEncoding(self):
        newWords = sorted(set(self.corpus))
        self.wordToIndex = {w: i for i, w in enumerate(newWords)}
        self.indexToWord = {i: w for i, w in enumerate(newWords)}

    #converts list of words into np array of indices 
    def indexWords(self, words: list) -> np.array:
        return np.array(
            [self.wordToIndex[word] for word in words],
            dtype=np.int32
            )
    
    #initialises weights randomly, for a (V x d) matrix
    def initialiseWeights(self) -> np.array:
        return np.random.rand(self.V, self.d)

    #saves weights of matrices in a file
    def storeModel(self):
        pass

    #returns list of indices, corresponding to randomly sampled negatives
    def sampleNegatives(self, targetIndex):
        negatives = []
        while len(negatives) < self.k:
            sample = np.random.randint(0, self.V)
            if sample != targetIndex:
                negatives.append(sample)
        return negatives

    def forward(self, centreIndex: int, targetIndex: int):
        centreVector = self.W_in[centreIndex] #one-hot encoding, which is equivalent to a lookup
        negatives = self.sampleNegatives(targetIndex)
        sampled = [targetIndex] + negatives
        scores = self.W_out[sampled] @ centreVector    
        probDistVector = sigmoid(scores)

        return centreVector, sampled, probDistVector

    def forward1(self, centreIndex: int) -> (np.array, np.array):
        centreVector = self.W_in[centreIndex] #one-hot encoding, which is equivalent to a lookup
        scores = self.W_out @ centreVector    
        probDistVector = softmax(scores)
        return (centreVector, probDistVector)
    
    def crossEntropyLoss(self, pd, targetIndex):
        #target is one-hot, so all other terms are 0
        return -log(pd[targetIndex])
    
    def loss(self, probDistVector):
        positive = probDistVector[0]
        negative = probDistVector[1:]

        loss = -np.log(positive + 1e-10)
        loss -= np.sum(np.log(1 - negative)) #algebraic trick with sigmoid
        
        return loss 
    
    #abstracted out as this will change when swapping to negative sampling
    def calculateGradients1(self, probDistVector: np.array, targetEncoding: np.array, centreVector: np.array):
        error = probDistVector - targetEncoding 
        gradOut = np.outer(error, centreVector)
        gradIn = self.W_out.T @ error
        return (gradOut, gradIn)
    
    def calculateGradients(self, probDistVector: np.array, targetEncoding: np.array, centreVector: np.array, contextVectors: np.array):
        error = probDistVector - targetEncoding
        gradOut = np.outer(error, centreVector)
        gradIn = error @ contextVectors
        return (gradOut, gradIn)
    
    def updateMatrices(self, sampled, centreIndex, gradOut, gradIn):
        for i, wordIndex in enumerate(sampled):
            self.W_out[wordIndex] -= self.lr * gradOut[i]
        self.W_in[centreIndex] -= self.lr * gradIn
    
    def backprop(self, centreIndex: int, targetIndex: int, centreVector: np.array, sampled: list, probDistVector: np.array):
        targetEncoding = np.zeros(len(sampled))
        targetEncoding[0] = 1 

        contextVectors = self.W_out[sampled]

        gradOut, gradIn = self.calculateGradients(probDistVector, targetEncoding, centreVector, contextVectors)

        self.updateMatrices(sampled, centreIndex, gradOut, gradIn)
    
    def trainPair(self, centreIndex, targetIndex):
        centreVector, sampled, probDistVector = self.forward(centreIndex, targetIndex)
        loss = self.loss(probDistVector)
        self.backprop(centreIndex, targetIndex, centreVector, sampled, probDistVector)
        return loss

    #single epoch
    def trainingPass(self):
        r = self.r 
        netLoss = 0
        for i in range(r, self.V - r):
            window = self.indexCorpus[i-r:i+r+1]
            centreIndex = window[r]

            for j, targetIndex in enumerate(window):
                if j == r:
                    continue
                netLoss += self.trainPair(centreIndex, targetIndex)

        return netLoss
    
    def train(self, epochs = 100):
        for epoch in range(epochs):
            loss = self.trainingPass()  
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")


def main():

    corpus = """
In a village of La Mancha, the name of which I have no desire to call to mind, 
there lived not long since one of those gentlemen that keep a lance in the lance-rack, 
an old buckler, a lean hack, and a greyhound for coursing. 
An olla of rather more beef than mutton, a salad on most nights, scraps on Saturdays, 
lentils on Fridays, and a pigeon or so extra on Sundays, made away with three-quarters of his income.
"""

    model = Word2Vec()
    model.setup(corpus)
    model.train()

if __name__ == '__main__':
    main()