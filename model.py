import numpy as np
from collections import Counter 
import random, pickle

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

class Word2Vec:

    '''
        General method:
        1. Get one-hot encoding of all the words
        2. Intialise window 
        
        ---forward---
        3. Select pair from window
        4. Sample k negatives
        3. Multiply by W_in, to get embedding for centre word
        4. Multiply by part of W_out, to get scores (only using pair and negatives)
        6. Calculate loss
        -------------

        7. Backprop, update sampled words
        8. Slide window by 1, repeat
    '''

    def __init__(self, d: int = 50, r: int = 5, k: int = 5, lr: float = 0.02):
        self.corpus = []
        self.wordToIndex = {}
        self.indexToWord = {}
        self.wordToFreq = {}

        self.d = d # embedding size
        self.r = r # window size
        self.lr = lr # learning rate 
        self.k = k # number of negative sample

        self.unigramDist = np.array([])

        self.V = 0 #number of unique words
        self.W_in = self.initialiseWeights() #centre word embeddings (V, d)
        self.W_out = self.initialiseWeights() #context word embeddings (V, d)

    def setup(self, corpus):
        self.setCorpus(corpus)
        self.setOneHotEncoding()
        self.setFreqAndDist()
        self.subSampleCorpus()

        self.V = len(self.wordToIndex)
        self.W_in = self.initialiseWeights()
        self.W_out = self.initialiseWeights()

        self.indexCorpus = self.indexWords(self.corpus) 

    def setCorpus(self, corpus: str):
        if isinstance(corpus, str):
            self.corpus = corpus.lower().split()
        else:
            self.corpus = [word.lower() for word in corpus]

    #takes in corpus, removes copies, returns mappings to indices
    def setOneHotEncoding(self):
        newWords = sorted(set(self.corpus))
        self.wordToIndex = {w: i for i, w in enumerate(newWords)}
        self.indexToWord = {i: w for i, w in enumerate(newWords)}
    
    def setFreqAndDist(self):
        counts = Counter(self.corpus)
        total = len(self.corpus)

        freqs = np.array([counts[w] for w in counts]) ** 0.75
        self.unigramDist = freqs / freqs.sum()
        self.wordToFreq = {w: counts[w] / total for w in counts}

    def subSampleCorpus(self, t=1e-3):
        newCorpus = []

        for word in self.corpus:
            freq = self.wordToFreq[word]
            discardProb = 1 - np.sqrt(t / freq)
            if random.random() > discardProb: 
                newCorpus.append(word)
        
        print(f"Original corpus: {len(self.corpus)}\nNew corpus: {len(newCorpus)}")
        self.corpus = newCorpus
    
    #converts list of words into np array of indices 
    def indexWords(self, words: list) -> np.array:
        return np.array(
            [self.wordToIndex[word] for word in words],
            dtype=np.int32
            )
    
    #initialises weights randomly, for a (V x d) matrix
    def initialiseWeights(self) -> np.array:
        return np.random.rand(self.V, self.d)

    #saves parameters at filepath
    def save(self, filepath):
        data = {
            "W_in": self.W_in,
            "W_out": self.W_out,
            "wordToIndex": self.wordToIndex,
            "indexToWord": self.indexToWord,
            "d": self.d,
            "r": self.r,
            "lr": self.lr,
            "k": self.k
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        model = cls(d=data['d'], r=data['r'], lr=data['lr'], k=data['k'])

        model.W_in = data['W_in']
        model.W_out = data['W_out']
        model.wordToIndex = data['wordToIndex']
        model.indexToWord = data['indexToWord']
        model.V = len(model.wordToIndex)

        print(f"Model loaded from {filepath}")
        return model
    
    def sampleUnigram(self):
        return np.random.choice(self.V, p=self.unigramDist)

    #returns list of indices, corresponding to randomly sampled negatives
    def sampleNegatives(self, targetIndex):
        negatives = []
        while len(negatives) < self.k:
            sample = self.sampleUnigram()
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
    
    def loss(self, probDistVector):
        positive = probDistVector[0]
        negative = probDistVector[1:]

        result = -np.log(positive + 1e-10)
        result -= np.sum(np.log(1 - negative)) #algebraic trick with sigmoid
        
        return result
    
    def calculateGradients(self, probDistVector: np.array, targetEncoding: np.array, centreVector: np.array, contextVectors: np.array):
        error = probDistVector - targetEncoding
        gradOut = np.outer(error, centreVector)
        gradIn = error @ contextVectors
        return (gradOut, gradIn)
    
    def updateMatrices(self, sampled, centreIndex, gradOut, gradIn):
        for i, wordIndex in enumerate(sampled):
            self.W_out[wordIndex] -= self.lr * gradOut[i]
        self.W_in[centreIndex] -= self.lr * gradIn
    
    def backprop(self, centreIndex: int, centreVector: np.array, sampled: list, probDistVector: np.array):
        targetEncoding = np.zeros(len(sampled))
        targetEncoding[0] = 1 

        contextVectors = self.W_out[sampled]

        gradOut, gradIn = self.calculateGradients(probDistVector, targetEncoding, centreVector, contextVectors)

        self.updateMatrices(sampled, centreIndex, gradOut, gradIn)
    
    def trainPair(self, centreIndex, targetIndex):
        centreVector, sampled, probDistVector = self.forward(centreIndex, targetIndex)
        loss = self.loss(probDistVector)
        self.backprop(centreIndex, centreVector, sampled, probDistVector)
        return loss

    #single epoch
    def trainingPass(self):
        r = self.r 
        netLoss = 0
        for i in range(r, len(self.indexCorpus) - r):
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
