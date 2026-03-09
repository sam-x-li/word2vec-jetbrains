import numpy as np 
from model import Word2Vec
from numpy.linalg import norm

def cosineSim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def most_similar(model, word, top_n=5):
    if word not in model.wordToIndex:
        return []

    idx = model.wordToIndex[word]
    vec = model.W_in[idx]

    similarities = []
    for i, w in model.indexToWord.items():
        if w == word:
            continue
        v = model.W_in[i]
        sim = cosineSim(vec, v)
        similarities.append((w, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

def main():
    model = Word2Vec.load("word2vec_model.pkl")
    examples = ["alice", "queen", "king", "rabbit", "white"]

    for word in examples:
        print(f"Top similar words to '{word}':")
        print(most_similar(model, word, top_n=5))
        print()

if __name__ == '__main__':
    main()