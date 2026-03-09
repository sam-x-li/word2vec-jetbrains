from model import Word2Vec
import numpy as np 
from numpy.linalg import norm
import nltk, re
nltk.download("gutenberg")
from nltk.corpus import gutenberg

def cosine_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def most_similar(model, word, top_n=5):
    if word not in model.wordToIndex:
        return []

    idx = model.wordToIndex[word]
    vec = model.W_in[idx]  # or (W_in + W_out)/2

    similarities = []
    for i, w in model.indexToWord.items():
        if w == word:
            continue
        v = model.W_in[i]
        sim = cosine_sim(vec, v)
        similarities.append((w, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]


def main():
    rawText = gutenberg.raw("shakespeare-hamlet.txt")
    cleanedText = re.sub(r'[^a-zA-Z\s]', '', rawText.lower())

    corpus = cleanedText.split()

    model = Word2Vec()
    model.setup(corpus)
    model.train(50)

    model.save("word2vec_model.pkl")

    examples = ["hamlet", "king", "queen", "ghost", "lord"]

    for word in examples:
        print(f"Top similar words to '{word}':")
        print(most_similar(model, word, top_n=5))
        print()

if __name__ == '__main__':
    main()