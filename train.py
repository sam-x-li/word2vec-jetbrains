from model import Word2Vec
import nltk, re
from nltk.corpus import gutenberg

nltk.download("gutenberg")

epochs = 30

def main():
    rawText = gutenberg.raw("carroll-alice.txt")
    cleanedText = re.sub(r'[^a-zA-Z\s]', '', rawText.lower())

    corpus = cleanedText

    model = Word2Vec()
    model.setup(corpus)
    model.train(epochs)

    model.save("word2vec_model.pkl")

if __name__ == '__main__':
    main()