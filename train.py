from model import Word2Vec
import nltk, re
nltk.download("gutenberg")
from nltk.corpus import gutenberg

def main():
    rawText = gutenberg.raw("carroll-alice.txt")
    cleanedText = re.sub(r'[^a-zA-Z\s]', '', rawText.lower())

    corpus = cleanedText

    model = Word2Vec()
    model.setup(corpus)
    model.train(30)

    model.save("word2vec_model.pkl")

if __name__ == '__main__':
    main()