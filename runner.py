import nltk
from nltk.corpus import words
from nltk.tokenize import word_tokenize
from nltk.stem import *
from NGramModel import NGramModel
from nltk.corpus import brown, state_union
from SpellChecker import SpellChecker

if __name__ == "__main__":

    nltk.download("brown")
    nltk.download("state_union")
    nltk.download("words")
    nltk.download('punkt')
    nltk.download('wordnet')

    checker = SpellChecker()

    wrong = checker.check("The szn did not shene. It was too weet to play. Sio we sat in the house All that cold, cold, wet day. I sat there with Sally. We sat there, we two. And I said,  How I wish We had something to do.")

    print(wrong)
