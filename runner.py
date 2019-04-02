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

    wrong = checker.check("This is a misspell word united statez apples aples")
    print(wrong)
