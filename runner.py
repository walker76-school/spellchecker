import nltk
from SpellChecker import SpellChecker

if __name__ == "__main__":

    nltk.download("brown")
    nltk.download("state_union")
    nltk.download("words")
    nltk.download('punkt')

    checker = SpellChecker()

    checker.check("This is a misspell word united statez")



