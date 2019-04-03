import nltk
from nltk.corpus import words
from nltk.tokenize import word_tokenize
from nltk.stem import *
from NGramModel import NGramModel
from nltk.corpus import brown, state_union
from SpellChecker import SpellChecker

if __name__ == "__main__":

    # nltk.download("brown")
    # nltk.download("state_union")
    # nltk.download("words")
    # nltk.download('punkt')
    # nltk.download('wordnet')

    checker = SpellChecker()

    # paragraphs, are, papers, many, students
    wrong = checker.check("This has already been debunked by lingustic professors, maybe someone has the link. It doesn’t seem to be very reliable, especially since it doesn’t evn sem to be from Cambridge. Check this article out at Cambridge. It has even been labeled an urban legend by some.")

    print(wrong)
