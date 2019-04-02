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

    # paragraphs, are, papers, many, students
    wrong = checker.check("Pargraphs ar the building blocks of ppers. Muny stedents define paragraphs in terms of length: a paragraph is a group of at least five sentences, a paragraph is half a page long, etc. In reality, though, the unity and coherence of ideas among sentences is what constitutes a paragraph. Length and appearance do not determine whether a section in a paper is a paragraph. For instance, in some styles of writing, particularly journalistic styles, a paragraph can be just one sentence long")
    print(wrong)
