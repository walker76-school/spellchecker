import nltk
from nltk.corpus import words
from nltk.tokenize import word_tokenize
from nltk.stem import *
from NGramModel import NGramModel
from nltk.corpus import brown, state_union


class SpellChecker:

    def __init__(self):
        self.allWords = set(words.words())
        self.snowball = SnowballStemmer("english")
        self.porter = PorterStemmer()
        self.ngram = NGramModel(state_union, 2, 100)

    def check(self, words):
        if not isinstance(words, str):
            raise Exception("check() takes a string")

        misspelled = []
        count = 0
        for word in word_tokenize(words):
            if word.lower() in self.allWords:
                count += 1
                continue

            if self.snowball.stem(word.lower()) in self.allWords:
                count += 1
                continue

            if self.porter.stem(word.lower()) in self.allWords:
                count += 1
                continue

            misspelled += (count, word, [])
            count += 1

        return misspelled
