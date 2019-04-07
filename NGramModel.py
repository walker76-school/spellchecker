import nltk
from collections import Counter
from nltk import ngrams
from nltk.stem import *
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import brown, state_union, shakespeare, gutenberg, twitter_samples, PlaintextCorpusReader, reuters
import string


class NGramModel:

    def __init__(self, n):

        # Number of Grams
        self.numberGrams = n

        # The ngrams of the the new listing of words
        self.ngram_major = []

        # Dictionary of a number of occurrences for a particular gram
        self.ngram_major_counter = Counter()

        # get all the grams of smaller size
        self.ngram_minor = []

        # get the counter of the gramsOfSmaller
        self.ngram_minor_counter = Counter()

        # raw word counter
        self.raw_counter = Counter()

        self.tokenizer = RegexpTokenizer(r'\w+')

        self.wnl = WordNetLemmatizer()
        self.punctuations = '''’!()-[]{};:'"\,<>./?@#$%^&*_~'''

        self.gen_ngrams(brown)
        self.gen_ngrams(state_union)
        self.gen_ngrams(shakespeare)
        self.gen_ngrams(gutenberg)
        self.gen_ngrams(reuters)
        # self.gen_ngrams(PlaintextCorpusReader(fileids=["test.txt"], root="./"))

    def gen_ngrams(self, corpus):
        # raw_raw = corpus.words()
        # raw_raw = [e1.lower() for e1 in raw_raw]
        # raw = []
        # for e1 in raw_raw:
        #     valid = True
        #     for char in e1:
        #         if char in self.punctuations:
        #             valid = False
        #     if valid:
        #         raw.append(e1)
        #
        # self.raw_counter = self.raw_counter + Counter(raw)

        raw = word_tokenize(corpus.raw())
        raw = [e1.lower() for e1 in raw]

        # The ngrams of the the new listing of words
        self.ngram_major += ngrams(raw, self.numberGrams)

        # Dictionary of a number of occurrences for a particular gram
        self.ngram_major_counter = Counter(self.ngram_major)

        # get all the grams of smaller size
        self.ngram_minor += ngrams(raw, self.numberGrams - 1)

        # get the counter of the gramsOfSmaller
        self.ngram_minor_counter = Counter(self.ngram_minor)

    def freq(self, word):
        if word in self.raw_counter:
            return self.raw_counter[word]

        return -1

    def prob(self, l):
        # make sure the list l has same length of corpus trained on
        if len(l) != self.numberGrams:
            return -1

        # get the total amount of occurrences for that gram
        num_total_gram = 0

        listOfPossibleGrams = []
        for term in self.ngram_major_counter.keys():
            outerKey = ''.join(term[0]).lower()
            if outerKey == l[0]:
                listOfPossibleGrams.append(term)
                num_total_gram += self.ngram_major_counter[term]

        listOfTuples = []
        for tup in listOfPossibleGrams:
            prob = self.ngram_major_counter[tup] / num_total_gram
            listOfTuples.append((''.join(tup[1]).lower(), prob))

        # return the probability
        return listOfTuples

    def contains(self, word):
        return word in self.ngram_minor_counter
