from collections import Counter
from nltk import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import brown, state_union, shakespeare


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

        self.gen_ngrams(brown)
        self.gen_ngrams(state_union)
        self.gen_ngrams(shakespeare)

    def gen_ngrams(self, corpus):
        raw = word_tokenize(corpus.raw())

        # The ngrams of the the new listing of words
        self.ngram_major += ngrams(raw, self.numberGrams)

        # Dictionary of a number of occurrences for a particular gram
        self.ngram_major_counter = Counter(self.ngram_major)

        # get all the grams of smaller size
        self.ngram_minor += ngrams(raw, self.numberGrams - 1)

        # get the counter of the gramsOfSmaller
        self.ngram_minor_counter = Counter(self.ngram_minor)

    def prob_bad(self, l):
        # make sure the list l has same length of corpus trained on
        if len(l) != self.numberGrams:
            return -1

        gram = tuple(l)

        # get the shorter gram
        shorter_list = l[:-1]
        small_gram = tuple(shorter_list)

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
