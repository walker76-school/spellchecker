from collections import Counter
from nltk import ngrams


class NGramModel:

    def __init__(self, corpus, n, maxword):
        if n < 1 or maxword < 1:
            raise Exception("Silly noodle, negatives aren't fun")

        # Special Word
        self.SPECIALWORD = "??"

        # corpus
        self.corpus = corpus

        # Number of Grams
        self.numberGrams = n

        # Number of words to consider
        self.maxNumberGrams = maxword

        # List of word in in corpus
        self.listOfTotalWords = self.corpus.words()

        # Dictionary of Words
        self.dictionaryOfWords = Counter(self.listOfTotalWords)

        # Dictionary of maxword most common
        self.listOfCommonWords = self.dictionaryOfWords.most_common(maxword)

        # Make sure our special word is special
        flag = True
        while flag:
            if self.SPECIALWORD in self.dictionaryOfWords:
                self.SPECIALWORD = self.SPECIALWORD + "?"
            else:
                flag = False

        # We have to use a dictionary here because a set cannot
        # house elements of length one, which, for example, a period can
        # be one of our most common words. Therefore we made a dictionary
        # with literally a dummy value
        self.dictOfCommonWords = dict()
        for word in self.listOfCommonWords:
            self.dictOfCommonWords[word.__getitem__(0)] = "dummy"

        # The new listing of words after we replace all the
        # undesired words
        self.newListingOfWords = []
        for word in self.listOfTotalWords:
            if word in self.dictOfCommonWords:
                self.newListingOfWords.append(word)
            else:
                self.newListingOfWords.append(self.SPECIALWORD)

        # Length of newListingOfWords
        self.lengthOfNewListingOfWords = len(self.newListingOfWords)

        # The ngrams of the the new listing of words
        self.ngramsOfNewList = ngrams(self.newListingOfWords, self.numberGrams)

        # Dictionary of a number of occurrences for a particular gram
        self.numberOccurancesOfGrams = Counter(self.ngramsOfNewList)

        # get all the grams of smaller size
        self.gramsOfSmaller = ngrams(self.newListingOfWords, self.numberGrams - 1)

        # get the counter of the gramsOfSmaller
        self.gramsOfSmallerCounter = Counter(self.gramsOfSmaller)

    # Returns the special word used
    def special_word(self):
        return self.SPECIALWORD

    # returns the frequency of a gram in a corpus
    def freq(self, l):
        # make sure the list l has same length of corpus trained on
        if len(l) != self.numberGrams:
            return -1

        term = tuple(l)
        # If our gram is in the corpus
        if term in self.numberOccurancesOfGrams:
            return self.numberOccurancesOfGrams[term]
        else:
            return 0

    def prob(self, l):
        # make sure the list l has same length of corpus trained on
        if len(l) != self.numberGrams:
            return -1

        gram = tuple(l)

        # get the shorter list
        shorter_list = l[:-1]

        # get the total amount of occurrences for that gram
        num_total_gram = self.numberOccurancesOfGrams[gram]

        small_gram = tuple(shorter_list)

        # return the probability
        return num_total_gram / self.gramsOfSmallerCounter[small_gram]
