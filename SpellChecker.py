import nltk
from nltk.corpus import words
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import *
from NGramModel import NGramModel
from nltk.corpus import brown, state_union
from collections import Counter
import string


class SpellChecker:

    def __init__(self):
        print("Establishing all words...")
        self.allWords = set(words.words()).union(set(string.punctuation))
        self.countWords = Counter(self.allWords)
        self.numWords = len(self.allWords)
        print("Establishing stemmers...")
        self.snowball = SnowballStemmer("english")
        self.porter = PorterStemmer()
        self.wnl = WordNetLemmatizer()
        self.tokenizer = RegexpTokenizer(r'\w+')
        print("Establishing NGramModel...")
        self.ngram = NGramModel(2)
        self.punctuations = '''’!()-[]{};:'"\,<>./?@#$%^&*_~'''
        print("Done with constructor...")

    def check(self, words_str):
        if not isinstance(words_str, str):
            raise Exception("check() takes a string")

        misspelled = []
        count = 0

        # Remove punctuation from the string
        filtered_str = ""
        for char in words_str:
            if char not in self.punctuations:
                filtered_str = filtered_str + char

        # Tokenize the sentence passed
        tokens = word_tokenize(filtered_str)
        print("Discovered %d tokens" % len(tokens))

        for word in tokens:

            # Valid token so continue searching
            if self.is_valid_word(word):
                count += 1
                continue

            # Retrieve all the edit distance candidates
            candidates = self.all_edits_candidates(word)

            final_candidates = []
            gram = []
            if count >= 1:
                gram.append(tokens[count - 1].lower())
                gram.append(tokens[count].lower())
                prob_tuples = self.ngram.prob(gram)

                # Filter out all candidates from ngram if it's not within reasonable edit distance
                final_candidates = [tup[0] for tup in prob_tuples if tup[0] in candidates]

            # If the ngram didn't have anything then use the edit distance candidates
            if len(final_candidates) == 0:
                final_candidates = candidates

            # Remove duplicates
            final_candidates = set(final_candidates)
            final_candidates = list(final_candidates)

            # Sort candidates based on freq in all corpra
            sorted(final_candidates, key=lambda it: self.ngram.freq(it))

            if len(final_candidates) > 5:
                final_candidates = final_candidates[:5]

            misspelled.append((count, word, final_candidates))
            count += 1

        return misspelled

    def is_valid_word(self, word):
        if word.lower() in self.allWords:
            return True

        if self.snowball.stem(word.lower()) in self.allWords:
            return True

        if self.porter.stem(word.lower()) in self.allWords:
            return True

        if self.wnl.lemmatize(word.lower()) in self.allWords:
            return True

        return False

    def all_edits_candidates(self, word):

        print("Retrieving edit distance candidates for %s ..." % word)

        candidates = []
        edit_words = [word.lower()]

        # Keep looping, moving more distance away from original word until we have candidates
        while len(candidates) <= 0:
            raw_candidates = []

            # Check all permutations of all words
            for e1 in edit_words:
                raw_candidates += list(self.edit_distance_candidates(e1))

            edit_words = raw_candidates

            # Filter possible candidates to valid words only
            candidates = [e1 for e1 in raw_candidates if self.is_valid_word(e1)]

        print("Retrieved %d candidates" % len(candidates))
        return candidates

    def edit_distance_candidates(self, word):
        letters = 'abcdefghijklmnopqrstuvwxyz'
        split_word = [(word[:i], word[i:]) for i in range(len(word) + 1)]

        delete_chars = [left + right[1:] for left, right in split_word if right]
        switch_chars = [left + right[1] + right[0] + right[2:] for left, right in split_word if len(right)>1]
        replace_chars = [left + let + right[1:] for left, right in split_word if right for let in letters]
        insert_chars = [left + let + right for left, right in split_word for let in letters]

        return set(delete_chars + switch_chars + replace_chars + insert_chars)
