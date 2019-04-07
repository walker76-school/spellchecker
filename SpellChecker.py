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
        print("Establishing all words ...")
        self.allWords = set(words.words()).union(set(string.punctuation))
        self.countWords = Counter(self.allWords)
        self.numWords = len(self.allWords)

        print("Establishing stemmers ...")
        self.snowball = SnowballStemmer("english")
        self.porter = PorterStemmer()
        self.wnl = WordNetLemmatizer()

        print("Establishing NGramModel ...")
        self.ngram = NGramModel(2)

        print("Establishing helper vars ... ")
        self.punctuations = '''’!()-[]{};:'"\,<>./?@#$%^&*_~'''
        self.misspelled_dict = {}
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
        print("Discovered %d tokens " % len(tokens))

        for word in tokens:

            # Valid token so continue searching
            if self.is_valid_word(word):
                count += 1
                continue

            # If we haven't already seen the word
            if word not in self.misspelled_dict:

                # Retrieve all the edit distance candidates
                edit_candidates = self.all_edits_candidates(word, 2)

                gram_candidates = []
                if count >= 1:
                    gram = [tokens[count - 1].lower(), word.lower()]
                    gram_candidates = self.ngram.prob(gram)
                    sorted(gram_candidates, key=lambda e1: e1[1])

                # These are candidates in both so we really want them
                absolute_candidates = [e1[0] for e1 in gram_candidates if e1[0] in edit_candidates]

                if len(absolute_candidates) > 0:

                    sorted(absolute_candidates, key=lambda e1: nltk.edit_distance(e1, word))

                    if len(absolute_candidates) > 5:
                        absolute_candidates = absolute_candidates[:5]

                    self.misspelled_dict[word] = absolute_candidates
                else:

                    gram_candidates = [e1 for e1 in gram_candidates if nltk.edit_distance(e1, word) < 3]
                    sorted(gram_candidates, key=lambda e1: e1[1])

                    if len(gram_candidates) > 5:
                        gram_candidates = gram_candidates[:5]

                    sorted(edit_candidates, key=lambda e1: nltk.edit_distance(e1, word))

                    if len(edit_candidates) > 5:
                        edit_candidates = edit_candidates[:5]

                    all_candidates = edit_candidates + gram_candidates

                    # Best - 0.6, 0.4
                    sorted(all_candidates, key=lambda e1: 0.5 * self.ngram.freq(e1) + 0.5 * (1 - nltk.edit_distance(e1, word)))
                    candidates = all_candidates[:5]
                    self.misspelled_dict[word] = candidates

            misspelled.append((count, word, self.misspelled_dict[word]))
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

    def all_edits_candidates(self, word, level):

        print("Retrieving edit distance candidates for %s ..." % word)

        candidates = []
        edit_words = [word.lower()]

        # Keep looping, moving more distance away from original word until we have candidates
        while len(candidates) <= 0 and level > 0:
            raw_candidates = []

            # Check all permutations of all words
            for e1 in edit_words:
                raw_candidates += list(self.edit_distance_candidates(e1))

            edit_words = raw_candidates

            # Filter possible candidates to valid words only
            candidates = [e1 for e1 in raw_candidates if self.is_valid_word(e1)]
            level -= 1

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
