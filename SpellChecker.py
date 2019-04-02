import nltk
from nltk.corpus import words
from nltk.tokenize import word_tokenize
from nltk.stem import *
from NGramModel import NGramModel
from nltk.corpus import brown, state_union
from collections import Counter
import string


class SpellChecker:

    def __init__(self):
        self.allWords = set(words.words()).union(set(string.punctuation))
        self.countWords = Counter(self.allWords)
        self.numWords = len(self.allWords)
        self.snowball = SnowballStemmer("english")
        self.porter = PorterStemmer()
        self.wnl = WordNetLemmatizer()
        self.ngram = NGramModel(2)

    def check(self, words_str):
        if not isinstance(words_str, str):
            raise Exception("check() takes a string")

        misspelled = []
        count = 0
        tokens = word_tokenize(words_str)
        for word in tokens:
            if self.is_valid_word(word):
                count += 1
                continue

            candidates = self.all_edits_candidates(word)
            sorted(candidates, key=lambda candidate: 1 - nltk.edit_distance(word, candidate))

            final_candidates = []
            gram = []
            if count >= 2:
                gram.append(tokens[count - 1])
                gram.append(tokens[count])
                prob_tuples = self.ngram.prob_bad(gram)
                for tup in prob_tuples:
                    if tup[0] in candidates:
                        final_candidates.append(tup)

            sorted(final_candidates, key=lambda final_tup: final_tup[1])

            if len(final_candidates) == 0:
                final_candidates = candidates

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
        raw_candidates = self.all_edits(word, 2)

        candidates = []

        for word in raw_candidates:
            if self.is_valid_word(word):
                candidates.append(word)

        return candidates

    def all_edits(self, word, level):

        if level == 0:
            return set()

        letters = 'abcdefghijklmnopqrstuvwxyz'
        split_word = [(word[:i], word[i:]) for i in range(len(word) + 1)]

        delete_chars = [left + right[1:] for left, right in split_word if right]
        switch_chars = [left + right[1] + right[0] + right[2:] for left, right in split_word if len(right)>1]
        replace_chars = [left + let + right[1:] for left, right in split_word if right for let in letters]
        insert_chars = [left + let + right for left, right in split_word for let in letters]

        return set(delete_chars + switch_chars + replace_chars + insert_chars).union(self.all_edits(word, level - 1))
