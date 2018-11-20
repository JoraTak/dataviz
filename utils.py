import re
import time
import itertools
import numpy as np

# For pretty-printing
import pandas as pd
from IPython.display import display, HTML

from . import constants

# Word processing functions
def canonicalize_digits(word):
    if any([c.isalpha() for c in word]): return word
    word = re.sub("\d", "DG", word)
    if word.startswith("DG"):
        word = word.replace(",", "") # remove thousands separator
    return word

def canonicalize_word(word, wordset=None, digits=True):
    word = word.lower()
    if digits:
        if (wordset != None) and (word in wordset): return word
        word = canonicalize_digits(word) # try to canonicalize numbers
    if (wordset == None) or (word in wordset):
        return word
    else:
        return constants.UNK_TOKEN

def canonicalize_words(words, **kw):
    return [canonicalize_word(word, **kw) for word in words]

def build_vocab(corpus, V=10000, **kw):
    from . import vocabulary
    if isinstance(corpus, list):
        token_feed = (canonicalize_word(w) for w in corpus)
        vocab = vocabulary.Vocabulary(token_feed, size=V, **kw)
    else:
        token_feed = (canonicalize_word(w) for w in corpus.words())
        vocab = vocabulary.Vocabulary(token_feed, size=V, **kw)

    print("Vocabulary: {:,} types".format(vocab.size))
    return vocab