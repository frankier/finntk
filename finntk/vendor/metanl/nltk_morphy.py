# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals

import nltk
from nltk.corpus import wordnet

try:
    morphy = wordnet._morphy
except LookupError:
    nltk.download('wordnet')
    morphy = wordnet._morphy

STOPWORDS = ['the', 'a', 'an']

EXCEPTIONS = {
    # Avoid obsolete and obscure roots, the way lexicographers don't.
    'wrought': 'wrought',   # not 'work'
    'media': 'media',       # not 'medium'
    'installed': 'install', # not 'instal'
    'installing': 'install',# not 'instal'
    'synapses': 'synapse',  # not 'synapsis'
    'soles': 'sole',        # not 'sol'
    'pubes': 'pube',        # not 'pubis'
    'dui': 'dui',           # not 'duo'
    'taxis': 'taxi',        # not 'taxis'

    # Work around errors that Morphy makes.
    'alas': 'alas',
    'corps': 'corps',
    'cos': 'cos',
    'enured': 'enure',
    'fiver': 'fiver',
    'hinder': 'hinder',
    'lobed': 'lobe',
    'offerer': 'offerer',
    'outer': 'outer',
    'sang': 'sing',
    'singing': 'sing',
    'solderer': 'solderer',
    'tined': 'tine',
    'twiner': 'twiner',
    'us': 'us',

    # Stem common nouns whose plurals are apparently ambiguous
    'teeth': 'tooth',
    'things': 'thing',
    'people': 'person',

    # Tokenization artifacts
    'wo': 'will',
    'ca': 'can',
    "n't": 'not',
}

AMBIGUOUS_EXCEPTIONS = {
    # Avoid nouns that shadow more common verbs.
    'am': 'be',
    'as': 'as',
    'are': 'be',
    'ate': 'eat',
    'bent': 'bend',
    'drove': 'drive',
    'fell': 'fall',
    'felt': 'feel',
    'found': 'find',
    'has': 'have',
    'lit': 'light',
    'lost': 'lose',
    'sat': 'sit',
    'saw': 'see',
    'sent': 'send',
    'shook': 'shake',
    'shot': 'shoot',
    'slain': 'slay',
    'spoke': 'speak',
    'stole': 'steal',
    'sung': 'sing',
    'thought': 'think',
    'tore': 'tear',
    'was': 'be',
    'won': 'win',
    'feed': 'feed',
}


def _word_badness(word):
    """
    Assign a heuristic to possible outputs from Morphy. Minimizing this
    heuristic avoids incorrect stems.
    """
    if word.endswith('e'):
        return len(word) - 2
    elif word.endswith('ess'):
        return len(word) - 10
    elif word.endswith('ss'):
        return len(word) - 4
    else:
        return len(word)


def _morphy_best(word, pos=None):
    """
    Get the most likely stem for a word using Morphy, once the input has been
    pre-processed by morphy_stem().
    """
    results = []
    if pos is None:
        pos = 'nvar'
    for pos_item in pos:
        results.extend(morphy(word, pos_item))
    if not results:
        return None
    results.sort(key=lambda x: _word_badness(x))
    return results[0]


def morphy_stem(word, pos=None):
    """
    Get the most likely stem for a word. If a part of speech is supplied,
    the stem will be more accurate.
    Valid parts of speech are:
    - 'n' or 'NN' for nouns
    - 'v' or 'VB' for verbs
    - 'a' or 'JJ' for adjectives
    - 'r' or 'RB' for adverbs
    Any other part of speech will be treated as unknown.
    """
    word = word.lower()
    if pos is not None:
        if pos.startswith('NN'):
            pos = 'n'
        elif pos.startswith('VB'):
            pos = 'v'
        elif pos.startswith('JJ'):
            pos = 'a'
        elif pos.startswith('RB'):
            pos = 'r'
    if pos is None and word.endswith('ing') or word.endswith('ed'):
        pos = 'v'
    if pos is not None and pos not in 'nvar':
        pos = None
    if word in EXCEPTIONS:
        return EXCEPTIONS[word]
    if pos is None:
        if word in AMBIGUOUS_EXCEPTIONS:
            return AMBIGUOUS_EXCEPTIONS[word]
    return _morphy_best(word, pos) or word


def tag_and_stem(tokens):
    """
    Returns a list of (stem, tag, token) triples:
    - stem: the word's uninflected form
    - tag: the word's part of speech
    - token: the original word, so we can reconstruct it later
    """
    tagged = nltk.pos_tag(tokens)
    out = []
    for token, tag in tagged:
        stem = morphy_stem(token, tag)
        out.append((stem, tag, token))
    return out


def good_lemma(lemma):
    return lemma and lemma not in STOPWORDS and lemma[0].isalnum()


def normalize_list(tokens):
    """
    Get a list of word stems that appear in the text. Stopwords and an initial
    'to' will be stripped.
    """
    pieces = [morphy_stem(word) for word in tokens]
    pieces = [piece for piece in pieces if good_lemma(piece)]
    if not pieces:
        return []
    if pieces[0] == 'to':
        pieces = pieces[1:]
    return pieces
