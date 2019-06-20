import ahocorasick
from finntk.data.wordnet import ALL_ABBRVS

_abbrv_auto = None


def get_abbrv_auto():
    global _abbrv_auto
    if _abbrv_auto is not None:
        return _abbrv_auto
    _abbrv_auto = ahocorasick.Automaton()
    for abbrv in ALL_ABBRVS:
        _abbrv_auto.add_word("_{}_".format(abbrv), abbrv)
    _abbrv_auto.make_automaton()
    return _abbrv_auto


def has_abbrv(lemma):
    """
    Given a FinnWordNet formatted lemma, e.g. saada_tehdä_jtak return whether
    it contains a placeholder abbreviation.
    """
    abbrv_auto = get_abbrv_auto()
    it = abbrv_auto.iter("_{}_".format(lemma))
    return len(list(it))
