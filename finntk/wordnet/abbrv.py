import ahocorasick

_abbrv_auto = None

PLACEHOLDER_ABBRVS = [
    "jstak",  # Abbreviation of jostakin.
    "jssak",  # Abbreviation of jossakin.
    "jnak",  # Abbreviation of jonakin.
    "jtak",  # Abbreviation of jotakin.
    "jllak",  # Abbreviation of jollakin.
    "jltak",  # Abbreviation of joltakin.
    "jksta",  # Abbreviation of jostakusta.
    "jllek",  # Abbreviation of jollekin.
    "jkssa",  # Abbreviation of jossakussa.
    "jkta",  # Abbreviation of jotakuta.
    "jklla",  # Abbreviation of jollakulla.
    "jklta",  # Abbreviation of joltakulta.
    "jhk",  # Abbreviation of johonkin.
    "jklle",  # Abbreviation of jollekulle.
    "jksik",  # Abbreviation of joksikin.
    "jkksi",  # Abbreviation of joksikuksi.
    "jkna",  # Abbreviation of jonakuna.
    "jkhun",  # Abbreviation of johonkuhun.
    "jku",  # Abbreviation of joku (“somebody”).
    "jk",  # Abbreviation of jokin (“something”).
    "jkn",  # Abbreviation of jonkun (“of somebody, somebody's”).
    "jnk",  # Abbreviation of jonkin.
]


def get_abbrv_auto():
    global _abbrv_auto
    if _abbrv_auto is not None:
        return _abbrv_auto
    _abbrv_auto = ahocorasick.Automaton()
    for abbrv in PLACEHOLDER_ABBRVS:
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
