"""
Functions for basic processing of OMorFi analyses.
"""
from more_itertools import split_at
import re
from itertools import product


def analysis_to_pairs(ana):
    for bit in ana.split("]["):
        k, v = bit.strip("[]").split("=", 1)
        yield k, v


def pairs_to_dict(it):
    return dict(((k.lower(), v) for k, v in it))


def analysis_to_dict(ana):
    return pairs_to_dict(analysis_to_pairs(ana))


def dict_to_analysis(d):
    return "[{}]".format(
        "][".join(["{}={}".format(k.upper(), v) for k, v in d.items()])
    )


def chunk_subwords(it):

    def is_cmp_bound(kv):
        return (kv[0] == "BOUNDARY" and kv[1] == "COMPOUND")

    return split_at(it, is_cmp_bound)


def analysis_to_subword_dicts(ana):
    """
    Returns a list of list of dicts. Each list element is an analysis. For each
    analysis, there is a list of subwords. Each dict contains an Omorfi
    analysis
    """
    return map(pairs_to_dict, chunk_subwords(analysis_to_pairs(ana)))


def generate_dict(ana):
    from .inst import get_omorfi

    omor = get_omorfi()
    ana_cp = ana.copy()
    if "weight" in ana_cp:
        del ana_cp["weight"]
    ana_txt = dict_to_analysis(ana_cp)
    return {gen["surf"] for gen in omor.generate(ana_txt)}


def generate_or_passthrough(ana):
    return {ana["word_id"] if s.startswith("[") else s for s in generate_dict(ana)}


def lemmas_of_subword_dicts(subword_dicts):
    subword_dicts = list(subword_dicts)
    return [
        "".join(prefixes) + norm_word_id(subword_dicts[-1]["word_id"])
        for prefixes in product(
            *(generate_or_passthrough(d) for d in subword_dicts[:-1])
        )
    ]


EXTRA_WORD_ID = re.compile("_\d$")


def norm_word_id(word_id):
    extra_match = EXTRA_WORD_ID.match(word_id)
    if extra_match:
        word_id = word_id[:extra_match.start()]
    return word_id.lower()
