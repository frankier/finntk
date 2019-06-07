"""
Functions for basic processing of OMorFi analyses.
"""
from more_itertools import split_at
import re
from itertools import product


def analysis_to_pairs(ana):
    assert ana[0] == "[" and ana[-1] == "]"
    ana = ana[1:-1]
    for bit in ana.split("]["):
        k, v = bit.split("=", 1)
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


def generate_dict(ana, no_passthrough=False):
    from .inst import get_omorfi

    omor = get_omorfi()
    ana_cp = ana.copy()
    if "weight" in ana_cp:
        del ana_cp["weight"]
    ana_txt = dict_to_analysis(ana_cp)
    return {
        gen["surf"]
        for gen in omor.generate(ana_txt)
        if not (no_passthrough and gen.get("genweight") == float("inf"))
    }


def generate_or_passthrough(ana):
    return {
        norm_word_id(ana["word_id"]) if s.startswith("[") else s
        for s in generate_dict(ana)
    }


def simple_lemmatise(subword_dict):
    """
    This just gets the lemma according to OMorFi.
    """
    return norm_word_id(subword_dict["word_id"])


def default_lemmatise(subword_dict):
    return [simple_lemmatise(subword_dict)]


VERB_ENDING = {"voice": "ACT", "inf": "A", "num": "SG", "case": "LAT"}

NOUN_ENDING = {"num": "SG", "case": "NOM"}

IGNORE_ALL = {("num", "SG")}

IGNORE_KEYS = {"weight", "casechange"}

IGNORE_VERB = {("inf", "A"), ("case", "LAT"), ("voice", "ACT")}

IGNORE_NOUN = {("case", "NOM")}


def add_feat(ending, feats, k, v):
    if (
        k in IGNORE_KEYS
        or (k, v) in IGNORE_ALL
        or (ending == "verb" and (k, v) in IGNORE_VERB)
        or (ending == "noun" and (k, v) in IGNORE_NOUN)
    ):
        return
    feats[k] = v


def app_lemma_feats(lemma_feats, lemma, feats):
    lemma_feats.setdefault(lemma, set()).add(feats)


def ext_lemma_feats(lemma_feats, lemma, feats_list):
    lemma_feats.setdefault(lemma, set()).update(feats_list)


def true_lemmatise(subword_dict, strict=False, return_feats=False):
    """
    This gets the lemma by setting all features to their lemma values, but
    being careful not to cross word derrivation boundaries; This will remove
    inflections including verb infinitive endings as well as particles, but
    not derivational morphemes.
    """

    def default_return():
        simple_lemma = simple_lemmatise(subword_dict)
        if return_feats:
            return {simple_lemma: {}}
        else:
            return [simple_lemma]

    upos = subword_dict.get("upos")
    if upos not in ("VERB", "AUX", "NOUN", "PROPN", "ADJ"):
        if strict:
            assert upos is not None, "no upos found in subword_dict passed to true_lemmatise"
            # As far as I know only verb, noun and adj can have drv
            assert "drv" in subword_dict, "true_lemmatise in strict mode found drv in subword for unsupported UPOS"
        return default_return()
    new_subword_dict = {}
    ending = None
    feats = {}
    for k, v in subword_dict.items():
        if k in ("mood", "voice"):
            ending = "verb"
        elif k == "num":
            ending = "noun"

        if ending is not None:
            if not return_feats:
                break
            add_feat(ending, feats, k, v)
        else:
            new_subword_dict[k] = v
    if ending is None:
        if strict:
            assert False, "true_lemmatise in strict mode couldn't determine which ending to add"
        else:
            return default_return()
    elif ending == "verb":
        new_subword_dict.update(VERB_ENDING)
    elif ending == "noun":
        new_subword_dict.update(NOUN_ENDING)
    # XXX: When does this generate multiple? Can we prune down to one?
    generated = generate_dict(new_subword_dict, no_passthrough=True)
    if return_feats:
        res = {}
        for gen in generated:
            app_lemma_feats(res, gen, tuple(feats.items()))
        return res
    else:
        return generated


def lemmas_of_subword_dicts(
    subword_dicts, lemmatise_func=default_lemmatise, return_feats=False
):
    subword_dicts = list(subword_dicts)
    res = {} if return_feats else set()
    for prefixes in product(*(generate_or_passthrough(d) for d in subword_dicts[:-1])):

        def form_lemma(lemma):
            return "".join(prefixes) + lemma

        if return_feats:
            for lemma, feats in lemmatise_func(
                subword_dicts[-1], return_feats=True
            ).items():
                ext_lemma_feats(res, form_lemma(lemma), feats)
        else:
            for lemma in lemmatise_func(subword_dicts[-1]):
                res.add(form_lemma(lemma))
    return res


EXTRA_WORD_ID = re.compile(r"_\d+$")


def norm_word_id(word_id):
    extra_match = EXTRA_WORD_ID.search(word_id)
    if extra_match:
        word_id = word_id[:extra_match.start()]
    return word_id.lower()


def yield_get(m, k):
    res = m.get(k)
    if res is not None:
        yield res


def normseg(subword_dict):
    """
    Generates a normalised segmentation from an OMorFi analysis dict
    `subword_dict`.

    This function is a work in progress. Currently, it *will* miss out
    morphemes.
    """
    from finntk.data.omorfi_normseg import (
        MOOD_MAP,
        TENSE_MAP,
        PERS_MAP,
        NUM_MAP,
        CASE_MAP,
    )

    for k, v in subword_dict.items():
        v_lower = v.lower()
        if k == "word_id":
            yield norm_word_id(v)
        elif k in ("drv", "clit"):
            if v_lower == "mpi":
                v_lower = "empi"
            yield "-" + v_lower
        elif k == "mood":
            yield from yield_get(MOOD_MAP, v_lower)
        elif k == "tense":
            yield from yield_get(TENSE_MAP, v_lower)
        elif k == "pers":
            yield from yield_get(PERS_MAP, v_lower)
        elif k == "num":
            yield from yield_get(NUM_MAP, v_lower)
        elif k == "case":
            yield from yield_get(CASE_MAP.get(v_lower))
