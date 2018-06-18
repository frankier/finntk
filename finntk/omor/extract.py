"""
Functions for extracting lemmas from OMorFi analyses.
"""
from .inst import get_omorfi
from .anlys import analysis_to_subword_dicts, lemmas_of_subword_dicts


def contig_slices(elems):
    elems = list(elems)
    n = len(elems)
    for start in range(n):
        for end in range(start + 1, n + 1):
            yield elems[start:end]


def _extract_lemmas(word_form, get_slices):
    omorfi = get_omorfi()
    analyses = omorfi.analyse(word_form)
    res = set()
    for analysis in analyses:
        analysis_dicts = analysis_to_subword_dicts(analysis["anal"])
        for analysis_slice in get_slices(analysis_dicts):
            for lemma in lemmas_of_subword_dicts(analysis_slice):
                res.add(lemma)
    return res


def extract_lemmas(word_form):
    """
    Extract lemmas specifically mentioned by OMorFi.
    """
    return _extract_lemmas(
        word_form, lambda analysis_dicts: [[d] for d in analysis_dicts]
    )


def extract_lemmas_span(word_form):
    """
    Works like `extract_lemmas`, but doesn't extract individual subwords.
    However, if a word is only recognised by as a compound word by OMorFi it
    will glue the parts together, lemmatising only the last subword. This means
    it extracts only lemmas which span the whole word form.
    """
    return _extract_lemmas(word_form, lambda analysis_dicts: [analysis_dicts])


def extract_lemmas_combs(word_form):
    """
    Works like `extract_lemmas`, but also tries to combine adjacent
    subwords to make lemmas which may be out of volcaburary for
    OMorFi.

    Note that this will over generate (by design). For example:
    voileipäkakku will generate voi, voileipä and voileipäkakku  as
    desired, but will also spuriously generate leipäkakku.
    """
    return _extract_lemmas(word_form, contig_slices)


def extract_lemmas_recurs(word_form):
    """
    Works like `extract_lemmas`, but also tries to expand each
    lemma into more lemmas. This helps in some cases (but can
    overgenerate even more). For example, it will mean that
    synnyinkaupunkini will generate synty, kaupunki,
    synnyinkaupunki, synnyin and syntyä.
    """
    expand_queue = [word_form]
    res = set()
    while len(expand_queue) > 0:
        word_form = expand_queue.pop()
        new_lemmas = extract_lemmas_combs(word_form)
        novel_lemmas = new_lemmas - res
        expand_queue.extend(novel_lemmas)
        for lemma in novel_lemmas:
            res.add(lemma)
    return res


def lemma_intersect(toks1, toks2):
    """
    Given two iterables of tokens, return the intersection of their lemmas.
    This can work as a simple, high recall, method of matching for example, two
    inflected noun phrases.
    """
    if len(toks1) != len(toks2):
        return
    res = []
    for t1, t2 in zip(toks1, toks2):
        l1 = extract_lemmas_span(t1)
        l2 = extract_lemmas_span(t2)
        inter = l1 & l2
        if len(inter) == 0:
            return
        res.append(inter)
    return res
