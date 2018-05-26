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
    return _extract_lemmas(
        word_form, lambda analysis_dicts: [[d] for d in analysis_dicts]
    )


def extract_lemmas_combs(word_form):
    return _extract_lemmas(word_form, contig_slices)


def extract_lemmas_recurs(word_form):
    expand_queue = [word_form]
    res = set()
    while len(expand_queue) > 0:
        word_form = expand_queue.pop()
        new_lemmas = extract_lemmas_combs(word_form)
        novel_lemmas = new_lemmas - res
        print("novel_lemmas", novel_lemmas)
        expand_queue.extend(novel_lemmas)
        for lemma in novel_lemmas:
            res.add(lemma)
    return res
