from finntk.emb.autoextend import vecs as fiwn_vecs, mk_lemma_vec
import numpy as np


def mk_context_vec(aggf, sent_lemmas, lang=None):
    fiwn_space = fiwn_vecs.get_vecs()

    def gen():
        for lemma_str, lemmas in sent_lemmas:
            yielded = False
            if len(lemmas) == 1:
                try:
                    yield lemma_str, mk_lemma_vec(lemmas[0])
                except KeyError:
                    pass
                else:
                    yielded = True
            if not yielded:
                try:
                    yield lemma_str, fiwn_space.get_vector(lemma_str)
                except KeyError:
                    pass

    words, vecs = zip(*gen())
    mat = np.stack(vecs)
    if hasattr(aggf, "needs_words"):
        return aggf(mat, words, lang)
    else:
        return aggf(mat)
