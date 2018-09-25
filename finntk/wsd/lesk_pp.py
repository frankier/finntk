from finntk.emb.autoextend import vecs as fiwn_vecs, mk_lemma_vec
import numpy as np


def mk_context_vec(aggf, sent_lemmas, lang=None):
    fiwn_space = fiwn_vecs.get_vecs()

    def gen():
        for lemma_str, lemmas in sent_lemmas:
            yielded = False
            if len(lemmas) == 1:
                try:
                    yield mk_lemma_vec(lemmas[0])
                except KeyError:
                    pass
                else:
                    yielded = True
            if not yielded:
                try:
                    yield fiwn_space.get_vector(lemma_str)
                except KeyError:
                    pass

    return aggf(np.stack(gen()))
