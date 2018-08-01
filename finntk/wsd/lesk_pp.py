from finntk.emb.utils import avg_vec_from_vecs
from finntk.emb.autoextend import vecs as fiwn_vecs, mk_lemma_vec


def mk_context_vec(sent_lemmas):
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
                    fiwn_space.get_vector(lemma_str)
                except KeyError:
                    pass

    return avg_vec_from_vecs(gen())
