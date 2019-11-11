"""
Implementation of Lesk++, which *may* be broken.
"""
from finntk.emb.autoextend import vecs as fiwn_vecs, mk_lemma_vec
import numpy as np
from .lesk_emb import MultilingualLesk
import logging


logger = logging.getLogger(__name__)


class LeskPP:

    def __init__(self, multispace, aggf, wn_filter=False, expand=False):
        self.multispace = multispace
        self.aggf = aggf
        self.wn_filter = wn_filter
        self.expand = expand
        self.ml_lesk = MultilingualLesk(multispace, aggf, wn_filter, expand)

    def mk_defn_vec(self, item):
        return self.ml_lesk.mk_defn_vec(item)

    def mk_ctx_vec(self, sent_lemmas, exclude_idx=None):
        fiwn_space = fiwn_vecs.get_vecs()

        words = []
        vecs = []
        for idx, (lemma_str, lemmas) in enumerate(sent_lemmas):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"#{idx}: {lemma_str}")
            if exclude_idx is not None and idx == exclude_idx:
                continue
            vec_appended = False
            if len(lemmas) == 1:
                try:
                    logger.debug("Used wn lemma vec")
                    vecs.append(mk_lemma_vec(lemmas[0]))
                except KeyError:
                    pass
                else:
                    vec_appended = True
            if not vec_appended:
                try:
                    logger.debug("Used surface vec")
                    vecs.append(fiwn_space.get_vector(lemma_str))
                except KeyError:
                    pass
                else:
                    vec_appended = True
            if vec_appended:
                words.append(lemma_str)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Used words: {words}")
            logger.debug(f"Got vecs: {vecs}")

        if not vecs:
            logger.debug(f"Early return!")
            return
        mat = np.stack(vecs)
        logger.debug(f"Mat: {mat}")
        if hasattr(self.aggf, "needs_words"):
            return self.aggf(mat, words, "fi")
        else:
            return self.aggf(mat)
