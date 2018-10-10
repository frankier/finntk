import numpy as np
from typing import Tuple

from .base import RefType
from .fasttext import multispace as fasttext_ms
from .numberbatch import multispace as numberbatch_ms
from .word2vec import space as word2vec_fi
from .base import MultilingualVectorSpace, MonoVectorSpaceAdapter


class FastTextNumberbatchMultiSpace(MultilingualVectorSpace):
    takes = RefType.BOTH

    def get_vec(self, lang: str, ref: Tuple[str, str]):
        wf, lemma = ref
        return np.hstack(
            [fasttext_ms.get_vec(lang, wf), numberbatch_ms.get_vec(lang, lemma)]
        )


ft_nb_multispace = FastTextNumberbatchMultiSpace()


fasttext_fi = MonoVectorSpaceAdapter(fasttext_ms, "fi")
numberbatch_fi = MonoVectorSpaceAdapter(numberbatch_ms, "fi")


class FastTextNumberbatchWord2VecFiSpace(MultilingualVectorSpace):
    takes = RefType.BOTH

    def get_vec(self, ref: Tuple[str, str]):
        wf, lemma = ref
        return np.hstack(
            [
                fasttext_fi.get_vec(wf),
                numberbatch_fi.get_vec(lemma),
                word2vec_fi.get_vec(wf),
            ]
        )


ft_nb_w2v_space = FastTextNumberbatchWord2VecFiSpace()
