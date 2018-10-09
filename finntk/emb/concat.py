import numpy as np
from typing import Tuple

from .base import RefType
from .fasttext import multispace as fasttext_ms
from .numberbatch import multispace as numberbatch_ms
from .base import MultilingualVectorSpace


class FastTextNumberbatchMultiSpace(MultilingualVectorSpace):
    takes = RefType.BOTH

    def get_vec(self, lang: str, ref: Tuple[str, str]):
        wf, lemma = ref
        return np.hstack(
            [fasttext_ms.get_vec(lang, wf), numberbatch_ms.get_vec(lang, wf)]
        )


ft_nb_multispace = FastTextNumberbatchMultiSpace()
