from sklearn.neighbors import NearestNeighbors
import numpy as np
from os import makedirs
from os.path import join as pjoin, exists
import pickle
import logging


logger = logging.getLogger(__name__)


def normalize(vec):
    vec_norm = np.linalg.norm(vec)
    if not np.nonzero(vec_norm):
        return None
    return vec / vec_norm


class WordExpertBase:

    def __init__(self, algorithm="ball_tree"):
        self.ys = []
        self.clf = None
        self.algorithm = algorithm

    def add_word(self, ctx_vec, sense_key):
        self.ys.append(sense_key)

    def fit(self):
        clf = NearestNeighbors(n_neighbors=1, algorithm=self.algorithm)
        clf.fit(self.xs, self.ys)
        self.clf = clf
        del self.xs

    def predict(self, ctx_vec):
        indices = self.clf.kneighbors([ctx_vec], return_distance=False)
        index = indices[0, 0]
        return self.ys[index]


class OverfilledFixedWordExpert(Exception):
    pass


class FixedWordExpert(WordExpertBase):
    """
    This uses np.float64 because it's what ball_tree uses
    """

    def __init__(self, size, algorithm="ball_tree"):
        self.xs = None
        self.x_idx = 0
        self.size = size
        super().__init__(algorithm=algorithm)

    def fit(self):
        if self.xs is not None:
            self.xs.resize(self.x_idx, self.xs.shape[1])
        else:
            self.xs = [[]]
        super().fit()

    def add_word(self, ctx_vec, sense_key):
        if self.x_idx > self.size:
            raise OverfilledFixedWordExpert(
                f"Trying to add a word to a full FixedWordExpert with capacity: {self.size}"
            )
        normed = normalize(ctx_vec)
        if normed is None:
            # Actually don't add it
            return
        if self.xs is None:
            self.xs = np.ndarray((self.size, ctx_vec.shape[0]), dtype=np.float64)
        self.xs[self.x_idx] = normed
        self.x_idx += 1
        super().add_word(ctx_vec, sense_key)


class VarWordExpert(WordExpertBase):

    def __init__(self, algorithm="ball_tree"):
        self.xs = []
        super().__init__(algorithm=algorithm)

    def add_word(self, ctx_vec, sense_key):
        normed = normalize(ctx_vec)
        if normed is None:
            # Actually don't add it
            return
        self.xs.append(normed)
        super().add_word(ctx_vec, sense_key)


class ExpertExists(Exception):
    pass


class WordExpertManager:

    def __init__(self, path, mode="r", allow_overwrites=False):
        self.path = path
        self.allow_overwrites = allow_overwrites
        if mode == "w":
            makedirs(path, exist_ok=allow_overwrites)

    def _expert_path(self, word):
        return pjoin(self.path, word)

    def dump_expert(self, word, expert):
        expert_path = self._expert_path(word)
        if not self.allow_overwrites and exists(expert_path):
            raise ExpertExists(f"Not overwriting existing word expert: {expert_path}")
        logger.debug(f"Dumping word expert {word}; {expert} to {expert_path}")
        with open(expert_path, "wb") as outf:
            pickle.dump(expert, outf, protocol=4)
        logger.debug("Dumped")

    def load_expert(self, word):
        expert_path = self._expert_path(word)
        logger.debug(f"Loading {word} from {expert_path}")
        if not exists(expert_path):
            logger.debug("(doesn't exist)")
            return
        with open(expert_path, "rb") as inf:
            return pickle.load(inf)
