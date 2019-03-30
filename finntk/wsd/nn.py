from sklearn.neighbors import NearestNeighbors
import numpy as np
from os import makedirs
from os.path import join as pjoin, exists
import pickle


def normalize(vec):
    vec_norm = np.linalg.norm(vec)
    return vec / vec_norm


class WordExpertBase:

    def __init__(self):
        self.ys = []
        self.clf = None

    def add_word(self, ctx_vec, sense_key):
        self.ys.append(sense_key)

    def fit(self):
        clf = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
        clf.fit(self.xs, self.ys)
        self.clf = clf
        del self.xs

    def predict(self, ctx_vec):
        indices = self.clf.kneighbors([ctx_vec], return_distance=False)
        index = indices[0, 0]
        return self.ys[index]


class FixedWordExpert(WordExpertBase):
    """
    This uses np.float64 because it's what ball_tree uses
    """

    def __init__(self, size):
        self.xs = None
        self.x_idx = 0
        self.size = size
        super().__init__()

    def add_word(self, ctx_vec, sense_key):
        if self.xs is None:
            self.xs = np.ndarray((self.size, ctx_vec.shape[0]), dtype=np.float64)
        self.xs[self.x_idx] = normalize(ctx_vec)
        self.x_idx += 1
        super().add_word(ctx_vec, sense_key)


class VarWordExpert(WordExpertBase):

    def __init__(self):
        self.xs = []
        super().__init__()

    def add_word(self, ctx_vec, sense_key):
        self.xs.append(normalize(ctx_vec))
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
        with open(expert_path, "wb") as outf:
            pickle.dump(expert, outf)

    def load_expert(self, word):
        expert_path = self._expert_path(word)
        if not exists(expert_path):
            return
        with open(expert_path, "rb") as inf:
            return pickle.load(inf)
