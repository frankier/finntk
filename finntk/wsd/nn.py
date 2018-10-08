from sklearn.neighbors import NearestNeighbors
import numpy as np


class WsdNn:

    def __init__(self):
        self.classifiers = {}
        self.xs = {}
        self.ys = {}

    def add_word(self, word, ctx_vec, sense_key):
        vec_norm = np.linalg.norm(ctx_vec)
        self.xs.setdefault(word, []).append(ctx_vec / vec_norm)
        self.ys.setdefault(word, []).append(sense_key)

    def fit_word(self, word):
        if word not in self.xs or word not in self.ys:
            return
        clf = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
        clf.fit(self.xs[word], self.ys[word])
        self.classifiers[word] = clf
        del self.xs[word]

    def fit_all(self):
        for word in self.xs.keys():
            self.fit_word(word)
        del self.xs

    def predict(self, word, ctx_vec):
        indices = self.classifiers[word].kneighbors([ctx_vec], return_distance=False)
        index = indices[0, 0]
        return self.ys[word][index]
