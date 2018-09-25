import wordfreq
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from numpy.ma import size


def power_mean_inner(p, mat, axis=0):
    mat_cmplx = mat.astype(np.complex128)
    result_cmplx = np.power(
        (np.power(mat_cmplx, p)).sum(axis=axis) / size(mat_cmplx, axis=axis), 1 / p
    )
    return result_cmplx.real.astype(np.float_)


def power_mean(p, mat):
    if p == float("inf"):
        return mat.max(axis=0)
    elif p == float("-inf"):
        return mat.min(axis=0)
    elif p == 1:
        return mat.mean(axis=0)
    else:
        return power_mean_inner(p, mat, axis=0)


def bow_to_mat(vec_getter, stream):
    words = []
    vecs = []
    for word in stream:
        try:
            vec = vec_getter(word)
        except KeyError:
            continue
        else:
            words.append(word)
            vecs.append(vec)
    if not words:
        return words, None
    return words, np.stack(vecs)


CATP_3 = (float("-inf"), 1, float("inf"))
CATP_4 = (float("-inf"), 1, 3, float("inf"))


def catp_mean(mat, ps=CATP_3):
    """
    From *Concatenated p-mean Word Embeddings as Universal Cross-Lingual
    Sentence Representations*
    https://arxiv.org/pdf/1803.01400.pdf
    """
    p_means = []
    for p in ps:
        p_means.append(power_mean(p, mat))
    return np.hstack(p_means)


def compute_pc(X):
    svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_


def sif_mean_inner(mat, freqs, a):
    """
    From *A Simple but Tough-to-Beat Baseline for Sentence Embeddings*
    https://openreview.net/forum?id=SyK00v5xx
    https://github.com/PrincetonML/SIF
    """
    # 1. Normalize
    mat = normalize(mat)
    # 2. Reweight
    rows, cols = mat.shape
    for coli, freq in enumerate(freqs):
        mat[coli, :] = (a / (a + freq)) * mat[coli, :]
    # 3. Remove principal component
    vec = mat.mean(axis=0)
    if np.size(mat, 0) <= 1:
        return vec
    pc = compute_pc(mat)
    return vec - pc[0, :]


def sif_mean(mat, words, lang):
    return sif_mean_inner(
        mat, (wordfreq.word_frequency(wf, lang) for wf in words), 1e-3
    )


sif_mean.needs_words = True


def unnormalized_mean(mat):
    return mat.mean(axis=0)


def normalized_mean(mat):
    normalized_mat = normalize(mat)
    return normalized_mat.mean(axis=0)


def apply_vec(func, vec_getter, stream, lang=None, **extra):
    words, mat = bow_to_mat(vec_getter, stream)
    if mat is None:
        return
    if hasattr(func, "needs_words"):
        return func(mat, words, lang, **extra)
    else:
        return func(mat, **extra)


def cosine_sim(u, v):
    from scipy.spatial.distance import cosine

    return 1 - cosine(u, v)
