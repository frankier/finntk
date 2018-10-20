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


def bow_to_mat(space, stream):
    refs = []
    vecs = []
    for ref in stream:
        vec = space.get_vec(ref)
        if vec is None:
            continue
        refs.append(ref)
        vecs.append(vec)
    if not refs:
        return refs, None
    return refs, np.stack(vecs)


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


def pre_sif_mean_inner(mat, freqs, a):
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
    return mat.mean(axis=0)


def get_wf(maybe_pair):
    if isinstance(maybe_pair, tuple):
        return maybe_pair[0]
    return maybe_pair


def pre_sif_mean(mat, refs, lang):
    return pre_sif_mean_inner(
        mat, (wordfreq.word_frequency(get_wf(ref), lang) for ref in refs), 1e-3
    )


pre_sif_mean.needs_words = True


def mk_sif_mean(pc):

    def sif_mean(mat, refs, lang):
        emb = pre_sif_mean(mat, refs, lang)
        return emb - emb.dot(pc) * pc

    sif_mean.needs_words = True
    return sif_mean


def unnormalized_mean(mat):
    return mat.mean(axis=0)


def normalized_mean(mat):
    normalized_mat = normalize(mat)
    return normalized_mat.mean(axis=0)


def apply_vec(func, vec_getter, stream, lang=None, **extra):
    refs, mat = bow_to_mat(vec_getter, stream)
    if mat is None:
        return
    if hasattr(func, "needs_words"):
        return func(mat, refs, lang, **extra)
    else:
        return func(mat, **extra)


def cosine_sim(u, v):
    from scipy.spatial.distance import cosine

    return 1 - cosine(u, v)


def get(space, ref):
    try:
        return space.get_vector(ref)
    except KeyError:
        pass
