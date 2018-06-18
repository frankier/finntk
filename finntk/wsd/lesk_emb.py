from scipy.spatial.distance import cosine
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
import logging
from urllib.request import urlretrieve
import os
from finntk.utils import ResourceMan

logger = logging.getLogger(__name__)


class WordVecs(ResourceMan):
    RESOURCE_NAME = "fasttext-multilingual"

    FI_URL = "https://s3.amazonaws.com/arrival/embeddings/wiki.multi.fi.vec"
    EN_URL = "https://s3.amazonaws.com/arrival/embeddings/wiki.multi.en.vec"

    def __init__(self):
        super().__init__()
        self._resources["fi_vec"] = "wiki.multi.fi.binvec"
        self._resources["en_vec"] = "wiki.multi.en.binvec"
        self._en = None
        self._fi = None

    def _download(self, lang, url, dest):
        logger.info("Downloading {} word vectors".format(lang))
        tmp_fn, _ = urlretrieve(url)
        try:
            logger.info("Converting {} word vectors".format(lang))
            fi = KeyedVectors.load_word2vec_format(tmp_fn)
            fi.save(dest)
        finally:
            os.remove(tmp_fn)

    def _bootstrap(self, _res):
        self._download("Finnish", self.FI_URL, self._get_res_path("en_vec"))
        self._download("English", self.EN_URL, self._get_res_path("fi_vec"))

    def get_en(self):
        if self._en is None:
            en_vec_path = self.get_res("en_vec")
            logger.info("Loading English word vectors")
            self._en = KeyedVectors.load(en_vec_path, mmap="r")
            logger.info("Loaded English word vectors")
        return self._en

    def get_fi(self):
        if self._fi is None:
            fi_vec_path = self.get_res("fi_vec")
            logger.info("Loading Finnish word vectors")
            self._fi = KeyedVectors.load(fi_vec_path, mmap="r")
            logger.info("Loaded Finnish word vectors")
        return self._fi


word_vecs = WordVecs()


def avg_vec(space, stream):
    word_count = 0
    vec_sum = 0
    for word in stream:
        try:
            vec = space.get_vector(word)
        except KeyError:
            continue
        else:
            vec_sum += vec
            word_count += 1
    if word_count == 0:
        return None
    return vec_sum / word_count


def mk_context_vec(word_forms):
    fi = word_vecs.get_fi()
    return avg_vec(fi, word_forms)


def get_defn_distance(context_vec, defn):
    en = word_vecs.get_en()
    defn_vec = avg_vec(en, word_tokenize(defn))
    if defn_vec is None or context_vec is None:
        return 7
    return cosine(defn_vec, context_vec)


def disambg(lemmas, context):
    context_vec = mk_context_vec(context)
    best_lemma = None
    best_dist = 8
    for lemma in lemmas:
        dist = get_defn_distance(context_vec, lemma.synset().definition())
        if dist < best_dist:
            best_lemma = lemma
            best_dist = dist
    return best_lemma, best_dist
