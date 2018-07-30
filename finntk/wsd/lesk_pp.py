import gzip
from gensim.models import KeyedVectors
from finntk.utils import ResourceMan
import logging
from .lesk_emb import avg_vec_from_vecs
from finntk.wordnet.reader import fiwn
from urllib.request import urlretrieve
import os
from shutil import copyfileobj

logger = logging.getLogger(__name__)


class AutoExtendNumberBatchFiWNWordVecs(ResourceMan):
    RESOURCE_NAME = "autoextend-fi"

    URL = (
        "https://github.com/frankier/AutoExtend/releases/download/"
        + "fiwn-conceptnet/fiwn-conceptnet.txt.gz"
    )

    def __init__(self):
        super().__init__()
        self._resources["vecs"] = "fiwn.numberbatch.binvec"
        self._vecs = None

    def _bootstrap(self, res):
        from gensim.test.utils import get_tmpfile

        logger.info("Downloading FiWN ConceptNet word vectors")
        gzipped_tmp_fn, _ = urlretrieve(self.URL)
        try:
            tmp_fn = get_tmpfile("fiwn-conceptnet.txt")
            copyfileobj(gzip.open(gzipped_tmp_fn), open(tmp_fn, "wb"))
            logger.info("Converting FiWN ConceptNet word vectors")
            fi = KeyedVectors.load_word2vec_format(tmp_fn)
            fi.save(self._get_res_path("vecs"))
        finally:
            os.remove(tmp_fn)

    def get_vecs(self):
        if self._vecs is None:
            vec_path = self.get_res("vecs")
            logger.info("Loading word vectors")
            self._vecs = KeyedVectors.load(vec_path, mmap="r")
            logger.info("Loaded word vectors")
        return self._vecs


fiwn_vecs = AutoExtendNumberBatchFiWNWordVecs()


def cosine_sim(u, v):
    from scipy.spatial.distance import cosine

    return 1 - cosine(u, v)


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


def mk_lemma_vec(lemma):
    synset = lemma.synset()
    fiwn_space = fiwn_vecs.get_vecs()
    lemma_id = "{}-wn-fi-2.0-{}".format(lemma.name(), fiwn.ss2of(synset))
    return fiwn_space.get_vector(lemma_id)
