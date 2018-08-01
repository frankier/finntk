from nltk.util import binary_search_file
import zipfile
import logging
from finntk.utils import ResourceMan, urlretrieve
from gensim.models import KeyedVectors
from shutil import copyfileobj
import os

logger = logging.getLogger(__name__)


class AutoExtendNumberBatchFiWNWordVecs(ResourceMan):
    RESOURCE_NAME = "autoextend-fi"

    URL = (
        "https://github.com/frankier/AutoExtend/releases/download/"
        + "fiwn-conceptnet/fiwn-conceptnet.zip"
    )

    def __init__(self):
        super().__init__()
        self._resources["vecs"] = "fiwn.numberbatch.binvec"
        self._resources["synsets"] = "synsets.txt"
        self._vecs = None

    def _bootstrap(self, res):
        from gensim.test.utils import get_tmpfile

        logger.info("Downloading FiWN ConceptNet word vectors")
        zipped_tmp_fn = urlretrieve(self.URL)
        try:
            tmp_zip = zipfile.ZipFile(zipped_tmp_fn)
            tmp_fn = get_tmpfile("fiwn-conceptnet.txt")
            try:
                copyfileobj(tmp_zip.open("outputVectors.txt"), open(tmp_fn, "wb"))
                logger.info("Converting FiWN ConceptNet word vectors")
                fi = KeyedVectors.load_word2vec_format(tmp_fn)
                fi.save(self._get_res_path("vecs"))
            finally:
                os.remove(tmp_fn)
            copyfileobj(
                tmp_zip.open("synsets.txt"), open(self._get_res_path("synsets"), "wb")
            )
        finally:
            os.remove(zipped_tmp_fn)

    def get_vecs(self):
        if self._vecs is None:
            vec_path = self.get_res("vecs")
            logger.info("Loading word vectors")
            self._vecs = KeyedVectors.load(vec_path, mmap="r")
            logger.info("Loaded word vectors")
        return self._vecs


vecs = AutoExtendNumberBatchFiWNWordVecs()


class SynsetMapper:

    def __init__(self, res_man):
        self.res_man = res_man
        self._map_file = None

    @property
    def map_file(self):
        if self._map_file is None:
            synsets_fn = self.res_man.get_res("synsets")
            self._map_file = open(synsets_fn)
        return self._map_file

    def __call__(self, synset_id):
        line = binary_search_file(self.map_file, synset_id)
        bits = line.split(" ", 1)
        return bits[1] or None


synset_map = SynsetMapper(vecs)


def mk_lemma_vec(lemma):
    from finntk.wordnet.reader import fiwn

    synset = lemma.synset()
    fiwn_space = vecs.get_vecs()
    lemma_id = "{}-wn-fi-2.0-{}".format(lemma.name(), fiwn.ss2of(synset))
    return fiwn_space[lemma_id]


def mk_synset_vec(synset):
    fiwn_space = vecs.get_vecs()
    synset_id = "".join((lemma.key() + "," for lemma in synset.lemmas()))
    return fiwn_space[synset_id]
