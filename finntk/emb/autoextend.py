from nltk.util import binary_search_file
import zipfile
import logging
from functools import total_ordering
from finntk.utils import ResourceMan, urlretrieve
from finntk.wordnet.reader import fiwn
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


@total_ordering
class AsKey(object):

    def __init__(self, obj, key_func):
        self.obj = obj
        self.obj_key = key_func(obj)
        self.key_func = key_func

    def __lt__(self, other):
        return self.obj_key < self.key_func(other)

    def __eq__(self, other):
        return self.obj_key == self.key_func(other)

    def __add__(self, other):
        return AsKey(self.obj + other.encode("utf-8"), self.key_func)

    def __len__(self):
        return len(self.obj)


POS_ORDER = b"nvar"


def synset_map_key(line):
    key = line.split(b" ", 1)[0]
    _, off, pos = key.strip().rsplit(b"-", 2)
    return (POS_ORDER.index(pos), off)


class SynsetMapper:

    def __init__(self, res_man):
        self.res_man = res_man
        self._map_file = None

    @property
    def map_file(self):
        if self._map_file is None:
            synsets_fn = self.res_man.get_res("synsets")
            self._map_file = open(synsets_fn, "rb")
        return self._map_file

    def __call__(self, synset_id):
        full_synset_id = "wn-fi-2.0-" + synset_id
        line = binary_search_file(
            self.map_file, AsKey(full_synset_id.encode("utf-8"), synset_map_key)
        )
        if line is None:
            return
        bits = line.split(b" ", 1)
        return bits[1].rstrip().decode("utf-8") or None


synset_map = SynsetMapper(vecs)


def get_lemma_id(lemma):
    synset = lemma.synset()
    return "{}-wn-fi-2.0-{}".format(lemma.name().lower(), fiwn.ss2of(synset))


def mk_lemma_vec(lemma):
    fiwn_space = vecs.get_vecs()
    return fiwn_space[get_lemma_id(lemma)]


def mk_lemmas_mat(lemmas):
    fiwn_space = vecs.get_vecs()
    return fiwn_space[[get_lemma_id(lemma) for lemma in lemmas]]


def mk_synset_vec(synset):
    fiwn_space = vecs.get_vecs()
    synset_id = synset_map(fiwn.ss2of(synset))
    if synset_id is None:
        return
    return fiwn_space[synset_id]
