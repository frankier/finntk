import logging
from finntk.utils import ResourceMan, urlretrieve
from gensim.models import KeyedVectors
from shutil import copyfileobj
import os
import zipfile
from .base import MonolingualVectorSpace, RefType
from .utils import get

logger = logging.getLogger(__name__)


class Word2VecWordVecs(ResourceMan):
    RESOURCE_NAME = "word2vec-fi"

    URL = "http://vectors.nlpl.eu/repository/11/42.zip"

    def __init__(self):
        super().__init__()
        self._resources["vecs"] = "word2vec.binvec"
        self._vecs = None

    def _bootstrap(self, res):
        from gensim.test.utils import get_tmpfile

        logger.info("Downloading Word2Vec word vectors")
        zipped_tmp_fn = urlretrieve(self.URL)
        try:
            tmp_zip = zipfile.ZipFile(zipped_tmp_fn)
            tmp_fn = get_tmpfile("word2vec-fi.txt")
            try:
                copyfileobj(tmp_zip.open("model.txt"), open(tmp_fn, "wb"))
                logger.info("Converting Word2Vec word vectors")
                fi = KeyedVectors.load_word2vec_format(tmp_fn, unicode_errors="replace")
                fi.save(self._get_res_path("vecs"))
            finally:
                os.remove(tmp_fn)
        finally:
            os.remove(zipped_tmp_fn)

    def get_vecs(self):
        if self._vecs is None:
            vec_path = self.get_res("vecs")
            logger.info("Loading word vectors")
            self._vecs = KeyedVectors.load(vec_path, mmap="r")
            logger.info("Loaded word vectors")
        return self._vecs


vecs = Word2VecWordVecs()


class Word2VecFiSpace(MonolingualVectorSpace):
    takes = RefType.WORD_FORM
    dim = 100

    def get_vec(self, ref: str):
        return get(vecs.get_vecs(), ref)


space = Word2VecFiSpace()
