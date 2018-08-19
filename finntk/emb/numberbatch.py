import gzip
import logging
from finntk.utils import ResourceMan, urlretrieve
from finntk.vendor.conceptnet5.uri import concept_uri
from gensim.models import KeyedVectors
from shutil import copyfileobj
import os

logger = logging.getLogger(__name__)


class NumberbatchWordVecs(ResourceMan):
    RESOURCE_NAME = "numberbatch-multilingual"

    URL = (
        "https://conceptnet.s3.amazonaws.com/downloads/2017/numberbatch"
        "/numberbatch-17.06.txt.gz"
    )

    def __init__(self):
        super().__init__()
        self._resources["vecs"] = "numberbatch.multi.binvec"
        self._vecs = None

    def _bootstrap(self, _res):
        from gensim.test.utils import get_tmpfile

        logger.info("Downloading word vectors")
        gzipped_glove_tmp_fn = urlretrieve(self.URL)
        try:
            glove_tmp_fn = get_tmpfile("glove.txt")
            try:
                copyfileobj(gzip.open(gzipped_glove_tmp_fn), open(glove_tmp_fn, "wb"))
                logger.info("Converting word vectors")
                fi = KeyedVectors.load_word2vec_format(glove_tmp_fn)
                fi.save(self._get_res_path("vecs"))
            finally:
                try:
                    os.remove(glove_tmp_fn)
                except OSError:
                    pass
        finally:
            try:
                os.remove(gzipped_glove_tmp_fn)
            except OSError:
                pass

    def get_vecs(self):
        if self._vecs is None:
            vec_path = self.get_res("vecs")
            logger.info("Loading word vectors")
            self._vecs = KeyedVectors.load(vec_path, mmap="r")
            logger.info("Loaded word vectors")
        return self._vecs


vecs = NumberbatchWordVecs()


def mk_concept_vec(lang, text, *more):
    return vecs.get_vecs()[concept_uri(lang, text, *more)]
