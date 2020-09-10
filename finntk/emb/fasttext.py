import logging
from finntk.utils import ResourceMan, urlretrieve
import os
from .base import MultilingualVectorSpace, RefType
from .utils import get, load_word2vec_format, load

logger = logging.getLogger(__name__)


class FasttextWordVecs(ResourceMan):
    RESOURCE_NAME = "fasttext-multilingual"

    FI_URL = "https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.fi.vec"
    EN_URL = "https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.en.vec"

    def __init__(self):
        super().__init__()
        self._resources["fi_vec"] = "wiki.multi.fi.binvec"
        self._resources["en_vec"] = "wiki.multi.en.binvec"
        self._en = None
        self._fi = None

    def _download(self, lang, url, dest):
        logger.info("Downloading {} word vectors".format(lang))
        tmp_fn = urlretrieve(url)
        try:
            logger.info("Converting {} word vectors".format(lang))
            fi = load_word2vec_format(tmp_fn)
            fi.save(dest)
        finally:
            os.remove(tmp_fn)

    def _bootstrap(self, _res):
        self._download("Finnish", self.FI_URL, self._get_res_path("fi_vec"))
        self._download("English", self.EN_URL, self._get_res_path("en_vec"))

    def get_en(self):
        if self._en is None:
            en_vec_path = self.get_res("en_vec")
            logger.info("Loading English word vectors")
            self._en = load(en_vec_path, mmap="r")
            logger.info("Loaded English word vectors")
        return self._en

    def get_fi(self):
        if self._fi is None:
            fi_vec_path = self.get_res("fi_vec")
            logger.info("Loading Finnish word vectors")
            self._fi = load(fi_vec_path, mmap="r")
            logger.info("Loaded Finnish word vectors")
        return self._fi


vecs = FasttextWordVecs()


class FasttextMultiSpace(MultilingualVectorSpace):
    takes = RefType.WORD_FORM
    dim = 300

    def get_vec(self, lang: str, ref: str):
        if lang == "en":
            space = vecs.get_en()
        elif lang == "fi":
            space = vecs.get_fi()
        else:
            assert False, "Unsupported language"
        return get(space, ref)


multispace = FasttextMultiSpace()
