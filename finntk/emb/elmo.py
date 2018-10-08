import tarfile
import logging
from finntk.utils import ResourceMan, urlretrieve
import os

logger = logging.getLogger(__name__)


class ElmoWordVecs(ResourceMan):
    RESOURCE_NAME = "elmo-fi"

    URL = "http://pbmpb9h15.bkt.gdipper.com/fi.model.tar.xz"

    def __init__(self):
        super().__init__()
        self._model = None

    def _get_res_filename(self, res):
        return res

    def _bootstrap(self, res):
        logger.info("Downloading Elmo word vectors")
        tarred_tmp_fn = urlretrieve(self.URL)
        try:
            tmp_tar = tarfile.open(tarred_tmp_fn, "r:xz")
            tmp_tar.extractall(self.get_res(""))
        finally:
            os.remove(tarred_tmp_fn)

    def get(self):
        from finntk.vendor.elmo import get_elmo

        if self._model is None:
            logger.info("Loading ELMo")
            use_cuda = not os.environ.get("DISABLE_CUDA")
            self._model = get_elmo(self.get_res(""), use_cuda=use_cuda)
            logger.info("Loaded ELMo")
        return self._model


vecs = ElmoWordVecs()
