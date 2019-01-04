import logging
import os

logger = logging.getLogger(__name__)


class BertWordVecs:

    def __init__(self):
        super().__init__()
        self._model = None

    def get(self):
        from finntk.vendor.bert import get_bert

        if self._model is None:
            logger.info("Loading BERT")
            use_cuda = not os.environ.get("DISABLE_CUDA")
            self._model = get_bert("bert-base-multilingual-cased", use_cuda=use_cuda)
            logger.info("Loaded BERT")
        return self._model


vecs = BertWordVecs()
