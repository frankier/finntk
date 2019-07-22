from tempfile import TemporaryDirectory
from finntk.utils import ResourceMan, urlretrieve
import os

# If we don't ensure this, we might end up with a relative path for LEMMA_FILENAME
os.makedirs(os.path.expanduser("~/.conceptnet5"), exist_ok=True)
from conceptnet5.language.lemmatize import LEMMA_FILENAME  # noqa


class ConceptNetWiktionaryResMan(ResourceMan):
    RESOURCE_NAME = "conceptnet_wiktionary"

    URL = "https://archive.org/download/wiktionary.db/wiktionary.db.gz"

    def __init__(self):
        self._cached_is_bootstrapped = None
        super().__init__()

    def _is_bootstrapped(self):
        if self._cached_is_bootstrapped is not None:
            return self._cached_is_bootstrapped
        is_boostrapped = os.path.exists(LEMMA_FILENAME)
        self._cached_is_bootstrapped = is_boostrapped
        return is_boostrapped

    def _bootstrap(self, _res=None):
        from plumbum.cmd import gunzip

        tempdir = TemporaryDirectory()
        en_wiktionary_gz = urlretrieve(
            self.URL, filename=os.path.join(tempdir.name, "enwiktionary.gz")
        )
        try:
            gunzip(en_wiktionary_gz, LEMMA_FILENAME)
        finally:
            os.remove(en_wiktionary_gz)
            tempdir.cleanup()
        self._cached_is_bootstrapped = True


conceptnet_wiktionary = ConceptNetWiktionaryResMan()
