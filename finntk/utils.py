from os.path import join as pjoin
from appdirs import AppDirs
import os
import gc
from nltk.corpus.util import _make_bound_method


class ResourceMan:
    RESOURCE_NAME = None

    def __init__(self):
        self._dirs = AppDirs("finntk", "Frankie Robertson")
        self._resources = {}

    def _get_data_dir(self):
        if "FINNTK_DATA" in os.environ:
            dir = os.environ["FINNTK_DATA"]
        else:
            dir = self._dirs.user_data_dir
        if self.RESOURCE_NAME:
            dir = pjoin(dir, self.RESOURCE_NAME)
        return dir

    def _ensure_data_dir(self):
        dir = self._get_data_dir()
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir

    def _get_res_filename(self, res):
        return self._resources[res]

    def _get_res_path(self, res):
        data_dir = self._get_data_dir()
        filename = self._get_res_filename(res)
        return pjoin(data_dir, filename)

    def get_res(self, res):
        path = self._get_res_path(res)
        if not os.path.exists(path):
            self._ensure_data_dir()
            self._bootstrap(path)
        return path

    def resource_names(self):
        return self._resources.keys()

    def _is_bootstrapped(self):
        dir = self._get_data_dir()
        return os.path.exists(dir)

    def bootstrap(self, res=None):
        if not self.RESOURCE_NAME:
            assert False
        if not self._is_bootstrapped():
            self._ensure_data_dir()
            self._bootstrap(res)


# Taken from nltk with data file management stuff taken out
class LazyCorpusLoader:

    def __init__(self, name, reader_cls, *args, **kwargs):
        from nltk.corpus.reader.api import CorpusReader

        assert issubclass(reader_cls, CorpusReader)
        self.__name = self.__name__ = name
        self.__reader_cls = reader_cls
        self.__args = args
        self.__kwargs = kwargs

    def __load(self):
        # Load the corpus.
        corpus = self.__reader_cls("", *self.__args, **self.__kwargs)

        # This is where the magic happens!  Transform ourselves into
        # the corpus by modifying our own __dict__ and __class__ to
        # match that of the corpus.

        args, kwargs = self.__args, self.__kwargs
        name, reader_cls = self.__name, self.__reader_cls

        self.__dict__ = corpus.__dict__
        self.__class__ = corpus.__class__

        # _unload support: assign __dict__ and __class__ back, then do GC.
        # after reassigning __dict__ there shouldn't be any references to
        # corpus data so the memory should be deallocated after gc.collect()
        def _unload(self):
            lazy_reader = LazyCorpusLoader(name, reader_cls, *args, **kwargs)
            self.__dict__ = lazy_reader.__dict__
            self.__class__ = lazy_reader.__class__
            gc.collect()

        self._unload = _make_bound_method(_unload, self)

    def __getattr__(self, attr):

        # Fix for inspect.isclass under Python 2.6
        # (see http://bugs.python.org/issue1225107).
        # Without this fix tests may take extra 1.5GB RAM
        # because all corpora gets loaded during test collection.
        if attr == "__bases__":
            raise AttributeError("LazyCorpusLoader object has no attribute '__bases__'")

        self.__load()
        # This looks circular, but its not, since __load() changes our
        # __class__ to something new:
        return getattr(self, attr)

    def __repr__(self):
        return "<%s in %r (not loaded yet)>" % (
            self.__reader_cls.__name__, ".../corpora/" + self.__name
        )

    def _unload(self):
        # If an exception occures during corpus loading then
        # '_unload' method may be unattached, so __getattr__ can be called;
        # we shouldn't trigger corpus loading again in this case.
        pass


def urlretrieve(url):
    if os.environ.get("OVERRIDE_FETCHES"):
        return input("Enter path for local copy of {}: ".format(url))
    else:
        from urllib.request import urlretrieve

        result, _ = urlretrieve(url)
        return result
