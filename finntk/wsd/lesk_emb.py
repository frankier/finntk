import gzip
from scipy.spatial.distance import cosine
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from gensim.models import KeyedVectors
import logging
from urllib.request import urlretrieve
import os
from finntk.utils import ResourceMan
from shutil import copyfileobj

logger = logging.getLogger(__name__)


class FasttextWordVecs(ResourceMan):
    RESOURCE_NAME = "fasttext-multilingual"

    FI_URL = "https://s3.amazonaws.com/arrival/embeddings/wiki.multi.fi.vec"
    EN_URL = "https://s3.amazonaws.com/arrival/embeddings/wiki.multi.en.vec"

    def __init__(self):
        super().__init__()
        self._resources["fi_vec"] = "wiki.multi.fi.binvec"
        self._resources["en_vec"] = "wiki.multi.en.binvec"
        self._en = None
        self._fi = None

    def _download(self, lang, url, dest):
        logger.info("Downloading {} word vectors".format(lang))
        tmp_fn, _ = urlretrieve(url)
        try:
            logger.info("Converting {} word vectors".format(lang))
            fi = KeyedVectors.load_word2vec_format(tmp_fn)
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
            self._en = KeyedVectors.load(en_vec_path, mmap="r")
            logger.info("Loaded English word vectors")
        return self._en

    def get_fi(self):
        if self._fi is None:
            fi_vec_path = self.get_res("fi_vec")
            logger.info("Loading Finnish word vectors")
            self._fi = KeyedVectors.load(fi_vec_path, mmap="r")
            logger.info("Loaded Finnish word vectors")
        return self._fi


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
        try:
            gzipped_glove_tmp_fn, _ = urlretrieve(self.URL)
            glove_tmp_fn = get_tmpfile("glove.txt")
            copyfileobj(gzip.open(gzipped_glove_tmp_fn), open(glove_tmp_fn, "wb"))
            logger.info("Converting word vectors")
            fi = KeyedVectors.load_word2vec_format(glove_tmp_fn)
            fi.save(self._get_res_path("vecs"))
        finally:
            try:
                os.remove(gzipped_glove_tmp_fn)
            except OSError:
                pass
            try:
                os.remove(glove_tmp_fn)
            except OSError:
                pass

    def get_vecs(self):
        if self._vecs is None:
            vec_path = self.get_res("vecs")
            logger.info("Loading word vectors")
            self._vecs = KeyedVectors.load(vec_path, mmap="r")
            logger.info("Loaded word vectors")
        return self._vecs


fasttext_word_vecs = FasttextWordVecs()
numberbatch_word_vecs = NumberbatchWordVecs()


def avg_vec_from_vecs(vecs):
    word_count = 0
    vec_sum = 0
    for vec in vecs:
        vec_sum += vec
        word_count += 1
    if word_count == 0:
        return None
    return vec_sum / word_count


def avg_vec(space, stream):

    def gen():
        for word in stream:
            try:
                yield space.get_vector(word)
            except KeyError:
                continue

    return avg_vec_from_vecs(gen())


def mk_context_vec_fasttext_fi(word_forms):
    fi = fasttext_word_vecs.get_fi()
    return avg_vec(fi, word_forms)


def mk_context_vec_conceptnet_fi(lemmas):
    vecs = numberbatch_word_vecs.get_vecs()
    return avg_vec(vecs, ("/c/fi/" + lemma for lemma in lemmas))


def mk_defn_vec(vecs, proc_tok, lemma, wn_filter=False):
    return avg_vec(
        vecs,
        (
            proc_tok(tok.lower())
            for tok in word_tokenize(lemma.synset().definition())
            if not wn_filter or len(wordnet.lemmas(tok.lower()))
        ),
    )


def mk_defn_vec_fasttext_en(lemma, wn_filter=False):
    en = fasttext_word_vecs.get_en()
    return mk_defn_vec(en, lambda x: x, lemma, wn_filter=wn_filter)


def mk_defn_vec_conceptnet_en(lemma, wn_filter=False):
    vecs = numberbatch_word_vecs.get_vecs()
    return mk_defn_vec(vecs, lambda x: "/c/en/" + x, lemma, wn_filter=wn_filter)


def get_defn_distance(context_vec, defn_vec):
    if defn_vec is None or context_vec is None:
        return 7
    return cosine(defn_vec, context_vec)


def disambg(lemma_defns, context_vec):
    best_lemma = None
    best_dist = 8
    for lemma, defn_vec in lemma_defns:
        dist = get_defn_distance(context_vec, defn_vec)
        if dist < best_dist:
            best_lemma = lemma
            best_dist = dist
    return best_lemma, best_dist


def disambg_fasttext(lemmas, context, wn_filter=False):
    return disambg(
        (
            (lemma, mk_defn_vec_fasttext_en(lemma, wn_filter=wn_filter))
            for lemma in lemmas
        ),
        mk_context_vec_fasttext_fi(context),
    )


def disambg_conceptnet(lemmas, context, wn_filter=False):
    return disambg(
        (
            (lemma, mk_defn_vec_conceptnet_en(lemma, wn_filter=wn_filter))
            for lemma in lemmas
        ),
        mk_context_vec_conceptnet_fi(context),
    )
