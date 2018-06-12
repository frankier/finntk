from finntk.utils import ResourceMan, LazyCorpusLoader
from os.path import join as pjoin
import os
from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import (
    WordNetCorpusReader,
    Synset,
    Lemma,
    VERB_FRAME_STRINGS,
    ADJ_SAT,
    WordNetError,
)
from urllib.request import urlretrieve
import logging
import zipfile
import shutil
import re

logger = logging.getLogger(__name__)


class FinnWordNetResMan(ResourceMan):
    RESOURCE_NAME = "fiwn"
    RELS = ["transls"]

    URL_BASE = (
        "http://www.ling.helsinki.fi/kieliteknologia/tutkimus/finnwordnet"
        "/download_files/"
    )
    DICT_URL = URL_BASE + "fiwn_dict_fi-2.0.zip"
    RELS_URL = URL_BASE + "fiwn_rels_fi-2.0.zip"

    def __init__(self):
        super().__init__()
        for name in self.RELS:
            self._resources[name] = pjoin("rels", "fiwn-{}.tsv".format(name))

    def _get_res_filename(self, res):
        return self._resources.get(res, pjoin("dict", res))

    def _download(self, part, url, dest):
        logger.info("Downloading FinnWordNet {}".format(part))
        tmp_fn, headers = urlretrieve(url)
        assert headers["Content-Type"] == "application/zip"
        data_dir = self._get_data_dir()
        try:
            zip = zipfile.ZipFile(tmp_fn, "r")
            zip.extractall(data_dir)
        finally:
            os.remove(tmp_fn)
        shutil.move(pjoin(data_dir, "fiwn-2.0", part), data_dir)

    def _bootstrap(self, _res):
        self._download("dict", self.DICT_URL, self._get_data_dir())
        self._download("rels", self.RELS_URL, self._get_data_dir())


fiwn_resman = FinnWordNetResMan()


class FinnWordNetReader(WordNetCorpusReader):

    def __init__(self, *args, **kwargs):
        fiwn_resman.bootstrap()
        args = list(args)
        args[0] = fiwn_resman.get_res("")
        super().__init__(*args, **kwargs)

    def open(self, filename):
        if filename == "lexnames":
            return wordnet.open("lexnames")
        return super().open(filename)

    def _synset_from_pos_and_line(self, pos, data_file_line):
        # Copied and modified for FiWN. The only change is the handling of
        # FiWN's special <tag/> syntactic markers.

        # Construct a new (empty) synset.
        synset = Synset(self)

        # parse the entry for this synset
        try:

            # parse out the definitions and examples from the gloss
            columns_str, gloss = data_file_line.split("|")
            gloss = gloss.strip()
            definitions = []
            for gloss_part in gloss.split(";"):
                gloss_part = gloss_part.strip()
                if gloss_part.startswith('"'):
                    synset._examples.append(gloss_part.strip('"'))
                else:
                    definitions.append(gloss_part)
            synset._definition = "; ".join(definitions)

            # split the other info into fields
            _iter = iter(columns_str.split())

            def _next_token():
                return next(_iter)

            # get the offset
            synset._offset = int(_next_token())

            # determine the lexicographer file name
            lexname_index = int(_next_token())
            synset._lexname = self._lexnames[lexname_index]

            # get the part of speech
            synset._pos = _next_token()

            # create Lemma objects for each lemma
            n_lemmas = int(_next_token(), 16)
            for _ in range(n_lemmas):
                # get the lemma name
                lemma_name = _next_token()
                # get the lex_id (used for sense_keys)
                lex_id = int(_next_token(), 16)
                # If the lemma has a syntactic marker, extract it.
                m = re.match(r"([^<]*)(<.*)?$", lemma_name)
                lemma_name, syn_mark = m.groups()
                # create the lemma object
                lemma = Lemma(self, synset, lemma_name, lexname_index, lex_id, syn_mark)
                synset._lemmas.append(lemma)
                synset._lemma_names.append(lemma._name)

            # collect the pointer tuples
            n_pointers = int(_next_token())
            for _ in range(n_pointers):
                symbol = _next_token()
                offset = int(_next_token())
                pos = _next_token()
                lemma_ids_str = _next_token()
                if lemma_ids_str == "0000":
                    synset._pointers[symbol].add((pos, offset))
                else:
                    source_index = int(lemma_ids_str[:2], 16) - 1
                    target_index = int(lemma_ids_str[2:], 16) - 1
                    source_lemma_name = synset._lemmas[source_index]._name
                    lemma_pointers = synset._lemma_pointers
                    tups = lemma_pointers[source_lemma_name, symbol]
                    tups.add((pos, offset, target_index))

            # read the verb frames
            try:
                frame_count = int(_next_token())
            except StopIteration:
                pass
            else:
                for _ in range(frame_count):
                    # read the plus sign
                    plus = _next_token()
                    assert plus == "+"
                    # read the frame and lemma number
                    frame_number = int(_next_token())
                    frame_string_fmt = VERB_FRAME_STRINGS[frame_number]
                    lemma_number = int(_next_token(), 16)
                    # lemma number of 00 means all words in the synset
                    if lemma_number == 0:
                        synset._frame_ids.append(frame_number)
                        for lemma in synset._lemmas:
                            lemma._frame_ids.append(frame_number)
                            lemma._frame_strings.append(frame_string_fmt % lemma._name)
                    # only a specific word in the synset
                    else:
                        lemma = synset._lemmas[lemma_number - 1]
                        lemma._frame_ids.append(frame_number)
                        lemma._frame_strings.append(frame_string_fmt % lemma._name)

        # raise a more informative error with line text
        except ValueError as e:
            raise WordNetError("line %r: %s" % (data_file_line, e))

        # set sense keys for Lemma objects - note that this has to be
        # done afterwards so that the relations are available
        for lemma in synset._lemmas:
            if synset._pos == ADJ_SAT:
                head_lemma = synset.similar_tos()[0]._lemmas[0]
                head_name = head_lemma._name
                head_id = "%02d" % head_lemma._lex_id
            else:
                head_name = head_id = ""
            tup = (
                lemma._name,
                WordNetCorpusReader._pos_numbers[synset._pos],
                lemma._lexname_index,
                lemma._lex_id,
                head_name,
                head_id,
            )
            lemma._key = ("%s%%%d:%02d:%02d:%s:%s" % tup).lower()

        # the canonical name is based on the first lemma
        lemma_name = synset._lemmas[0]._name.lower()
        offsets = self._lemma_pos_offset_map[lemma_name][synset._pos]
        sense_index = offsets.index(synset._offset)
        tup = lemma_name, synset._pos, sense_index + 1
        synset._name = "%s.%s.%02i" % tup

        return synset


fiwn = LazyCorpusLoader("fiwn", FinnWordNetReader, None)


def get_transl_iter():
    transls = fiwn_resman.get_res("transls")
    for line in open(transls):
        fi_synset, fi_lemma, en_synset, en_lemma, rel, extra = line[:-1].split("\t")
        _, fi_synset = fi_synset.split(":", 1)
        _, en_synset = en_synset.split(":", 1)
        assert fi_synset == en_synset
        yield fi_synset, fi_lemma, en_lemma, rel, extra
