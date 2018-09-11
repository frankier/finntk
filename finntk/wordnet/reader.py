from finntk.utils import ResourceMan, LazyCorpusLoader
from os.path import exists, join as pjoin
from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import (
    WordNetCorpusReader,
    Synset,
    Lemma,
    VERB_FRAME_STRINGS,
    ADJ,
    ADJ_SAT,
    WordNetError,
)
import logging
import re
from plumbum.cmd import git
import portalocker

logger = logging.getLogger(__name__)


# This lemma is broken in data.noun
LEMMA_NAME_FIXES = {"moderniusAdditional_synonym": "modernius"}

# These lemmas occur in differently upper/lower case wise in transls and data.POS
IGNORE_CASE_LEMMAS = {"ci", "otc", "rh-negatiivinen", "rh-positiivinen"}


class FinnWordNetResMan(ResourceMan):
    RESOURCE_NAME = "fiwn"
    RELS = ["transls"]

    REPO = "https://github.com/frankier/fiwn.git"

    def __init__(self):
        super().__init__()
        for name in self.RELS:
            self._resources[name] = pjoin("data", "rels", "fiwn-{}.tsv".format(name))
        self._resources["synset_map"] = "synset_map.tsv"

    def _get_res_filename(self, res):
        return self._resources.get(res, pjoin("data", "dict", res))

    @property
    def _done_file(self):
        data_dir = self._get_data_dir()
        return pjoin(data_dir, ".done")

    def _is_bootstrapped(self):
        return exists(self._done_file)

    def _bootstrap(self, _res=None):
        data_dir = self._get_data_dir()
        with portalocker.Lock(pjoin(data_dir, "../fiwn.lock"), timeout=3600):
            if not self._is_bootstrapped():
                git("clone", self.REPO, data_dir)
                with open(self._done_file, "w"):
                    pass


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
                    tups.append((pos, offset, target_index))

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

    def _load_lemma_pos_offset_map(self):
        # Copied and modified for FiWN. The only changes are to handle a single
        # entry: FiWN contains an empty lemma for `taken` as in `taken
        # ill/drunk`. Thus lines beginning with two spaces as are treated as a
        # comment instead of ones starting with one space and tokenisation is
        # done on spaces instead of all whitespace.

        for suffix in self._FILEMAP.values():

            # parse each line of the file (ignoring comment lines)
            for i, line in enumerate(self.open("index.%s" % suffix)):
                if line.startswith("  "):
                    continue

                _iter = iter(line.split(" "))

                def _next_token():
                    return next(_iter)

                try:

                    # get the lemma and part-of-speech
                    lemma = _next_token()
                    pos = _next_token()

                    # get the number of synsets for this lemma
                    n_synsets = int(_next_token())
                    assert n_synsets > 0

                    # get and ignore the pointer symbols for all synsets of
                    # this lemma
                    n_pointers = int(_next_token())
                    [_next_token() for _ in range(n_pointers)]

                    # same as number of synsets
                    n_senses = int(_next_token())
                    assert n_synsets == n_senses

                    # get and ignore number of senses ranked according to
                    # frequency
                    _next_token()

                    # get synset offsets
                    synset_offsets = [int(_next_token()) for _ in range(n_synsets)]

                # raise more informative error with file name and line number
                except (AssertionError, ValueError) as e:
                    tup = ("index.%s" % suffix), (i + 1), e
                    raise WordNetError("file %s, line %i: %s" % tup)

                # map lemmas and parts of speech to synsets
                self._lemma_pos_offset_map[lemma][pos] = synset_offsets
                if pos == ADJ:
                    self._lemma_pos_offset_map[lemma][ADJ_SAT] = synset_offsets


fiwn = LazyCorpusLoader("fiwn", FinnWordNetReader, None)


class CountCachingMixin:

    def __init__(self, *args, **kwargs) -> None:
        self._count_cache = {}
        super().__init__(*args, **kwargs)

    def lemma_count(self, lemma: Lemma) -> int:
        if lemma._key not in self._count_cache:
            self._count_cache[lemma._key] = super().lemma_count(lemma)
        return self._count_cache[lemma._key]


class EnCountsFinnWordNetReader(CountCachingMixin, FinnWordNetReader):

    def lemmas(self, lemma, pos=None, lang="eng"):
        lemmas = super().lemmas(lemma, pos, lang)
        return sorted(lemmas, key=lambda l: -l.count())


fiwn_encnt = LazyCorpusLoader("fiwn", EnCountsFinnWordNetReader, None)


en_fi_maps = None


def get_en_fi_maps():

    def fix_synset_key(synset):
        synset = synset.split(":", 1)[1]
        if synset[0] == "s":
            synset = "a" + synset[1:]
        return synset

    global en_fi_maps
    if en_fi_maps is None:
        synset_map = fiwn_resman.get_res("synset_map")
        fi2en = {}
        en2fi = {}
        for line in open(synset_map):
            en_synset_key, fi_synset_key = line[:-1].split("\t")
            en_synset_key = fix_synset_key(en_synset_key)
            fi_synset_key = fix_synset_key(fi_synset_key)
            fi2en[fi_synset_key] = en_synset_key
            en2fi[en_synset_key] = fi_synset_key
        en_fi_maps = fi2en, en2fi
    return en_fi_maps
