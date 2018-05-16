import re
from omorfi.omorfi import Omorfi
from more_itertools import split_at


def analysis_to_pairs(ana):
    for bit in ana.split(']['):
        k, v = bit.strip('[]').split('=', 1)
        yield k, v


def pairs_to_dict(it):
    return dict(((k.lower(), v) for k, v in it))


def analysis_to_dict(ana):
    return pairs_to_dict(analysis_to_pairs(ana))


def chunk_subwords(it):
    def is_cmp_bound(kv):
        return (kv[0] == 'BOUNDARY' and
                kv[1] == 'COMPOUND')

    return split_at(it, is_cmp_bound)


def analysis_to_subword_dicts(ana):
    return map(pairs_to_dict, chunk_subwords(analysis_to_pairs(ana)))


LABELSEGMENT_RE = r'''
    \{ (?P<seg> [^\}]* ) \} |
    \[ (?P<tag> [^\]]* ) \] |
    (?P<surf> [^\[\{]+ )
'''

_labelsegment_lex = None


def get_labelsegment_lex():
    global _labelsegment_lex
    if _labelsegment_lex is None:
        _labelsegment_lex = re.compile(LABELSEGMENT_RE, re.VERBOSE)
    return _labelsegment_lex


def labelsegment_to_tokens(labelsegmented):
    lex = get_labelsegment_lex()
    for match in re.finditer(lex, labelsegmented):
        typ = match.lastgroup
        value = match.group(typ)
        yield typ, value


def tokens_to_subword_tokens(it):
    def is_cmp_bound(kv):
        return (kv[0] == 'seg' and
                kv[1] == 'wB')

    return split_at(it, is_cmp_bound)


def tokens_to_surf(it):
    return "".join(v for (t, v) in it if t == 'surf')


def labelsegment_to_subword_tokens(labelsegmented):
    return tokens_to_subword_tokens(labelsegment_to_tokens(labelsegmented))


def get_token_positions(tokenised, text):
    starts = []
    start = 0
    for token in tokenised:
        start = text.index(token['surf'], start)
        starts.append(start)
    return starts


_omorfi = None


def get_omorfi():
    global _omorfi
    if _omorfi is None:
        _omorfi = Omorfi()
        _omorfi.load_from_dir(
            analyse=True,
            generate=True,
            accept=True,
            tokenise=True,
            lemmatise=True,
            hyphenate=True,
            segment=True,
            labelsegment=True,
            guesser=True,
            udpipe=True)
    return _omorfi
