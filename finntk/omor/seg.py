import re
from more_itertools import split_at

LABELSEGMENT_RE = r"""
    \{ (?P<seg> [^\}]* ) \} |
    \[ (?P<tag> [^\]]* ) \] |
    (?P<surf> [^\[\{]+ )
"""

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
        return (kv[0] == "seg" and kv[1] == "wB")

    return split_at(it, is_cmp_bound)


def tokens_to_surf(it):
    return "".join(v for (t, v) in it if t == "surf")


def labelsegment_to_subword_tokens(labelsegmented):
    return tokens_to_subword_tokens(labelsegment_to_tokens(labelsegmented))
