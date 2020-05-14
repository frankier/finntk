"""
Function to get ahold of an OMorFi instance.
"""

_omorfi = None


FSTS = [
    ("acceptor", "accept.hfst"),
    ("analyser", "describe.hfst"),
    ("generator", "generate.hfst"),
    ("hyphenator", "hyphenate-rules.hfst"),
    ("labelsegmenter", "labelsegment.hfst"),
    ("lemmatiser", "lemmatise.hfst"),
    ("segmenter", "segment.hfst"),
    ("tokeniser", "tokenise.pmatchfst"),
]


def get_omorfi():
    """
    Gets an Omorfi instance with everything possible enabled. Reuses the
    existing instance if already called once.
    """
    from omorfi.omorfi import Omorfi

    global _omorfi
    if _omorfi is None:
        _omorfi = Omorfi()
        for var, fn in FSTS:
            getattr(_omorfi, "load_" + var)(
                "/usr/local/share/omorfi/omorfi." + fn
            )
    return _omorfi
