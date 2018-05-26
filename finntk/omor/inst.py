"""
Function to get ahold of an OMorFi instance.
"""

from omorfi.omorfi import Omorfi

_omorfi = None


def get_omorfi():
    """
    Gets an Omorfi instance with everything possible enabled. Reuses the
    existing instance if already called once.
    """
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
            udpipe=True,
        )
    return _omorfi
