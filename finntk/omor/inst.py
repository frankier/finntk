"""
Function to get ahold of an OMorFi instance.
"""

_omorfi = None


def get_omorfi():
    """
    Gets an Omorfi instance with everything possible enabled. Reuses the
    existing instance if already called once.
    """
    from omorfi.omorfi import Omorfi

    global _omorfi
    if _omorfi is None:
        _omorfi = Omorfi(use_describe=True)
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
            describe=True,
        )
    return _omorfi
