from omorfi.omorfi import Omorfi


def analysis_to_dict(ana):
    d = {}
    for bit in ana.split(']['):
        k, v = bit.strip('[]').split('=', 1)
        d[k.lower()] = v
    return d


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
