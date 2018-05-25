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
    """
    Returns a list of list of dicts. Each list element is an analysis. For each
    analysis, there is a list of subwords. Each dict contains an Omorfi
    analysis
    """
    return map(pairs_to_dict, chunk_subwords(analysis_to_pairs(ana)))
