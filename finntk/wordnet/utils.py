from nltk.corpus.reader.wordnet import Synset, WordNetError
from typing import Optional


def pre_id_to_post(pre_id):
    """
    Convert a synset id of the format n88888888 to 88888888-n
    """
    return "{}-{}".format(pre_id[1:], pre_id[0])


def post_id_to_pre(post_id, conv_s=False):
    """
    Convert a synset id of the format 88888888-n to n88888888
    """
    pos = post_id[-1]
    if conv_s and pos == "s":
        pos = "a"
    return "{}{}".format(pos, post_id[:-2])


def ss2pre(ss):
    pos = ss.pos()
    if pos == "s":
        pos = "a"
    return ("{}{:08d}".format(pos, ss.offset()))


def pre2ss(wn, pre):
    return wn.synset_from_pos_and_offset(pre[0], int(pre[1:]))


def fi2en_post(post_id):
    from .reader import get_en_fi_maps

    fi2en, en2fi = get_en_fi_maps()
    return pre_id_to_post(fi2en[post_id_to_pre(post_id, conv_s=True)])


def en2fi_post(post_id):
    from .reader import get_en_fi_maps

    fi2en, en2fi = get_en_fi_maps()
    return pre_id_to_post(en2fi[post_id_to_pre(post_id, conv_s=True)])


def maybe_fi2en_ss(ss: Synset) -> Optional[Synset]:
    from .reader import get_en_fi_maps
    from .utils import ss2pre, pre2ss
    from nltk.corpus import wordnet

    fi2en, _en2fi = get_en_fi_maps()
    pre_fi = ss2pre(ss)
    pre_en = fi2en.get(pre_fi)
    if pre_en is None:
        return None
    try:
        return pre2ss(wordnet, pre_en)
    except WordNetError:
        return None
