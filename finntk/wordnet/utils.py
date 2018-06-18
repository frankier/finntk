def pre_id_to_post(pre_id):
    """
    Convert a synset id of the format n88888888 to 88888888-n
    """
    return "{}-{}".format(pre_id[1:], pre_id[0])


def post_id_to_pre(post_id):
    """
    Convert a synset id of the format 88888888-n to n88888888
    """
    return "{}{}".format(post_id[-1], post_id[:-2])


def ss2pre(ss):
    pos = ss.pos()
    if pos == "s":
        pos = "a"
    return ("{}{:08d}".format(pos, ss.offset()))


def pre2ss(wn, pre):
    return wn.synset_from_pos_and_offset(pre[0], int(pre[1:]))
