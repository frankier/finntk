import subprocess as sp
import multiprocessing as mp
from finntk.omor.anlys import pairs_to_dict


def analysis_to_pairs(ana):
    for bit in ana.split("]|["):
        k, v = bit.strip("[]").split("=", 1)
        yield k, v


def parse_finnpos_line(line):
    surf, _, lemma, feats, _ = line.split("\t")
    return surf, lemma, feats


def analysis_to_dict(ana):
    return pairs_to_dict(analysis_to_pairs(ana))


def batch_finnpos(source_iter, *args, maxsize=0, **kwargs):
    """
    Shovel sentences through FinnPOS. A typical use case for this would be when
    you want to tag a bunch of sentences with FinnPOS at the same time as doing
    some other kind of transformation.

    `source_iter` should be an iterator returning pairs `(sent, extra)` where
    sent is a list of tokens and `extra` is any sentence identifier you would
    like to pass through. It will be run in a new process. If you would like to
    pass arguments to it, pass them in as extra arguments to `batch_finnpos`.
    """
    done_sentinel = object()

    def source_func(pipe, queue, *args, **kwargs):
        for sent, extra in source_iter(*args, **kwargs):
            for token in sent:
                pipe.write(token)
                pipe.write("\n")
            pipe.write("\n")
            queue.put(extra)
        queue.put(done_sentinel)

    finnpos_proc = sp.Popen(
        ["ftb-label"], stdin=sp.PIPE, stdout=sp.PIPE, universal_newlines=True
    )
    ctx = mp.get_context("fork")
    id_queue = ctx.Queue(maxsize=0)
    source_proc = ctx.Process(
        target=source_func, args=(finnpos_proc.stdin, id_queue) + args, kwargs=kwargs
    )
    source_proc.start()
    while 1:
        extra = id_queue.get()
        if extra is done_sentinel:
            break
        tagged_sent = []
        for line in finnpos_proc.stdout:
            surf, lemma, feats_str = line[:-1]
            feats = analysis_to_dict(feats_str)
            tagged_sent.append((surf, lemma, feats))
        yield tagged_sent, extra
    source_proc.join()
