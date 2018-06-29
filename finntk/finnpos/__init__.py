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

    def source_func(finnpos, queue, *args, **kwargs):
        for sent, extra in source_iter(*args, **kwargs):
            finnpos.feed_sent(sent)
            queue.put(extra)
        queue.put(done_sentinel)

    with FinnPOSCtx() as finnpos:
        ctx = mp.get_context("fork")
        id_queue = ctx.Queue(maxsize=0)
        source_proc = ctx.Process(
            target=source_func, args=(finnpos, id_queue) + args, kwargs=kwargs
        )
        source_proc.start()
        while 1:
            extra = id_queue.get()
            if extra is done_sentinel:
                break
            tagged_sent = finnpos.get_analys()
            yield tagged_sent, extra
        source_proc.join()


class FinnPOS():

    def __init__(self):
        self.proc = sp.Popen(
            ["ftb-label"], stdin=sp.PIPE, stdout=sp.PIPE, universal_newlines=True
        )

    def cleanup(self):
        if not self.proc.stdin.closed:
            self.proc.stdin.close()

    def feed_sent(self, sent):
        for token in sent:
            self.proc.stdin.write(token)
            self.proc.stdin.write("\n")
        self.proc.stdin.write("\n")
        self.proc.stdin.flush()

    def get_analys(self):
        tagged_sent = []
        for line in self.proc.stdout:
            if line == "\n":
                break
            surf, lemma, feats_str = parse_finnpos_line(line[:-1])
            feats = analysis_to_dict(feats_str)
            tagged_sent.append((surf, lemma, feats))
        return tagged_sent

    def __del__(self):
        self.cleanup()

    def __call__(self, sent):
        """
        Transform a single sentence with FinnPOS.

        Note that using this repeatedly serialises your processing pipeline
        sentence-by-sentence. If performance is a concern, consider using
        `batch_finnpos` if possible in this situation.
        """
        self.feed_sent(sent)
        return self.get_analys()


class FinnPOSCtx():
    """
    This helper lets you get an instance of `FinnPOS` and ensures it is
    correctly cleaned up. Usually you should use this instead of instantiating
    FinnPOS directly.
    """

    def __enter__(self):
        self.finnpos = FinnPOS()

    def __exit__(self):
        self.finnpos.cleanup()
        del self.finnpos


_global_finnpos = None


def sent_finnpos(sent):
    FinnPOS.__call__.__doc__ + """

    This function will keep a single global copy of FinnPOS running.

    Note that this function is not thread safe and is a convenience for
    exploratory programming only. The recommended method is to use FinnPOSCtx.
    """
    global _global_finnpos
    if _global_finnpos is None:
        _global_finnpos = FinnPOS()
    return _global_finnpos(sent)


def cleanup():
    """
    Cleanup the global FinnPOS instance kept by `sent_finnpos`. If you want to
    use this, you should consider using `FinnPOSCtx` instead.
    """
    global _global_finnpos
    _global_finnpos = None
