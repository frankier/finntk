from scipy.spatial.distance import cosine
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from finntk.emb.utils import avg_vec
from finntk.emb.fasttext import vecs as fasttext_word_vecs
from finntk.emb.numberbatch import vecs as numberbatch_word_vecs


def mk_context_vec_fasttext_fi(word_forms):
    fi = fasttext_word_vecs.get_fi()
    return avg_vec(fi, word_forms)


def mk_context_vec_conceptnet_fi(lemmas):
    vecs = numberbatch_word_vecs.get_vecs()
    return avg_vec(vecs, ("/c/fi/" + lemma for lemma in lemmas))


def mk_defn_vec(vecs, proc_tok, lemma, wn_filter=False):
    return avg_vec(
        vecs,
        (
            proc_tok(tok.lower())
            for tok in word_tokenize(lemma.synset().definition())
            if not wn_filter or len(wordnet.lemmas(tok.lower()))
        ),
    )


def mk_defn_vec_fasttext_en(lemma, wn_filter=False):
    en = fasttext_word_vecs.get_en()
    return mk_defn_vec(en, lambda x: x, lemma, wn_filter=wn_filter)


def mk_defn_vec_conceptnet_en(lemma, wn_filter=False):
    vecs = numberbatch_word_vecs.get_vecs()
    return mk_defn_vec(vecs, lambda x: "/c/en/" + x, lemma, wn_filter=wn_filter)


def get_defn_distance(context_vec, defn_vec):
    if defn_vec is None or context_vec is None:
        return 7
    return cosine(defn_vec, context_vec)


def disambg(lemma_defns, context_vec):
    best_lemma = None
    best_dist = 8
    for lemma, defn_vec in lemma_defns:
        dist = get_defn_distance(context_vec, defn_vec)
        if dist < best_dist:
            best_lemma = lemma
            best_dist = dist
    return best_lemma, best_dist


def disambg_fasttext(lemmas, context, wn_filter=False):
    return disambg(
        (
            (lemma, mk_defn_vec_fasttext_en(lemma, wn_filter=wn_filter))
            for lemma in lemmas
        ),
        mk_context_vec_fasttext_fi(context),
    )


def disambg_conceptnet(lemmas, context, wn_filter=False):
    return disambg(
        (
            (lemma, mk_defn_vec_conceptnet_en(lemma, wn_filter=wn_filter))
            for lemma in lemmas
        ),
        mk_context_vec_conceptnet_fi(context),
    )
