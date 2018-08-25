import numpy as np
from scipy.spatial.distance import cosine
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from finntk.emb.utils import apply_vec
from finntk.emb.fasttext import vecs as fasttext_word_vecs
from finntk.emb.numberbatch import vecs as numberbatch_word_vecs
from finntk.vendor.metanl.nltk_morphy import tag_and_stem


def fasttext_fi_getter(word_form):
    space = fasttext_word_vecs.get_fi()
    return space.get_vector(word_form)


def conceptnet_fi_getter(lemma):
    space = numberbatch_word_vecs.get_vecs()
    return space.get_vector("/c/fi/" + lemma)


def double_fi_getter(pair):
    return np.hstack([fasttext_fi_getter(pair[0]), conceptnet_fi_getter(pair[1])])


def fasttext_en_getter(word_form):
    space = fasttext_word_vecs.get_en()
    return space.get_vector(word_form)


def conceptnet_en_getter(lemma):
    space = numberbatch_word_vecs.get_vecs()
    return space.get_vector("/c/en/" + lemma)


def double_en_getter(pair):
    return np.hstack([fasttext_en_getter(pair[0]), conceptnet_en_getter(pair[1])])


def mk_mk_context_vec(getter):

    def inner(aggf, tokens):
        return apply_vec(aggf, getter, tokens, "fi")

    return inner


mk_context_vec_fasttext_fi = mk_mk_context_vec(fasttext_fi_getter)
mk_context_vec_conceptnet_fi = mk_mk_context_vec(conceptnet_fi_getter)
mk_context_vec_double_fi = mk_mk_context_vec(double_fi_getter)


def trans_toks_passthrough(tokens, lemmatized):
    return tokens


def trans_toks_lemmatise(tokens, lemmatized):
    return lemmatized


def trans_toks_pair(tokens, lemmatized):
    return zip(tokens, lemmatized)


def lemmatize_tokens(tokens):
    # TODO: Consider switching to ConceptNet5 or Stanford taggger for lemmatization
    # TODO: Consider not doing lemmatization at all and just relying on glosswordnet
    return (stem for (stem, _, _) in tag_and_stem(tokens))


def mk_defn_vec(aggf, vec_getter, lemma, transform_tokens):
    return apply_vec(
        aggf,
        vec_getter,
        (tok for tok in transform_tokens(word_tokenize(lemma.synset().definition()))),
        "en",
    )


def mk_mk_defn_vec(getter, token_transformer):

    def inner(aggf, lemma, wn_filter=False):
        if wn_filter:

            def transform_tokens(tokens):
                lemmatized = lemmatize_tokens(tokens)
                transformed = token_transformer(tokens, lemmatized)
                return (
                    tr
                    for tr, le in zip(transformed, lemmatized)
                    if len(wordnet.lemmas(le))
                )

        else:

            def transform_tokens(tokens):
                lemmatized = lemmatize_tokens(tokens)
                return token_transformer(tokens, lemmatized)

        return mk_defn_vec(aggf, getter, lemma, transform_tokens)

    return inner


mk_defn_vec_fasttext_en = mk_mk_defn_vec(fasttext_en_getter, trans_toks_passthrough)
mk_defn_vec_conceptnet_en = mk_mk_defn_vec(conceptnet_en_getter, trans_toks_lemmatise)
mk_defn_vec_double_en = mk_mk_defn_vec(double_en_getter, trans_toks_pair)


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


def mk_disambg(repr_defn, repr_ctx):

    def inner(aggf, stream, context, wn_filter=False):
        return disambg(
            ((item, repr_defn(aggf, item, wn_filter=wn_filter)) for item in stream),
            repr_ctx(aggf, context),
        )

    return inner


disambg_fasttext = mk_disambg(mk_defn_vec_fasttext_en, mk_context_vec_fasttext_fi)
disambg_conceptnet = mk_disambg(mk_defn_vec_conceptnet_en, mk_context_vec_conceptnet_fi)
disambg_double = mk_disambg(mk_defn_vec_double_en, mk_context_vec_double_fi)
