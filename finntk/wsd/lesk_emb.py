from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from finntk.emb.base import BothVectorSpaceAdapter, MonoVectorSpaceAdapter
from finntk.emb.utils import apply_vec
from finntk.vendor.metanl.nltk_morphy import tag_and_stem
from finntk.wordnet.reader import fiwn
from itertools import repeat


def lemmatize_tokens(tokens):
    # TODO: Consider switching to ConceptNet5 or Stanford taggger for lemmatization
    # TODO: Consider not doing lemmatization at all and just relying on glosswordnet
    return (stem for (stem, _, _) in tag_and_stem(tokens))


def unexpanded_defn_getter(lemma):
    return word_tokenize(lemma.synset().definition())


def expanded_defn_getter(lemma):
    lemma_name = lemma.name()
    synset = lemma.synset()
    pos = synset.pos()
    if pos == "n":
        schedule = ["Hypernym", "Hyponym", "Holonym", "Meronym", "Attribute"]
    elif pos == "v":
        schedule = ["Hypernym", "Hyponym", "Also see"]
    elif pos in ("s", "a"):
        schedule = ["Attribute", "Also see", "Similar to", "Pertainym of"]
    else:
        schedule = []
    related_synsets = {synset}
    for relation in schedule:
        if relation == "Hypernym":
            related_synsets.update(synset.hypernyms())
            related_synsets.update(synset.instance_hypernyms())
        elif relation == "Hyponym":
            related_synsets.update(synset.hyponyms())
            related_synsets.update(synset.instance_hyponyms())
        elif relation == "Holonym":
            related_synsets.update(synset.member_holonyms())
            related_synsets.update(synset.substance_holonyms())
            related_synsets.update(synset.part_holonyms())
        elif relation == "Meronym":
            related_synsets.update(synset.member_meronyms())
            related_synsets.update(synset.substance_meronyms())
            related_synsets.update(synset.part_meronyms())
        elif relation == "Attribute":
            related_synsets.update(synset.attributes())
        elif relation == "Also see":
            related_synsets.update(synset.also_sees())
        elif relation == "Similar to":
            related_synsets.update(synset.similar_tos())
        elif relation == "Pertainym of":
            related_synsets.update((l.synset() for l in lemma.pertainyms()))
        else:
            assert False, "Unknown relation"
    tokens = []
    for synset in related_synsets:
        tokens.extend(word_tokenize(synset.definition()))
        tokens.extend(
            (lemma.name() for lemma in synset.lemmas() if lemma.name() != lemma_name)
        )
    return tokens


def safe_cosine_sim(u, v):
    from finntk.emb.utils import cosine_sim

    if u is None or v is None:
        return -2
    return cosine_sim(u, v)


def disambg_one(lemma_defns, context_vec, freqs=None):
    if freqs is None:
        freqs = repeat(None)
    best_lemma = None
    best_sim = float("-inf")
    if context_vec is None:
        return best_lemma, best_sim
    for (lemma, defn_vec), freq in zip(lemma_defns, freqs):
        if defn_vec is None:
            continue
        sim = safe_cosine_sim(context_vec, defn_vec)
        if freq is not None:
            if sim >= 0:
                sim *= freq
            else:
                sim /= freq
        if sim > best_sim:
            best_lemma = lemma
            best_sim = sim
    return best_lemma, best_sim


def wn_filter_stream(wn, stream):
    return ((to, le) for to, le in stream if len(wn.lemmas(le)))


class MultilingualLesk:

    def __init__(self, multispace, aggf, wn_filter=False, expand=False, use_freq=False):
        self.multispace = multispace
        self.defn_space = BothVectorSpaceAdapter(
            MonoVectorSpaceAdapter(multispace, "en")
        )
        self.ctx_space = BothVectorSpaceAdapter(
            MonoVectorSpaceAdapter(multispace, "fi")
        )
        self.aggf = aggf
        self.wn_filter = wn_filter
        self.expand = expand
        self.use_freq = use_freq

    def mk_defn_vec(self, item):
        if self.expand:
            defn_tokens = expanded_defn_getter(item)
        else:
            defn_tokens = unexpanded_defn_getter(item)

        lemmatized = lemmatize_tokens(defn_tokens)
        stream = zip(defn_tokens, lemmatized)
        if self.wn_filter:
            stream = wn_filter_stream(wordnet, stream)

        vec = apply_vec(self.aggf, self.defn_space, stream, "en")
        return vec

    def mk_ctx_vec(self, context):
        if self.wn_filter:
            context = wn_filter_stream(fiwn, context)
        vec = apply_vec(self.aggf, self.ctx_space, context, "fi")
        return vec

    def disambg_one(self, choice_freqs, context):
        if self.use_freq:
            choices = []
            freqs = []
            for choice, freq in choice_freqs:
                choices.append(choice)
                freqs.append(freq)
        else:
            choices = choice_freqs
            freqs = None
        return disambg_one(
            ((item, self.mk_defn_vec(item)) for item in choices),
            self.mk_ctx_vec(context),
            freqs,
        )
