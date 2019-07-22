def conceptnet5_lemmatize_en(tokens):
    from finntk.conceptnet5 import conceptnet_wiktionary
    from conceptnet5.language.lemmatize import lemmatize

    conceptnet_wiktionary.bootstrap()
    return (lemmatize("en", token)[0] for token in tokens)


def metanl_lemmatize_en(tokens):
    from finntk.vendor.metanl.nltk_morphy import tag_and_stem

    return (stem for (stem, _, _) in tag_and_stem(tokens))


default_lemmatize_en = metanl_lemmatize_en
