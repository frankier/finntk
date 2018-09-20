def main():
    from finntk.emb.fasttext import vecs as fasttext_vecs
    from finntk.emb.autoextend import vecs as autoextend_vecs
    from finntk.emb.numberbatch import vecs as numberbatch_vecs
    from finntk.wordnet.reader import fiwn_resman

    for res_man in [fasttext_vecs, autoextend_vecs, numberbatch_vecs, fiwn_resman]:
        for resource in res_man.resource_names():
            res_man.bootstrap(resource)


if __name__ == "__main__":
    main()
