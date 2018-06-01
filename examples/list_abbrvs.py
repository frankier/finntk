from nltk.corpus import wordnet
from finntk.wordnet import has_abbrv


def main():
    for l in wordnet.all_lemma_names(lang="fin"):
        if has_abbrv(l):
            print(l)


if __name__ == "__main__":
    main()
