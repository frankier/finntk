import click
from nltk.corpus import wordnet
from finntk.data.wordnet import ALL_ABBRVS


@click.command()
@click.option(
    "--output",
    type=click.Choice(["pron", "abbrv", "group"]),
    default="abbrvs",
    help="What to output.",
)
def main(output):
    pronouns = {}
    for l in wordnet.all_lemma_names(lang="fin"):
        if "_" not in l:
            continue
        for b in l.split("_"):
            if b not in ALL_ABBRVS:
                continue
            pronouns.setdefault(b, []).append(l)
    if output == "group":
        for p, v in sorted(pronouns.items()):
            print(p, v)
    elif output == "abbrv":
        for p, v in sorted(pronouns.items()):
            print(p)
    elif output == "pron":
        for v in sorted((v for l in pronouns.values() for v in l)):
            print(v)


if __name__ == "__main__":
    main()
