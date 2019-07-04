from .wiktionary_normseg import CASE_NORMSEG_MAP

MOOD_MAP = {"cond": "-isi", "impv": "(!)", "potn": "-ne", "opt": "-os"}

TENSE_MAP = {"past": "-i", "present": None}

PERS_MAP = {
    "pl1": "-mme", "pl2": "-tte", "pl3": "-vat", "sg1": "-n", "sg2": "-t", "sg3": None
}

CASE_NAME_MAP = {
    "abe": "abessive",
    "abl": "ablative",
    "ade": "adessive",
    "all": "allative",
    "com": "comitative",
    "ela": "elative",
    "ess": "essive",
    "gen": "genitive",
    "ill": "illative",
    "ine": "inessive",
    "ins": "instructive",
    "nom": "nominative",
    "par": "partitive",
    "tra": "translative",
    "lat": None,  # "-s"?
    "acc": None,  # -n / -ut ?
}

INF_MAP = {"e": "-e", "ma": "-ma", "minen": "-minen"}

CASE_MAP = {
    k: CASE_NORMSEG_MAP[p] for k, p in CASE_NAME_MAP.items() if p in CASE_NORMSEG_MAP
}

NUM_MAP = {"pl": "-t", "sg": None}

POSS_MAP = {"sg1": "-ni", "sg2": "-si", "pl1": "-mme", "pl2": "-nne", "3": "-nsa"}
