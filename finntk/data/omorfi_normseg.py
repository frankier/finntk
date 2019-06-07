from .wiktionary_normseg import CASE_NORMSEG_MAP

MOOD_MAP = {"COND": "-isi", "IMPV": "!", "POTN": "-ne", "OPT": "-os"}

TENSE_MAP = {"PAST": "-i", "PRESENT": None}

PERS_MAP = {
    "PL1": "-mme", "PL2": "-tte", "PL3": "-vat", "SG1": "-n", "SG2": "-t", "SG3": None
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

CASE_MAP = {k: CASE_NORMSEG_MAP[p] for k, p in CASE_NAME_MAP if p in CASE_NORMSEG_MAP}

NUM_MAP = {"PL": "-t", "SG": None}
