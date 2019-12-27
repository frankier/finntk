"""
Mappings from terms used on English Wiktionary to normalised morphemes
"""

TEMPLATE_NORMSEG_MAP = {
    "comparative of": "-mpi",
    "superlative of": "-in",
    "agent noun of": "-ja",
    "plural of": "-t",
}

CASE_NORMSEG_MAP = {
    "nominative": None,
    "genitive": "-n",
    "partitive": "-ta",
    "inessive": "-ssa",
    "elative": "-sta",
    "illative": "-seen",  # XXX: -an? -Vn?
    "adessive": "-lla",
    "ablative": "-lta",
    "allative": "-lle",
    "essive": "-na",
    "translative": "-ksi",
    "instructive": "-in",
    "abessive": "-tta",
    "comitative": "-ine",
}

PL_CASES = {"instructive", "comitative"}

PL_NORMSEG_MAP = {"singular": None, "plural": "-t"}

NOUN_FORM_OF_FIELDS_MAP = {"case": CASE_NORMSEG_MAP, "pl": PL_NORMSEG_MAP}

VERB_FORM_OF_FIELDS_MAP = {
    "mood": {
        "indicative": None,
        "conditional": "-isi",
        "imperative": "(!)",
        "potential": "-ne",
        "optative": "-os",
    },
    "tense": {"past": "-i", "present": None, "present connegative": "(ei)"},
    ("pr", "pl"): {  # pr and pl need to be considered together
        ("first-person", "plural"): "-mme",
        ("second-person", "plural"): "-tte",
        ("third-person", "plural"): "-vat",
        ("impersonal", "plural"): "-taan",
        ("first-person", "singular"): "-n",
        ("second-person", "singular"): "-t",
        ("third-person", "singular"): None,
        ("impersonal", "singular"): "-taan",
    },
}

PARTICIPLES_MAP = {
    "pres": "-va",
    "pres_pass": "-ttava",
    "past": "-nut",
    "past_pass": "-ttu",
    "agnt": "-ma",
    "nega": ("-ma", "-ton"),
}

PARTICIPLES_NORM = {"pres_pasv": "pres_pass", "past_pasv": "past_pass"}

# XXX: Is this identical to CASE_NAME_MAP in omorfi_normseg?
FI_INFINITIVE_OF_ABBRVS = {
    "nom": "nominative",
    "gen": "genitive",
    "par": "partitive",
    "acc": "accusative",
    "ine": "inessive",
    "ela": "elative",
    "ill": "illative",
    "ade": "adessive",
    "abl": "ablative",
    "all": "allative",
    "ess": "essive",
    "tra": "translative",
    "ins": "instructive",
    "abe": "abessive",
    "com": "comitative",
}

FI_INFINITIVES = {
    "1l": "-ksi",
    "2a": "-e",
    "2p": "-tae",
    "3a": "-ma",
    "3p": "-tama",
    "4": "-minen",
    "5": ("-ma", "-isi"),
}

FI_INFINITIVE_DEFAULT_CASES = {
    "1l": "illative",
    "2a": "inessive",
    "2p": "inessive",
    "3a": None,
    "3p": "instructive",
    "4": "nominative",
    "5": "adessive",
}
