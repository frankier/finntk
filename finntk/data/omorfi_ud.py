# OMorFi's Python library can output UD but there's no way to do the reverse
# map: UD to Omor


PASSTHROUGHS = {"CASE", "NUMTYPE", "PRONTYPE", "ADPTYPE", "FOREIGN"}
PASSTHROUGHS_KEY_MAP = {
    "Clitic": "CLIT",
    "Degree": "CMP",
    "Person": "PERS",
    "Person[psor]": "POSS",
    # XXX: Except perhaps this should sometimes be LEX
    "Derivation": "DRV",
}
NUM_KEY_MAP = {"Number": "PERS", "Number[psor]": "POSS"}

NUM_VAL_MAP = {"Sing": "SG", "Plur": "PL"}

TENSE_MAP = {"Pres": "PRESENT", "Past": "PAST"}

MOOD_MAP = {
    "Ind": "INDV",
    "Cnd": "COND",
    "Imp": "IMPV",
    # Passthrough
}

VOICE_MAP = {"Pass": "PSS", "Act": "ACT"}

PART_FORM_MAP = {
    "Pres": "VA", "Past": "NUT", "Agent": "MA", "Agt": "MA", "Neg": "MATON"
}

INF_FORM_MAP = {"1": "A", "2": "E", "3": "MA", "4": "MINEN", "5": "MAISILLA"}
