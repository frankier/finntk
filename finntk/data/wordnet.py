STD_ABBRVS = {
    "jstak": "jostakin",
    "jssak": "jossakin",
    "jnak": "jonakin",
    "jtak": "jotakin",
    "jllak": "jollakin",
    "jltak": "joltakin",
    "jksta": "jostakusta",
    "jllek": "jollekin",
    "jkssa": "jossakussa",
    "jkta": "jotakuta",
    "jklla": "jollakulla",
    "jklta": "joltakulta",
    "jhk": "johonkin",
    "jklle": "jollekulle",
    "jksik": "joksikin",
    "jkksi": "joksikuksi",
    "jkna": "jonakuna",
    "jkhun": "johonkuhun",
    "jku": "joku",  # (“somebody”).
    "jk": "jokin",  # (“something”).
    "jkn": "jonkun",  # (“of somebody, somebody's”).
    "jnk": "jonkin",
}

OTHER_ABBRVS = {
    "jhkin": "johonkin",
    "jhkun": "johonkuhun",
    "jksk": "joksikin",
    "jkskin": "joksikin",
    "jlkn": "jollakin",
    "jllk": "jollekin",
    "jltk": "joltakin",
    "jnnk": "jonnekin",  # new
    "jsatk": "jostakin",
    "jssk": "jossakin",
    "jstk": "jostakin",
    "jtk": "jotakin",
}

ALL_ABBRVS = {**STD_ABBRVS, **OTHER_ABBRVS}

PRON_CASE = {
    "jostakin": "ela",
    "jossakin": "ine",
    "jonakin": "ess",
    "jotakin": "par",
    "jollakin": "ade",
    "joltakin": "abl",
    "jostakusta": "ela",
    "jollekin": "all",
    "jossakussa": "ine",
    "jotakuta": "par",
    "jollakulla": "ade",
    "joltakulta": "abl",
    "johonkin": "ill",
    "jollekulle": "all",
    "joksikin": "tra",
    "joksikuksi": "tra",
    "jonakuna": "ess",
    "johonkuhun": "ill",
    "joku": "nom",
    "jokin": "nom",
    "jonkun": "gen",
    "jonkin": "gen",
    "jonnekin": "ill",
}

PRON_LEMMAS = {
    "jostakin": "jokin",
    "jossakin": "jokin",
    "jonakin": "jokin",
    "jotakin": "jokin",
    "jollakin": "jokin",
    "joltakin": "jokin",
    "jostakusta": "joku",
    "jollekin": "jokin",
    "jossakussa": "joku",
    "jotakuta": "joku",
    "jollakulla": "joku",
    "joltakulta": "joku",
    "johonkin": "jokin",
    "jollekulle": "joku",
    "joksikin": "joku",
    "joksikuksi": "joku",
    "jonakuna": "joku",
    "johonkuhun": "joku",
    "joku": "joku",
    "jokin": "jokin",
    "jonkun": "joku",
    "jonkin": "jokin",
    "jonnekin": "jokin",
}
