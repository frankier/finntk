from finntk.utils import urlretrieve
from conceptnet5.readers.wiktionary import prepare_db

URL = "https://conceptnet.s3.amazonaws.com/precomputed-data/2016/wiktionary/parsed-2/en.jsons.gz"
en_wiktionary_gz = urlretrieve(URL, filename="en.jsons.gz")
prepare_db([en_wiktionary_gz], "wiktionary.db")
