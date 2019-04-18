import pytest
from hypothesis import strategies as st, given

from finntk.omor.extract import (
    extract_lemmas,
    extract_lemmas_combs,
    extract_lemmas_recurs,
    extract_lemmas_span,
)
from finntk.wordnet.reader import fiwn
from scipy.spatial.distance import cosine
import heapq
import itertools


def intersect(*its):
    for key, values in itertools.groupby(heapq.merge(*its)):
        if len(list(values)) == len(its):
            yield key


@pytest.mark.parametrize(
    "compound, expected_lemmas",
    [
        pytest.param("merenranta", {"merenranta", "meri", "ranta"}, id="merenranta"),
        pytest.param(
            "koneapulainen", {"koneapulainen", "kone", "apulainen"}, id="koneapulainen"
        ),
        pytest.param(
            "voileipäkakku", {"voi", "voileipä", "voileipäkakku"}, id="voileipäkakku"
        ),
        pytest.param(
            "naissukupuoli",
            {"nainen", "sukupuoli", "suku", "puoli", "naissukupuoli"},
            id="naissukupuoli",
        ),
    ],
)
def test_lemmas_combs(compound, expected_lemmas):
    actual_lemmas = extract_lemmas_combs(compound)
    assert actual_lemmas.issuperset(expected_lemmas)


@pytest.mark.parametrize(
    "compound, expected_lemmas",
    [
        pytest.param(
            "synnyinkaupunkini", {"synty", "synnyin", "syntyä"}, id="synnyinkaupunkini"
        )
    ],
)
def test_lemmas_recurs(compound, expected_lemmas):
    actual_lemmas = extract_lemmas_recurs(compound)
    assert actual_lemmas.issuperset(expected_lemmas)


@pytest.mark.parametrize(
    "form, expected",
    [
        pytest.param("en", "ei", id="en"),
        pytest.param("pyykinpesuun", "pyykinpesu", id="pyykinpesuun"),
    ],
)
def test_extract_lemmas_span(form, expected):
    assert extract_lemmas_span(form) == {expected}


@pytest.mark.parametrize("brace", [pytest.param("["), pytest.param("]")])
def test_braces_roundtrip(brace):
    assert extract_lemmas(brace) == {brace}


def fiwn_conceptnet_common_lemmas():
    CONCEPTNET_FI = "/c/fi/"
    from finntk.emb.numberbatch import vecs as numberbatch_vecs

    vecs = numberbatch_vecs.get_vecs()

    def fi_lemmas():
        for entity in vecs.index2entity:
            if entity.startswith(CONCEPTNET_FI):
                yield entity[len(CONCEPTNET_FI):]

    return intersect(fiwn.all_lemma_names(), fi_lemmas())


fiwn_conceptnet_common_lemmas_300 = [
    x for _, x in zip(range(300), fiwn_conceptnet_common_lemmas())
]


@given(st.sampled_from(fiwn_conceptnet_common_lemmas_300))
def test_get_lemma_vec(lemma_name):
    from finntk.emb.autoextend import mk_lemma_vec

    for lemma in fiwn.lemmas(lemma_name):
        assert mk_lemma_vec(lemma) is not None


@given(st.sampled_from(fiwn_conceptnet_common_lemmas_300))
def test_get_synset_vec(lemma_name):
    from finntk.emb.autoextend import mk_synset_vec

    synset = fiwn.lemmas(lemma_name)[0].synset()
    assert mk_synset_vec(synset) is not None


@given(st.one_of(st.just("pitää"), st.just("saada")))
def test_surf_vec_matches(surf):
    from finntk.emb.autoextend import vecs as autoextend_vecs
    from finntk.emb.numberbatch import mk_concept_vec

    assert cosine(mk_concept_vec("fi", surf), autoextend_vecs.get_vecs()[surf]) < 0.01


def test_no_extra_lemmas():
    vararengas_lemmas = extract_lemmas_span("vararengas")
    assert "vara_2rengas" not in vararengas_lemmas
