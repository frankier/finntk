import pytest

from finntk.omor.extract import extract_lemmas_combs, extract_lemmas_recurs


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
