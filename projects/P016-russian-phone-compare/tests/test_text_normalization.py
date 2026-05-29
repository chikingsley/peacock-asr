import p016_compare.g2p as g2p_module
import p016_compare.text_normalization as text_norm_module
from p016_compare.g2p import TargetG2P
from p016_compare.text_normalization import _choose_russian_candidate, normalize_written_text


def test_english_acronyms_expand_to_spoken_letter_names(monkeypatch) -> None:
    monkeypatch.setattr(text_norm_module, "_nemo_normalize", lambda text, language: text)

    result = normalize_written_text("The UN and U.S. discussed AI with the PA.", "en_us")

    assert result.normalized_text == (
        "The you en and you ess discussed ay eye with the pea ay."
    )
    assert result.backend == "nemo-text-processing:en"


def test_russian_acronyms_expand_before_nemo(monkeypatch) -> None:
    monkeypatch.setattr(text_norm_module, "_nemo_normalize", lambda text, language: text)

    result = normalize_written_text("В США ООН и АЭС использовали РЛС.", "ru")

    assert result.normalized_text == (
        "В сэ шэ а о о эн и а э эс использовали эр эл эс."
    )
    assert result.backend == "nemo-text-processing:ru"


def test_russian_candidate_choice_prefers_plain_percent_plural() -> None:
    result = _choose_russian_candidate(
        "Тем не менее, 80% наших товаров.",
        {
            "Тем не менее, восемьдесят процент наших товаров.",
            "Тем не менее, восемьдесят процентами наших товаров.",
            "Тем не менее, восемьдесят процентов наших товаров.",
        },
    )

    assert result == "Тем не менее, восемьдесят процентов наших товаров."


def test_russian_candidate_choice_avoids_bad_thousands_form() -> None:
    result = _choose_russian_candidate(
        "4892 метра",
        {
            "четыре тысяч восемьсот девяносто два метра",
            "четыре тысячи восемьсот девяносто два метра",
        },
    )

    assert result == "четыре тысячи восемьсот девяносто два метра"


def test_russian_candidate_choice_prefers_genitive_after_iz() -> None:
    result = _choose_russian_candidate(
        "Из 1400 человек выросло на 8%.",
        {
            "Из тысяча четыреста человек выросло на восемь процентов.",
            "Из тысячи четыреста человек выросло на восемь процентов.",
            "Из тысяч четыреста человек выросло на восемь процентов.",
        },
    )

    assert result == "Из тысячи четыреста человек выросло на восемь процентов."


def test_russian_candidate_choice_prefers_quantity_before_tysyach() -> None:
    result = _choose_russian_candidate(
        "приблизительно 7 тысяч островов",
        {
            "приблизительно семи тысяч островов",
            "приблизительно семью тысячами островов",
            "приблизительно семь тысяч островов",
        },
    )

    assert result == "приблизительно семь тысяч островов"


def test_russian_candidate_choice_prefers_hundreds_before_tysyach() -> None:
    result = _choose_russian_candidate(
        "приблизительно 400 тысяч случаев",
        {
            "приблизительно четыремстам тысяч случаев",
            "приблизительно четырехсот тысяч случаев",
            "приблизительно четыреста тысяч случаев",
        },
    )

    assert result == "приблизительно четыреста тысяч случаев"


def test_g2p_from_text_scores_normalized_words(monkeypatch) -> None:
    monkeypatch.setattr(
        text_norm_module,
        "_nemo_normalize",
        lambda text, language: "eight percent",
    )
    monkeypatch.setattr(
        g2p_module,
        "_espeak_g2p",
        lambda words, voice: [[word] for word in words],
    )

    result = TargetG2P("espeak").from_text("8%", "en_us")

    assert result.words == ["eight", "percent"]
    assert result.normalized_text == "eight percent"
    assert result.text_normalization_backend == "nemo-text-processing:en"
    assert result.phones_per_word_raw == [["eight"], ["percent"]]
