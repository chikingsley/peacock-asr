"""Per-language gate: does this label actually belong to the target language?

The create path (YouTube) labels whatever is spoken, so a mixed-language source yields clips in
other languages — e.g. a Tajik broadcaster code-switching to Russian mid-bulletin. This registry,
keyed by the curator language code, decides whether a label belongs to its language; the create
pipeline drops the clips that fail. A language with no gate registered keeps every clip (the safe
default). Like the normalizer registry, gates are small, specialized, and live here in the package
so a project stays pure config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

# Tajik (tgk_Cyrl) vs its dominant contaminant, Russian: Tajik Cyrillic has six letters Russian
# never uses (ғ ӣ қ ӯ ҳ ҷ), and Russian has four Tajik never uses (ы ц щ ь). A label is Tajik when
# its Tajik-only letters outnumber its Russian-only ones — which drops Russian segments (zero
# Tajik-only letters), including Russian that also lacks Russian-only letters (0 is not > 0).
_TAJIK_ONLY = frozenset("ғӣқӯҳҷҒӢҚӮҲҶ")
_RUSSIAN_ONLY = frozenset("ыцщьЫЦЩЬ")


def _is_tajik(text: str) -> bool:
    tajik = sum(ch in _TAJIK_ONLY for ch in text)
    russian = sum(ch in _RUSSIAN_ONLY for ch in text)
    return tajik > russian


#: Language gates keyed by curator language code. Opt a language in by adding its predicate.
LANGUAGE_GATES: dict[str, Callable[[str], bool]] = {
    "tgk_Cyrl": _is_tajik,
}


def keep_for_language(text: str, language: str) -> bool:
    """Whether ``text`` belongs to ``language``. True (keep) when no gate is registered for it."""
    gate = LANGUAGE_GATES.get(language)
    return gate is None or gate(text)
