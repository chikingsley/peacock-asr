"""Per-language gate: does this label actually belong to the target language?

The create path (YouTube) labels whatever is spoken, so a mixed-language source yields clips in
other languages — e.g. a Tajik broadcaster code-switching to Russian mid-bulletin. This registry,
keyed by the curator language code, decides whether a label belongs to its language; the create
pipeline drops the clips that fail. A language with no gate registered keeps every clip (the safe
default). Like the normalizer registry, gates are small, specialized, and live here in the package
so a project stays pure config.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

# Tajik (tgk_Cyrl) vs its dominant contaminant, Russian: Tajik Cyrillic has six letters Russian
# never uses (ғ ӣ қ ӯ ҳ ҷ), and Russian has four Tajik never uses (ы ц щ ь). (NOT ъ/э/ё — Tajik
# uses all three: феъл, эзоҳ, ёр.) The letter test decides any clip that has either script's
# exclusive letters; the vocabulary test below resolves the rest.
_TAJIK_ONLY = frozenset("ғӣқӯҳҷҒӢҚӮҲҶ")
_RUSSIAN_ONLY = frozenset("ыцщьЫЦЩЬ")

# Shared-alphabet function words that are distinctly one language — for clips written entirely in
# the common Cyrillic letters (no exclusive letters either way), where letter-counting can't tell
# Tajik from Russian. A pure letter gate wrongly dropped ~1,600 such clips of real Tajik (e.g.
# "Хушбахтона боз", "Зебо калима месозад") because they use none of ғ ӣ қ ӯ ҳ ҷ. Validated on the
# tajik pool: recovers ~1,584 genuine Tajik, zero Russian re-admitted (Russian carries either an
# exclusive letter or these markers). Ambiguous words shared by both (то, он) are in neither set.
_TAJIK_WORDS = frozenset(
    "ва ки ба аз ин бо аст буд барои вале агар ё як ду се ман ту мо шумо нест мешавад мекунад шуд "
    "кард гуфт хеле боз дигар худ кор сол ному чи чаро кадом ҳар вай дорад дорам мехоҳам метавонад "
    "мардум давлат забон меояд меравад омад рафт чанд".split()
)
_RUSSIAN_WORDS = frozenset(
    "и в на что это как не с по мы вы он она они его ее их был была было или для все всё так там "
    "тут вот бы же я ты мне меня тебя нас вас из до от за под над при без про об да нет уже еще "
    "только даже чтобы потому очень когда если который можно нужно человек город год день мой твой "
    "наш ваш этот тот весь всех чем кто где почему".split()
)
_CYRILLIC_WORD = re.compile(r"[а-яёғӣқӯҳҷ]+")


def _is_tajik(text: str) -> bool:
    """Tajik unless it's Russian. Exclusive letters decide first; shared-only text by vocabulary."""
    tajik = sum(ch in _TAJIK_ONLY for ch in text)
    russian = sum(ch in _RUSSIAN_ONLY for ch in text)
    if tajik > russian:
        return True
    if russian > 0:  # Russian-exclusive letters and no Tajik-exclusive ones -> Russian
        return False
    # Written entirely in shared Cyrillic letters: function words decide (drop on no signal).
    words = _CYRILLIC_WORD.findall(text.lower())
    return sum(w in _TAJIK_WORDS for w in words) > sum(w in _RUSSIAN_WORDS for w in words)


#: Language gates keyed by curator language code. Opt a language in by adding its predicate.
LANGUAGE_GATES: dict[str, Callable[[str], bool]] = {
    "tgk_Cyrl": _is_tajik,
}


def keep_for_language(text: str, language: str) -> bool:
    """Whether ``text`` belongs to ``language``. True (keep) when no gate is registered for it."""
    gate = LANGUAGE_GATES.get(language)
    return gate is None or gate(text)
