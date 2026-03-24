import re
from collections import Counter


_ALPHABET = 'абвгдеёжзийклмнопрстуфхцчшщьыъэюяabcdefghijklmnopqrstuvwxyz'

# словарь частот слов корпуса: {слово: частота}
_WORDS: dict[str, int] = {}
_N: int = 1  # сумма всех частот


def init_vocabulary(documents: list[str]) -> None:
    """
    Строит словарь частот из корпуса документов.
    Вызывается один раз при старте из engine.py
    """
    global _WORDS, _N
    all_words = re.findall(r'[а-яёa-z]+', ' '.join(documents).lower())
    _WORDS = dict(Counter(all_words))
    _N = sum(_WORDS.values()) or 1


def _prob(word: str) -> float:
    """Вероятность слова по корпусу"""
    return _WORDS.get(word, 0) / _N


def _splits(word: str) -> list[tuple[str, str]]:
    """Все способы разбить слово на (левая часть, правая часть)"""
    return [(word[:i], word[i:]) for i in range(len(word) + 1)]


def _edits1(word: str) -> set[str]:
    """Все слова на расстоянии 1 от исходного"""
    sp = _splits(word)
    deletes = [l + r[1:] for l, r in sp if r]
    transposes = [l + r[1] + r[0] + r[2:] for l, r in sp if len(r) > 1]
    replaces = [l + c + r[1:] for l, r in sp if r for c in _ALPHABET]
    inserts = [l + c + r for l, r in sp for c in _ALPHABET]
    return set(deletes + transposes + replaces + inserts)


def _edits2(word: str) -> set[str]:
    """Все слова на расстоянии 2 от исходного"""
    return {e2 for e1 in _edits1(word) for e2 in _edits1(e1)}


def _known(words: set[str]) -> set[str]:
    """Оставить только слова, которые есть в словаре корпуса"""
    return {w for w in words if w in _WORDS}


def _correct_word(word: str) -> str:
    """Возвращает наиболее вероятное исправление для одного слова"""
    candidates = (
        _known({word})        # слово уже правильное
        or _known(_edits1(word))  # расстояние 1
        or _known(_edits2(word))  # расстояние 2
        or {word}              # оставить как есть
    )
    return max(candidates, key=_prob)


def correct_query(query: str) -> tuple[str, bool]:
    """
    Исправляет опечатки в запросе
    """
    words = query.split()
    corrected_words = [_correct_word(w.lower()) for w in words]
    corrected = ' '.join(corrected_words)
    changed = corrected != query.lower()
    return corrected, changed
