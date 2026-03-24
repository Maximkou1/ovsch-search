import numpy as np
from rank_bm25 import BM25Okapi
from gensim.models import KeyedVectors
from navec import Navec


_POS_TAGS = ['NOUN', 'VERB', 'ADJ', 'PROPN', 'ADV', 'NUM', 'PRON']


def _lookup_w2v(token: str, model: KeyedVectors) -> np.ndarray or None:
    """Ищет вектор в word2vec: сначала plain, затем token_POS (формат RusVectores)"""
    if token in model:
        return model[token]
    for pos in _POS_TAGS:
        tagged = f'{token}_{pos}'
        if tagged in model:
            return model[tagged]
    return None


def _doc_to_vector_w2v(tokens: list[str], model: KeyedVectors) -> np.ndarray or None:
    """Усредняет word2vec-векторы слов документа"""
    vectors = [v for v in (_lookup_w2v(t, model) for t in tokens) if v is not None]
    return np.mean(vectors, axis=0) if vectors else None


def _doc_to_vector_navec(tokens: list[str], model: Navec) -> np.ndarray or None:
    """Усредняет navec-векторы слов документа"""
    vectors = [model[t] for t in tokens if t in model]
    return np.mean(vectors, axis=0) if vectors else None


def build_bm25_index(docs: list[list[str]]) -> dict:
    """
    Строит BM25-индекс через rank_bm25.
    Возвращает {model: BM25Okapi}
    """
    return {'model': BM25Okapi(docs)}


def build_vector_index(docs: list[list[str]], model, model_type: str) -> dict:
    """
    Строит матрицу усреднённых векторов документов
    """
    vectors = []
    valid_indices = []

    for i, tokens in enumerate(docs):
        vec = _doc_to_vector_w2v(tokens, model) if model_type == 'word2vec' \
            else _doc_to_vector_navec(tokens, model)

        if vec is not None:
            vectors.append(vec)
            valid_indices.append(i)

    return {
        'matrix': np.array(vectors),
        'valid_indices': valid_indices,
    }


def load_word2vec(path: str) -> KeyedVectors:
    """Загружает предобученную word2vec модель (.bin)"""
    print(f"Загрузка word2vec из {path}...")
    return KeyedVectors.load_word2vec_format(path, binary=True)


def load_navec(path: str) -> Navec:
    """Загружает предобученную navec модель (.tar)"""
    print(f"Загрузка navec из {path}...")
    return Navec.load(path)