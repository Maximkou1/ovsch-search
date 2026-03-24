import time
import numpy as np
from preprocessing import preprocess

# для поиска в word2vec моделях
_POS_TAGS = ['NOUN', 'VERB', 'ADJ', 'PROPN', 'ADV', 'NUM', 'PRON']


def _lookup(token: str, model) -> np.ndarray or None:
    """
    Ищет вектор токена в модели.
    Пробует сначала plain-форму (navec), затем с POS-тегами word_POS (word2vec)
    """
    if token in model:
        return model[token]
    for pos in _POS_TAGS:
        tagged = f'{token}_{pos}'
        if tagged in model:
            return model[tagged]
    return None


def _cosine_similarities(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Косинусное сходство между вектором запроса и всеми строками матрицы"""
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1e-9
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return np.zeros(len(matrix))
    return (matrix @ query_vec) / (norms * query_norm)


def _rank(scores: np.ndarray, documents: list[str], top_n: int) -> list[dict]:
    """Возвращает топ-N документов по убыванию score"""
    top_indices = np.argsort(scores)[::-1][:top_n]
    return [
        {'doc': documents[i], 'score': float(scores[i])}
        for i in top_indices if scores[i] > 0
    ]


def search(query: str, method: str, documents: list[str], indices: dict,
           models: dict, top_n: int = 5) -> tuple[list[dict], float]:
    """
    Выполняет поиск и возвращает результаты вместе со временем поиска
    """
    query_tokens = preprocess(query)
    start = time.perf_counter()

    if method == 'bm25':
        results = _search_bm25(query_tokens, indices['bm25'], documents, top_n)

    elif method in ('word2vec', 'navec'):
        model = models[method]
        results = _search_vector(query_tokens, indices[method], model, documents, top_n)

    else:
        raise ValueError(f"Неизвестный метод: {method!r}. Ожидается bm25, word2vec или navec.")

    elapsed = time.perf_counter() - start
    return results, elapsed


def _search_bm25(query_tokens: list[str], index_data: dict, documents: list[str], top_n: int) -> list[dict]:
    """Поиск через BM25"""
    scores = np.array(index_data['model'].get_scores(query_tokens))
    return _rank(scores, documents, top_n)


def _search_vector(query_tokens: list[str], index_data: dict, model,
                   documents: list[str], top_n: int) -> list[dict]:
    """
    Поиск через косинусное сходство усреднённых векторов (word2vec / navec).
    Для word2vec пробует plain-форму и формат token_POS
    """
    vectors = [v for v in (_lookup(t, model) for t in query_tokens) if v is not None]
    if not vectors:
        return []
    query_vec = np.mean(vectors, axis=0)

    matrix = index_data['matrix']
    valid_indices = index_data['valid_indices']

    similarities = _cosine_similarities(query_vec, matrix)

    top_local = np.argsort(similarities)[::-1][:top_n]
    return [
        {'doc': documents[valid_indices[i]], 'score': float(similarities[i])}
        for i in top_local if similarities[i] > 0
    ]
