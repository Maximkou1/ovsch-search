from search import search


def recommend(doc_text: str, method: str, documents: list[str], indices: dict,
              models: dict, top_n: int = 5) -> tuple[list[dict], float]:
    """
    Рекомендует документы, похожие на переданный текст.
    Исключает сам документ из результатов, если он есть в корпусе
    """
    # ищем на 1 больше, чтобы исключить сам документ если он есть в корпусе
    results, elapsed = search(doc_text, method, documents, indices, models, top_n=top_n + 1)
    filtered = [r for r in results if r['doc'] != doc_text][:top_n]
    return filtered, elapsed
