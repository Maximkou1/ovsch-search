import pandas as pd
from preprocessing import preprocess
from indexing import build_bm25_index, build_vector_index, load_word2vec, load_navec
from search import search as _search
from recommender import recommend as _recommend
from spellcheck import init_vocabulary

from config import DATASET_PATH, DATASET_COLUMN, DATASET_SIZE, DEFAULT_TOP_N


class SearchEngine:
    def __init__(self, w2v_path: str or None = None, navec_path: str or None = None):
        """
        Загружает датасет, предобрабатывает тексты и строит индексы
        """
        print("Загрузка датасета...")
        df = pd.read_csv(DATASET_PATH)
        if DATASET_SIZE:
            df = df.iloc[:DATASET_SIZE]

        self.documents: list[str] = df[DATASET_COLUMN].tolist()
        init_vocabulary(self.documents)

        # метаданные: {текст вопроса: {season, authors}}
        self.metadata: dict[str, dict] = {
            row[DATASET_COLUMN]: {'season': row['season'], 'authors': row['authors']}
            for _, row in df.iterrows()
        }

        print(f"Загружено {len(self.documents)} документов. Предобработка...")
        self.processed: list[list[str]] = [preprocess(q) for q in self.documents]

        self.models: dict = {}
        self.indices: dict = {}

        print("Построение BM25-индекса...")
        self.indices['bm25'] = build_bm25_index(self.processed)

        if w2v_path:
            model = load_word2vec(w2v_path)
            self.models['word2vec'] = model
            print("Построение word2vec-индекса...")
            self.indices['word2vec'] = build_vector_index(self.processed, model, 'word2vec')

        if navec_path:
            model = load_navec(navec_path)
            self.models['navec'] = model
            print("Построение navec-индекса...")
            self.indices['navec'] = build_vector_index(self.processed, model, 'navec')

        print("Готово.\n")

    def _enrich(self, results: list[dict]) -> list[dict]:
        """Добавляет метаданные (season, authors) к каждому результату поиска"""
        for r in results:
            meta = self.metadata.get(r['doc'], {})
            r['season'] = meta.get('season', '')
            r['authors'] = meta.get('authors', '')
        return results

    def search(self, query: str, method: str = 'bm25', top_n: int = DEFAULT_TOP_N) -> tuple[list[dict], float]:
        """Поиск по запросу. Возвращает (результаты с метаданными, время в сек)"""
        results, elapsed = _search(query, method, self.documents, self.indices, self.models, top_n)
        return self._enrich(results), elapsed

    def recommend(self, doc_text: str, method: str = 'bm25', top_n: int = DEFAULT_TOP_N) -> tuple[list[dict], float]:
        """Рекомендации похожих документов. Возвращает (результаты с метаданными, время в сек)"""
        results, elapsed = _recommend(doc_text, method, self.documents, self.indices, self.models, top_n)
        return self._enrich(results), elapsed
