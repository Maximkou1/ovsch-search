from flask import Flask, render_template, request
from spellcheck import correct_query
from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG


def _parse_top_n(value: str, default: int = 5) -> int:
    """Парсит top_n из строки; возвращает default если введено не число"""
    try:
        return max(1, int(value))
    except (ValueError, TypeError):
        return default


app = Flask(__name__)
_engine = None  # инициализируется из main.py


@app.route('/')
def index():
    return render_template('index.html', doc_count=len(_engine.documents))


@app.route('/search')
def search_page():
    return render_template(
        'search.html',
        query=request.args.get('query', ''),
        method=request.args.get('method', 'bm25'),
        top_n=_parse_top_n(request.args.get('top_n', '5')),
        spellcheck=request.args.get('spellcheck'),
    )


@app.route('/results')
def results_page():
    query = request.args.get('query', '').strip()
    method = request.args.get('method', 'bm25')
    top_n = _parse_top_n(request.args.get('top_n', '5'))
    do_spellcheck = bool(request.args.get('spellcheck'))

    if not query:
        return render_template('search.html', method=method, top_n=top_n)

    original_query = query
    corrected = False
    if do_spellcheck:
        query, corrected = correct_query(query)

    results, elapsed = _engine.search(query, method, top_n)

    return render_template(
        'results.html',
        query=query,
        original_query=original_query,
        corrected=corrected,
        method=method,
        top_n=top_n,
        results=results,
        elapsed=elapsed,
        page_title=f'Результаты: {query}',
    )


@app.route('/recommend')
def recommend_page():
    doc = request.args.get('doc', '').strip()
    method = request.args.get('method', 'bm25')
    top_n = _parse_top_n(request.args.get('top_n', '5'))

    if not doc:
        return render_template('search.html', method=method, top_n=top_n)

    results, elapsed = _engine.recommend(doc, method, top_n)

    return render_template(
        'results.html',
        query=doc,
        original_query=doc,
        corrected=False,
        method=method,
        top_n=top_n,
        results=results,
        elapsed=elapsed,
        page_title='Похожие вопросы',
        is_recommend=True,
    )


def run_app(engine) -> None:
    """Запускает приложение"""
    global _engine
    _engine = engine
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
