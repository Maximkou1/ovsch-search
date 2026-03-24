import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='search',
        description='Поиск по вопросам ОВСЧ',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # команда search
    search_parser = subparsers.add_parser('search', help='Поиск по запросу')
    search_parser.add_argument('--query', '-q', required=True, help='Текст запроса')
    search_parser.add_argument(
        '--method', '-m',
        choices=['bm25', 'word2vec', 'navec'],
        default='bm25',
        help='Метод поиска (по умолчанию: bm25)',
    )
    search_parser.add_argument(
        '--top-n', '-n',
        type=int,
        default=5,
        help='Количество результатов (по умолчанию: 5)',
    )
    search_parser.add_argument(
        '--spellcheck', '-s',
        action='store_true',
        help='Включить исправление опечаток',
    )

    # команда recommend
    rec_parser = subparsers.add_parser('recommend', help='Рекомендации похожих документов')
    rec_parser.add_argument('--query', '-q', required=True, help='Текст для поиска похожих')
    rec_parser.add_argument(
        '--method', '-m',
        choices=['bm25', 'word2vec', 'navec'],
        default='bm25',
    )
    rec_parser.add_argument('--top-n', '-n', type=int, default=5)

    # команда web
    subparsers.add_parser('web', help='Запустить веб-интерфейс')

    return parser


def run_cli(engine) -> None:
    """
    Запускает CLI. engine — объект SearchEngine с уже загруженными индексами
    """
    parser = build_parser()
    args = parser.parse_args()

    if args.command == 'web':
        from app import run_app
        run_app(engine)
        return

    query = args.query

    # исправление опечаток (только для команды search)
    if args.command == 'search' and args.spellcheck:
        from spellcheck import correct_query
        corrected, changed = correct_query(query)
        if changed:
            print(f"Исправлено: «{query}» → «{corrected}»")
            query = corrected

    if args.command == 'search':
        results, elapsed = engine.search(query, args.method, args.top_n)
        print(f"\nРезультаты ({args.method}, {elapsed * 1000:.1f} мс):")
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] {r['doc']}")

    elif args.command == 'recommend':
        results, elapsed = engine.recommend(query, args.method, args.top_n)
        print(f"\nПохожие документы ({args.method}, {elapsed * 1000:.1f} мс):")
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] {r['doc']}")
