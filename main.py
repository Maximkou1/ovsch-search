from engine import SearchEngine
from cli import run_cli, build_parser
from config import WORD2VEC_PATH, NAVEC_PATH

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    method = getattr(args, 'method', 'bm25')
    w2v_path = WORD2VEC_PATH if method == 'word2vec' else None
    navec_path = NAVEC_PATH if method == 'navec' else None

    # для веб-интерфейса загружаем все модели сразу
    if args.command == 'web':
        w2v_path = WORD2VEC_PATH
        navec_path = NAVEC_PATH

    engine = SearchEngine(w2v_path=w2v_path, navec_path=navec_path)
    run_cli(engine)