# датасет
DATASET_PATH = 'dataset_ovsch.csv'
DATASET_COLUMN = 'question'
DATASET_SIZE = None  # None — весь датасет

# пути к моделям
WORD2VEC_PATH = 'models/tayga_upos_skipgram_300_2_2019.bin'
NAVEC_PATH = 'models/navec_hudlit_v1_12B_500K_300d_100q.tar'

# параметры поиска
DEFAULT_TOP_N = 5
DEFAULT_METHOD = 'bm25'

# Flask
FLASK_HOST = '127.0.0.1'
FLASK_PORT = 5000
FLASK_DEBUG = False