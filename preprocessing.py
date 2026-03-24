from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from pymorphy3 import MorphAnalyzer


morph = MorphAnalyzer()
tokenizer = RegexpTokenizer(r'\w\w+')

# стандартные стоп-слова русского языка + специфичные для чгк
stop_words = set(stopwords.words('russian')) | {
    'вопрос', 'ответ', 'комментарий', 'автор', 'редактор',
    'альфа', 'бета', 'гамма', 'икс', 'игрек',
    'назвать', 'ответить', 'слово',
}


def preprocess(text: str) -> list[str]:
    """
    токенизация, лемматизация и фильтрация стоп-слов
    """
    if not isinstance(text, str):
        return []
    tokens = tokenizer.tokenize(text.lower())
    lemmas = [morph.parse(t)[0].normal_form for t in tokens]
    return [lem for lem in lemmas if lem not in stop_words]
