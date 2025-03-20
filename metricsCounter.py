import nltk
import math
import spacy
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
import os

# Загрузка русский NLP-парсер
nltk.download('punkt_tab')
nlp = spacy.load("ru_core_news_sm")

def flesch_reading_ease(text):
    """ Вычисление индекса удобочитаемости Flesch для русского текста """
    sentences = sent_tokenize(text, language="russian")
    words = word_tokenize(text, language="russian")
    syllables = sum(count_syllables(word) for word in words)
    
    num_sentences = len(sentences)
    num_words = len(words)

    if num_words == 0 or num_sentences == 0:
        return 0  # Избегаем деления на ноль

    fre = 206.835 - (1.52 * (num_words / num_sentences)) - (65.14 * (syllables / num_words))
    return round(fre, 2)

def count_syllables(word):
    """ Оценка количества слогов в слове """
    vowels = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
    return sum(1 for char in word if char in vowels)

def lemmatize_words(text):
    """ Приведение слов к начальной форме (лемматизация) """
    doc = nlp(text)
    return [token.lemma_ for token in doc if token.is_alpha]

def mtld(text, threshold=0.72):
    """ Вычисление Type-Token Ratio (TTR) с помощью метода MTLD """
    def ttr(segment):
        """ Вычисление Type-Token Ratio для сегмента текста """
        types = set(segment)
        return len(types) / len(segment) if len(segment) > 0 else 0

    def factors(text):
        """ Вычисление факторов MTLD в одном направлении """
        words = lemmatize_words(text)
        segment = []
        factors = []
        for word in words:
            segment.append(word)
            if ttr(segment) < threshold:
                factors.append(len(segment) - 1)
                segment = [word]
        if segment:
            factors.append(len(segment))
        return factors

    forward_factors = factors(text)
    backward_factors = factors(text[::-1])

    mtld_value = (sum(forward_factors) + sum(backward_factors)) / (len(forward_factors) + len(backward_factors))
    return round(mtld_value, 2)

def text_entropy(text):
    """ Вычисление энтропии текста """
    words = lemmatize_words(text)
    total_words = len(words)
    
    if total_words == 0:
        return 0
    
    word_counts = Counter(words)
    probs = [freq / total_words for freq in word_counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs)
    
    return round(entropy, 4)

def avg_sentence_length(text):
    """ Вычисление средней длины предложения в словах """
    sentences = sent_tokenize(text, language="russian")
    words = word_tokenize(text, language="russian")
    
    num_sentences = len(sentences)
    num_words = len(words)

    if num_sentences == 0:
        return 0
    
    return round(num_words / num_sentences, 2)

def syntax_tree_depth(text):
    """ Определение средней глубины синтаксического дерева предложений """
    sentences = sent_tokenize(text, language="russian")
    depths = []

    for sentence in sentences:
        doc = nlp(sentence)
        max_depth = max([find_depth(token) for token in doc] or [0])
        depths.append(max_depth)

    if len(depths) == 0:
        return 0

    return round(sum(depths) / len(depths), 2)

def find_depth(token):
    """ Определение глубины дерева для токена """
    depth = 0
    while token.head != token:
        depth += 1
        token = token.head
    return depth

def count_metrics(text):
    """ Вычисление метрик для текста """
    print("FRE:", flesch_reading_ease(text))
    print("MTLD:", mtld(text))
    print("Энтропия:", text_entropy(text))
    print("Средняя длина предложения:", avg_sentence_length(text))
    print("Глубина синтаксического дерева:", syntax_tree_depth(text))

def read_texts_from_directory(directory):
    """ Чтение текстов из директории """
    texts = {}
    for theme in os.listdir(directory):
        theme_path = os.path.join(directory, theme)
        if os.path.isdir(theme_path):
            texts[theme] = {}
            for file_name in os.listdir(theme_path):
                file_path = os.path.join(theme_path, file_name)
                if os.path.isfile(file_path):
                    with open(file_path, 'r', encoding='utf-8') as file:
                        texts[theme][file_name] = file.read()
    return texts

# Пример использования
texts = read_texts_from_directory('texts')

for type in texts:
    print(type)
    for key in texts[type]:
        print(key)
        count_metrics(texts[type][key])
        print()
