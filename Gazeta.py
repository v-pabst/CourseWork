import os
import pandas as pd
import numpy as np
import re
import json
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pymorphy3
import warnings
warnings.filterwarnings('ignore')
nltk.download('stopwords')
DATA_PATH = "C:/Users/User/source/repos/Gazeta/Gazeta/gazeta_jsonl_v2" 

def load_jsonl(file_path): #Загрузка JSONL файла в DataFrame
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return pd.DataFrame(data)

# Загрузка обучающей выборки
print("Загрузка локального датасета...")
df_train = load_jsonl(f"{DATA_PATH}/gazeta_train.jsonl")
print(f"Загружено {len(df_train)} новостных статей из train")

# Для анализа возьмём первые 500 примеров 
df = df_train.head(500).copy()
print(f"Для анализа используем {len(df)} статей")

# 1. ОЧИСТКА ТЕКСТА
def clean_text(text):
    text = text.lower()                    # Всё в нижний регистр
    text = re.sub(r'[^а-яё ]', '', text)   # Удаляем всё, кроме русских букв и пробелов
    text = ' '.join(text.split())          # Убираем лишние пробелы
    return text

df['clean_text'] = df['text'].apply(clean_text)

# Вывод примеров до и после очистки
print("\nПРИМЕРЫ ОЧИСТКИ ТЕКСТА")
for i in range(3):
    print(f"\n{i+1}. До очистки:   {df['text'].iloc[i][:200]}...")
    print(f"   После очистки: {df['clean_text'].iloc[i][:200]}...")

# 2. ЛЕММАТИЗАЦИЯ (pymorphy3)
# Инициализация морфологического анализатора
morph = pymorphy3.MorphAnalyzer()

def lemmatize_text(text):
    words = text.split()
    lemmas = []
    for word in words:
        try:
            parsed = morph.parse(word)[0]
            lemmas.append(parsed.normal_form)
        except:
            lemmas.append(word)
    return ' '.join(lemmas)

# Пример работы лемматизации
print("\nПРИМЕР ЛЕММАТИЗАЦИИ")

# Применяем лемматизацию
df['lemmas'] = df['clean_text'].apply(lemmatize_text)

# Вывод примеров
for i in range(3):
    print(f"\n{i+1}. До лемматизации:    {df['clean_text'].iloc[i][:200]}...")
    print(f"   После лемматизации: {df['lemmas'].iloc[i][:200]}...")

# 3. ПОДСЧЁТ ЧАСТОТЫ СЛОВ
# Объединяем все леммы в один список слов
all_words = ' '.join(df['lemmas']).split()
word_counts = Counter(all_words)

print("ТОП-10 САМЫХ ЧАСТЫХ СЛОВ")
for word, count in word_counts.most_common(10):
    print(f"{word}: {count}")

# Столбчатый график топ-10 слов
top_words = word_counts.most_common(10)
words, counts = zip(*top_words)

plt.figure(figsize=(12, 6))
plt.bar(words, counts, color='steelblue')
plt.title('Топ-10 самых частых слов в новостных статьях', fontsize=14)
plt.xlabel('Слово', fontsize=12)
plt.ylabel('Частота', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Облако слов (WordCloud)
print("\nОБЛАКО СЛОВ")
all_text = ' '.join(df['lemmas'])
wc = WordCloud(width=800, height=400, background_color='white', max_words=10, 
               colormap='viridis').generate(all_text)

plt.figure(figsize=(12, 6))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('Облако частых слов в новостях', fontsize=14)
plt.tight_layout()
plt.show()

# 4. УДАЛЕНИЕ СТОП-СЛОВ
# Загружаем русские стоп-слова
stop_words = set(stopwords.words('russian'))

def remove_stopwords(text, stop_words):
    words = text.split()
    return ' '.join([w for w in words if w not in stop_words])

df['no_stopwords'] = df['lemmas'].apply(lambda x: remove_stopwords(x, stop_words))

# Сравнение до и после удаления стоп-слов
print("\nСРАВНЕНИЕ ДО И ПОСЛЕ УДАЛЕНИЯ СТОП-СЛОВ")
for i in range(3):
    print(f"\n{i+1}. До удаления стоп-слов:    {df['lemmas'].iloc[i][:100]}...")
    print(f"   После удаления стоп-слов: {df['no_stopwords'].iloc[i][:100]}...")

# Топ-10 слов после удаления стоп-слов
all_words_filtered = ' '.join(df['no_stopwords']).split()
word_counts_filtered = Counter(all_words_filtered)

print("\nТОП-10 САМЫХ ЧАСТЫХ СЛОВ (ПОСЛЕ УДАЛЕНИЯ СТОП-СЛОВ)")
for word, count in word_counts_filtered.most_common(10):
    print(f"{word}: {count}")

# 5. TF-IDF (превращение текста в числа)
# Создаём и обучаем векторайзер
vectorizer = TfidfVectorizer(stop_words=list(stop_words), max_features=100)
tfidf_matrix = vectorizer.fit_transform(df['no_stopwords'])

# Получаем список всех слов (словарь)
feature_names = vectorizer.get_feature_names_out()

print("\nИНФОРМАЦИЯ О ВЕКТОРИЗАЦИИ")
print(f"Размер словаря (уникальных слов): {len(feature_names)}")
print(f"Размер TF-IDF матрицы: {tfidf_matrix.shape}")
print(f"\nПервые 20 слов из словаря:\n{list(feature_names[:20])}")

# Показываем пример вектора для одного текста
print("\nПРИМЕР ВЕКТОРА ДЛЯ ОДНОГО ТЕКСТА")
text_index = 0
print(f"Текст (после очистки от стоп-слов): {df['no_stopwords'].iloc[text_index][:200]}...")

# Получаем ненулевые значения
vector = tfidf_matrix[text_index].toarray()[0]
non_zero_indices = vector.nonzero()[0]

print(f"\nВектор имеет размерность {len(vector)}")
print(f"Количество ненулевых значений: {len(non_zero_indices)}")
print("\nНенулевые значения TF-IDF:")
for idx in non_zero_indices[:10]:
    print(f"  Слово '{feature_names[idx]}': TF-IDF = {vector[idx]:.4f}")

# 6. ИНФОРМАЦИОННЫЙ ПОИСК
def search_texts(query, vectorizer, tfidf_matrix, texts_df, stop_words_set, top_n=5):

    # очистка и лемматизация запроса
    query_clean = clean_text(query)
    query_lemmas = lemmatize_text(query_clean)
    query_no_stopwords = remove_stopwords(query_lemmas, stop_words_set)
    
    # векторизация запроса
    query_vec = vectorizer.transform([query_no_stopwords])
    
    # вычисление косинусного сходства
    similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
    
    # находим индексы топ-n самых похожих
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    # формируем результаты
    results = []
    for idx in top_indices:
        results.append({
            'индекс': idx,
            'текст (фрагмент)': texts_df['text'].iloc[idx][:200] + "...",
            'похожесть': similarities[idx],
            'суммаризация': texts_df['summary'].iloc[idx][:150] + "..."
        })
    
    return results

# Тестирование поиска
print("\nИнформационный поиск по новостному корпусу")

# Запрос 1
query1 = "россия"
print(f"\nЗапрос 1: '{query1}'")
results1 = search_texts(query1, vectorizer, tfidf_matrix, df, stop_words, top_n=3)
print("результаты поиска:")
for i, res in enumerate(results1, 1):
    print(f"\n{i}. похожесть: {res['похожесть']:.4f}")
    print(f"   суммаризация: {res['суммаризация']}")
    print(f"   текст: {res['текст (фрагмент)']}")

# Запрос 2
query2 = "президент"
print(f"\nЗапрос 2: '{query2}'")
results2 = search_texts(query2, vectorizer, tfidf_matrix, df, stop_words, top_n=3)
print("результаты поиска:")
for i, res in enumerate(results2, 1):
    print(f"\n{i}. похожесть: {res['похожесть']:.4f}")
    print(f"   суммаризация: {res['суммаризация']}")
    print(f"   текст: {res['текст (фрагмент)']}")

# Запрос 3
query3 = "матч"
print(f"\nЗапрос 3: '{query3}'")
results3 = search_texts(query3, vectorizer, tfidf_matrix, df, stop_words, top_n=3)
print("результаты поиска:")
for i, res in enumerate(results3, 1):
    print(f"\n{i}. похожесть: {res['похожесть']:.4f}")
    print(f"   суммаризация: {res['суммаризация']}")
    print(f"   текст: {res['текст (фрагмент)']}")