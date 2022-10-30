import pandas as pd
import string
import numpy as np
from matplotlib import pyplot as plt
import gensim.corpora

from gensim.models import Word2Vec

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize


def create_dataset(samples, mfields, sfields, info_about):
    stop_words = stopwords.words('russian') #cкачиваем массив русских стопслов
    lemmatizer = WordNetLemmatizer()
    trainX, trainY = list(), list() #входные и выходные данные нейронки

    for i in samples.index:
        company = samples.loc[i]

        #исключаем компанию из обучающей выборки
        if (company['Отрасль'] not in mfields['field_name'].tolist()
                or company['Подотрасль'] not in sfields['sub_field_name'].tolist()):
            continue

        print(f'Process company {company["global_id"]}')

        text = info_about[info_about['site'] == company['Сайт']]['info'].values[0]

        if text is None or len(text) == 0:
            text = company['Описание компании']

        tokens = word_tokenize(text) #разбиваем на отдельные слова в прложении

        if len(tokens) == 0:
            continue

        cleaned_tokens = list()

        for token, tag in pos_tag(tokens): #определяем часть речи слова
            if tag.startswith('NN'):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'

            token = lemmatizer.lemmatize(token, pos) #апиводим каждое слово в изначальную форму

            if (len(token) > 0 and
                    token not in string.punctuation and
                    token.lower() not in stop_words and
                    token not in '«»' and
                    token not in string.digits):
                cleaned_tokens.append(token.lower())

        if len(cleaned_tokens) == 0:
            continue

        # --- TF-IDF ---
        print(cleaned_tokens)
        dictionary = gensim.corpora.Dictionary()
        BoW_corpus = dictionary.doc2bow(cleaned_tokens, allow_update=True)
        tfidf = gensim.models.TfidfModel([BoW_corpus], smartirs='ntc')
        df = pd.DataFrame(tfidf[BoW_corpus], columns=['id', 'tfidf']).sort_values('tfidf', ascending=False)

        model = Word2Vec(sentences=[cleaned_tokens], vector_size=64, window=5, min_count=1) #перевод текста в вектор

        trainX.append(model.wv.get_mean_vector(cleaned_tokens, pre_normalize=True)) #среднеарефметический вектор предложения
        trainY.append([
            mfields[mfields['field_name'] == company['Отрасль']]['id'].values[0],
            sfields[sfields['sub_field_name'] == company['Подотрасль']]['id'].values[0]
        ])

    return np.array(trainX), np.array(trainY)


if __name__ == '__main__':
    data = pd.read_excel('./datasets/1. Companies.xlsx')
    main_fields = pd.read_csv('./datasets/main_fields.csv', delimiter=';')
    sub_fields = pd.read_csv('./datasets/sub_fields.csv', delimiter=';')
    info = pd.read_csv('./information.csv', index_col='Unnamed: 0', delimiter=';')

    trainX, trainY = create_dataset(data, main_fields, sub_fields, info)

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

    print('--- Start learning ---')

    keras_model = Sequential()
    keras_model.add(Dense(48, activation='sigmoid', input_shape=(1, 64)))
    keras_model.add(LSTM(48))
    keras_model.add(Dense(32))
    keras_model.add(Dense(8))
    keras_model.add(Dense(2))

    keras_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae']) #компилируется нейроная сеть (алгоритм обратного распространения ошибки; оптимизатор; оценка точности)
    ts_training = keras_model.fit(trainX, trainY, epochs=200, batch_size=2, verbose=2)

    keras_model.save_weights('./nn/nn_weights')

    history_dict = ts_training.history #построение графика
    plt.subplots(1, 1, figsize=(24, 12))
    plt.plot(range(1, 201), history_dict['mae'])
    plt.title('Средняя абсолютная ошибка')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.xticks(range(1, 201))
    plt.grid()
    plt.show()
