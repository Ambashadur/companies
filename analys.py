import pandas as pd
import string
import numpy as np
import tensorflow as tf

from gensim.models import Word2Vec

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize


def get_vector(text):
    stop_words = stopwords.words('russian')
    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = list()
    tokens = word_tokenize(text)

    if len(tokens) == 0:
        return list()

    for token, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())

    if len(cleaned_tokens) == 0:
        return list()

    model = Word2Vec(sentences=[cleaned_tokens], vector_size=64, window=5, min_count=1)

    return np.array([model.wv.get_mean_vector(cleaned_tokens, pre_normalize=True)])


if __name__ == '__main__':
    data = pd.read_excel('./datasets/1. Companies.xlsx')
    main_fields = pd.read_csv('./datasets/main_fields.csv', delimiter=';')
    sub_fields = pd.read_csv('./datasets/sub_fields.csv', delimiter=';')

    keras_model = Sequential()
    keras_model.add(Dense(48, activation='sigmoid', input_shape=(1, 64)))
    keras_model.add(LSTM(48))
    keras_model.add(Dense(32))
    keras_model.add(Dense(8))
    keras_model.add(Dense(2))

    keras_model.load_weights('./nn/nn_weights')
    keras_model.build()
    keras_model.summary()

    for i in data.index:
        company = data.loc[i]
        vector = get_vector(company['Описание компании'])
        vector = tf.expand_dims(vector, axis=0)
        prediction = keras_model.predict(vector)
        field = main_fields[main_fields['id'] == round(prediction[0][0])]['field_name'].values[0]
        sub_field = sub_fields[sub_fields['id'] == round(prediction[0][1])]['sub_field_name'].values[0]

        print(f'global_id: {company["global_id"]} filed - {field} sub_field - {sub_field}')
