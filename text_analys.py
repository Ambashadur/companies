import pandas as pd
import nltk
import string

from gensim.models import Word2Vec

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize


def create_dataset(samples: pd.DataFrame):
    stop_words = stopwords('russian')
    trainX, trainY = list(), list()

    for i in samples.index:
        company = samples.loc[i]
        tokens = word_tokenize(company['Описание компании'])
        cleaned_tokens = list()

        for token, tag in pos_tag(tokens):
            if tag.startswith('NN'):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'

            lemmatizer = WordNetLemmatizer()
            token = lemmatizer.lemmatize(token, pos)

            if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
                cleaned_tokens.append(token.lower())

        foo = 0


if __name__ == '__main__':
    data = pd.read_excel('./datasets/1. Companies.xlsx')
    main_fields = pd.read_csv('./datasets/main_fields.csv')
    sub_fields = pd.read_csv('datasets/sub_fields.csv')

# keras_model = Sequential()
# keras_model.add(Dense(64, activation='relu', input_shape=(1, 128)))
# keras_model.add(LSTM(32))
# keras_model.add(Dense(16))
# keras_model.add(Dense(8))
# keras_model.add(Dense(2))

#keras_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
#ts_training = keras_model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)