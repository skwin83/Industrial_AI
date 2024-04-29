import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

from konlpy.tag import Mecab

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.utils import to_categorical

# 카테고리 매칭 딕셔너리 생성
def generate_category_dict(data):
    data = data.copy()
    data = data[['cate_code1', 'cate_name1']].set_index('cate_code1')
    category_dict = data.to_dict()['cate_name1']
    # with open('flask/pickle/category_dict.pickle', 'wb') as f:
    #     pickle.dump(category_dict, f)

    # 중구난방인 숫자들을 0~29에 매칭하기!
    category_num_dict = {}
    for idx, key in enumerate(category_dict.keys()):
        category_num_dict[key] = idx
    # with open('flask/pickle/category_to_num_dict.pickle', 'wb') as f:
    #     pickle.dump(category_num_dict, f)

def read_stopwords():
    stopwords = []
    with open('flask/stopword.txt', 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line: break
            stopwords.append(line.split('\n')[0])
    return stopwords

class Preprocessor:
    def __init__(self, data):
        self.data = data

    def delete_missing_value(self):
        data = self.data.copy()
        data = data.dropna()
        return data

    def delete_duplicates(self):
        data = self.data.copy()
        data = data.drop_duplicates()
        return data

    def delete_column(self):
        data = self.data.copy()
        data = data[['title', 'content', 'category']]
        return data

    def preprocesse_normalization(self, data=None):
        if not data:
            data = self.data.copy()
        # 태그 제거
        data['content'] = data['content'].str.replace("<.+?>", "")

        # 정규화
        data['content'] = data['content'].str.replace("[^0-9a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
        data['title'] = data['title'].str.replace("[^0-9a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
        data['content'] = data['content'].str.replace("제목 없음", "")

        # 공백 제거
        data['content'] = data['content'].str.replace(' +', ' ')
        data['title'] = data['title'].str.replace(' +', ' ')

        data = data.dropna()

        # data.to_csv('news_data.csv', encoding='utf-8', index_label=False)
        return data

    def tokenization(self, stopwords, data_type):
        data = self.data.copy()
        # .morphs: 형태소 분석, .pos: 품사 태깅, .nouns: 명사 추출
        mecab = Mecab(dicpath="C:\\mecab\\mecab-ko-dic")
        train_input = []
        for sentence in tqdm(data[data_type]):
            tokenized_sentence = mecab.nouns(sentence)
            stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]  # 불용어 제거
            train_input.append(stopwords_removed_sentence)

        # with open(f'flask/pickle/{data_type}_token.pickle', 'wb') as f:
        #     pickle.dump(train_input, f)

    def encode_string_to_int(self, token, token_type):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(token)
        if token_type == 'title':
            threshold = 3
        else:
            threshold = 1500
        total_cnt = len(tokenizer.word_index)
        rare_cnt = 0  # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트

        # 데이터 확인
        total_freq = 0  # 훈련 데이터의 전체 단어 빈도수 총 합
        rare_freq = 0  # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

        for key, value in tokenizer.word_counts.items():
            total_freq += value
            # 단어 빈도수가 4보다 작으면
            if value < threshold:
                rare_cnt += 1
                rare_freq += value
        print(token_type)
        print('단어 집합(vocabulary)의 크기 :', total_cnt)
        print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s' % (threshold - 1, rare_cnt))
        print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt) * 100)
        print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq) * 100)
        print("------------------------------------------------------------")

        vocab_size = total_cnt - rare_cnt + 1
        tokenizer = Tokenizer(vocab_size)
        tokenizer.fit_on_texts(token)

        # with open(f'flask/pickle/{token_type}_tokenizer.pickle', 'wb') as f:
        #     pickle.dump(tokenizer, f)

        return tokenizer.texts_to_sequences(token), vocab_size

    def sort_target(self, train_target):
        # category_to_num_dict = {1: 0, 27: 1,...}
        with open('flask/pickle/category_to_num_dict.pickle', 'rb') as f:
            category_num_dict = pickle.load(f)

        drop_train_target = []
        for i in range(len(train_target)):
            if train_target[i] not in category_num_dict:
                drop_train_target.append(i)
            else:
                train_target[i] = category_num_dict[train_target[i]]

        return train_target, drop_train_target

    def delete_empty_sample(self, data, category, drop_train_target):
        # 빈 샘플 제거를 위한 인덱스 추출
        drop_data = [index for index, sentence in enumerate(data) if len(sentence) < 1]
        drop_train = list(set(drop_data + drop_train_target))
        drop_train.sort()
        # 빈 샘플 제거
        data = np.delete(data, drop_train, axis=0)
        train_target = np.delete(category, drop_train, axis=0)
        return data, train_target

    def generate_padding(self, train, train_type):
        if train_type == 'title':
            max_len = 10
        else:
            max_len = 700
        return pad_sequences(train, maxlen=max_len, padding='post')


class ModelFactory:
    def __init__(self, title_input, content_input, train_input, title_target, content_target, train_target):
        self.title_input = title_input
        self.content_input = content_input
        self.train_input = train_input
        self.title_target = title_target
        self.content_target = content_target
        self.train_target = train_target

    def LSTM(self, vocab_size, train_type):
        if train_type == 'title':
            train_input = self.title_input
            train_target = to_categorical(self.title_target)
        elif train_type =='content':
            train_input = self.content_input
            train_target = to_categorical(self.content_target)
        else:
            train_input = self.train_input
            train_target = to_categorical(self.train_target)

        # 임베딩 벡터의 차원 : 128, 은닉 상태의 크기 : 128, 모델 : 다 대 다 구조의 LSTM
        embedding_dim = 128
        hidden_units = 128
        num_classes = 28    # 28개의 카테고리

        model = Sequential()
        model.add(Embedding(vocab_size, embedding_dim))
        if train_type == 'title':
            model.add(Bidirectional(LSTM(hidden_units)))
        else:
            model.add(LSTM(hidden_units))

        model.add(Dense(num_classes, activation='softmax'))

        print(model.summary())
        # 검증데이터 손실(val_loss)이 증가하면, 과적합 징후므로 검증 데이터 손실이 4회 증가하면
        # 정해진 에포크가 도달하지 못하였더라도 학습을 조기 종료(Early Stopping)
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
        # 검증 데이터의 정확도(val_acc)가 이전보다 좋아질 경우에만 모델을 저장
        model_checkpoint = ModelCheckpoint(f'{train_type}_best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

        model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
        # 훈련 데이터의 20%를 검증 데이터로 분리해서 사용
        # 검증 데이터를 통해서 훈련이 적절히 되고 있는지 확인
        history = model.fit(train_input, train_target, epochs=8, callbacks=[early_stop, model_checkpoint], batch_size=32, validation_split=0.2)
        loaded_model = load_model(f'{train_type}_best_model.h5')
        print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(self.train_input, train_target)[1]))


class DataAnalysis:
    def __init__(self, title, content, train_input):
        self.title = title
        self.content = content
        self.train_input = train_input

    def show_sentence_lengh(self, type):
        if type =='title':
            data = self.title
            max_len = 10
        elif type == 'content':
            data = self.content
            max_len = 700
        else:
            data = self.train_input
            max_len = 700
        print(f'{type}의 최대 길이 :', max(len(review) for review in data))
        print(f'{type}의 평균 길이 :', sum(map(len, data)) / len(data))
        plt.hist([len(review) for review in data], bins=50)
        plt.title(type)
        plt.xlabel('length of samples')
        plt.ylabel('number of samples')
        plt.show()
        count = 0
        for sentence in data:
            if (len(sentence) <= max_len):
                count = count + 1
        print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s' % (max_len, (count / len(data)) * 100))

    def show_model_loss(self):
        with open('train_history', 'rb') as f:
            history = pickle.load(f)
        epochs = range(1, len(history['accuracy']) + 1)
        plt.plot(epochs, history['loss'])
        plt.plot(epochs, history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)

    # 1. 데이터 전처리
    data = pd.read_csv('news_data.csv')
    pre = Preprocessor(data)

    # 1-1) 결측치 제거
    data = pre.delete_missing_value()
    pre.data = data

    # 1-2) 중복값 제거
    data = pre.delete_duplicates()
    pre.data = data

    # 1-3) 컬럼 정리
    data = pre.delete_column()
    pre.data = data

    # 1-4) 정규화
    data = pre.preprocesse_normalization()
    pre.data = data

    # 1-5) 토큰화 및 저장
    stopwords = read_stopwords()    # 불용언 불러오기
    pre.tokenization(stopwords, 'title')
    pre.tokenization(stopwords, 'content')

    # 1-5-1) 토큰 불러오기
    with open('flask/pickle/title_token.pickle', 'rb') as f:
        title_token = pickle.load(f)
    with open('flask/pickle/content_token.pickle', 'rb') as f:
        content_token = pickle.load(f)
    train_token = []
    for i in range(len(title_token)):
        train_token.append(title_token[i] + content_token[i])

    # 1-6) 정수 인코딩
    title_input, title_vocab_size = pre.encode_string_to_int(title_token, 'title')
    content_input, content_vocab_size = pre.encode_string_to_int(content_token, 'content')
    train_input, train_vocab_size = pre.encode_string_to_int(train_token, 'train')

    # 1-7) 정답 값 전처리
    train_target = np.array(data['category'])
    train_target, drop_train_target = pre.sort_target(train_target)

    # 1-8) 빈 샘플 제거
    title_input, title_target = pre.delete_empty_sample(title_input, train_target, drop_train_target)
    content_input, content_target = pre.delete_empty_sample(content_input, train_target, drop_train_target)
    train_input, train_target = pre.delete_empty_sample(train_input, train_target, drop_train_target)

    # 1-9) 패딩
    # # 1-9-1) 패딩 사이즈를 정하기 위한 데이터 분석
    # da = DataAnalysis(title_input, content_input, train_input)
    # da.show_sentence_lengh('title')
    # da.show_sentence_lengh('content')
    # da.show_sentence_lengh('train')

    # 1-9-2) 패딩 작업
    title_input = pre.generate_padding(title_input, 'title')
    content_input = pre.generate_padding(content_input, 'content')
    train_input = pre.generate_padding(train_input, 'train')

    # # 2. 모델링
    dl_model = ModelFactory(title_input, content_input, train_input, title_target, content_target, train_target)
    # 2-1) title만으로 모델링
    dl_model.LSTM(title_vocab_size, 'title')

    # 2-2) content만으로 모델링
    dl_model.LSTM(content_vocab_size, 'content')

    # 2-3) title + content로 모델링
    dl_model.LSTM(train_vocab_size, 'train')
