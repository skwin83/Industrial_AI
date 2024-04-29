from flask import Flask
from flask import request, render_template
import re
from konlpy.tag import Okt
import logging
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


app = Flask(__name__)

stopwords = []
# 불용언 가져오기
with open('stopword.txt', 'r', encoding='utf-8') as f:
    while True:
        line = f.readline()
        if not line: break
        stopwords.append(line.split('\n')[0])

# 정수 인코딩을 위한 토큰 가져오기
with open('pickle/train_tokenizer.pickle', 'rb') as f:
    train_tokenizer = pickle.load(f)

with open('pickle/category_dict.pickle', 'rb') as f:
    category_dict = pickle.load(f)
with open('pickle/category_to_num_dict.pickle', 'rb') as f:
    category_to_num_dict = pickle.load(f)

print(category_dict)

# 베스트 모델 가져오기
train_model = load_model('model/train_best_model.h5')



def parse_news_request(request):
    title = request.form.get('title')
    content = request.form.get('content')
    return title, content

def preprocesse_normalization(title, content):
    # 정규화
    title = re.sub("[^0-9a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣 ]", "", title)
    content = re.sub("[^0-9a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣 ]", "", content)
    # 공백제거
    title = re.sub(' +', ' ', title)
    content = re.sub(' +', ' ', content)
    return title, content

def tokenization(data, stopwords):
    okt = Okt()
    tokenized_sentence = okt.nouns(data)  # 명사 추출
    stopwords_removed_sentence = [[word for word in tokenized_sentence if not word in stopwords]]  # 불용어 제거
    
    return stopwords_removed_sentence

def encode_string_to_int(train):
    train = train_tokenizer.texts_to_sequences(train)
    return train

def generate_padding(train):
    return pad_sequences(train, maxlen=700, padding='post')

def predict_data(data, model):
    predicted = model.predict(data)
    idx = np.argmax(predicted)
    category_num_to_num_dict = dict(map(reversed, category_to_num_dict.items()))
    category_idx = category_num_to_num_dict[idx]
    category = category_dict[category_idx]
    percentage = round(np.max(predicted) * 100, 1)
    return category, percentage


@app.route('/')
def main():
    return render_template('main.html')

@app.route(rule='/getsearch/', methods=['GET'])
def get_news():
    return render_template('get_news.html')

@app.route(rule='/postsearch/', methods=['POST'])
def post_news():
    # 1. 데이터 파싱
    title, content = parse_news_request(request)

    # 2. 정규표현식 전처리
    title, content = preprocesse_normalization(title, content)

    # 3. 토큰화
    title = tokenization(title, stopwords)
    content = tokenization(content, stopwords)
    train = [title[0] + content[0]]

    # 4. 정수 인코딩
    train = encode_string_to_int(train)

    # 5. 패딩 생성
    train = generate_padding(train)

    # 6. 모델 예측
    train_category, train_percentage = predict_data(train, train_model)
    infologger.info(train_category)
    infologger.info(train_percentage)

    
    return  render_template(
        'post_news.html',
        train_category = train_category,
        train_percentage = train_percentage
        )


if __name__ == '__main__':
    # DEBUG < INFO < WARNING < ERROR < CRITICAL
    infologger = logging.getLogger()
    infologger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_hander = logging.StreamHandler()
    stream_hander.setFormatter(formatter)
    infologger.addHandler(stream_hander)

    file_handler = logging.FileHandler('my.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    infologger.addHandler(file_handler)

    infologger.info('start')

    app.run(host='0.0.0.0', port=15000, debug=True)