"""
Quasi-SVM 출처 : https://keras.io/examples/keras_recipes/quasi_svm/
주의사항 : 데이터 Flatten (reshape (-1, 30000))을 실시할 때 상당한 메모리및 cpu점유율을 차지한다.
메모리가 충분하지 않다면 실행시 컴퓨터가 멈출 수 있다!
"""
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import RandomFourierFeatures

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import os


def imgToArray(path_dir):
    path_dir = path_dir
    file_list = os.listdir(path_dir)

    data_X = []
    data_y = []
    chk = False
    print("Get data from", path_dir)
    for label, file in enumerate(file_list):
        if file == '.DS_Store':
            chk = True
            continue
        if label % 10 == 0:
            print("Now", label, "folders are converted.")
        path_dir2 = path_dir + '/' + file
        file_name_list = os.listdir(path_dir2)
        for file2 in file_name_list:
            if file2 == '.DS_Store':
                continue
            image = Image.open(path_dir2 + '/' + file2)
            val = np.array(image)
            data_X.append(val)
            if chk == True:
                data_y.append(label - 1)
            else:
                data_y.append(label)

    return np.array(data_X), np.array(data_y)


if __name__ == "__main__":

    print('--------------------Loading Dataset--------------------')

    base_dir = '/Users/seni/Desktop/2021-01/AI/PJ'
    img_dir_tr = '/Users/seni/Desktop/2021-01/AI/PJ/fruits-360/Training'
    img_dir_ts = '/Users/seni/Desktop/2021-01/AI/PJ/fruits-360/Test'
    MODEL_SAVE_FOLDER_PATH = './model/'

    '''
    if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
        os.mkdir(MODEL_SAVE_FOLDER_PATH)

    model_path = MODEL_SAVE_FOLDER_PATH + 'fruit-360-' + '{epoch:02d}-{val_loss:.4f}.hdf5'

    cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss',
                                    verbose=1, save_best_only=True)

    cb_early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    '''

    '''
    Set HyperParameter
    '''
    BATCH_SIZE = 16
    EPOCHS = 10
    val_spllit = 0.0

    '''
    # Get Training Data
    path = img_dir_tr
    train_x, train_y = imgToArray(path)

    print("Start Saving Train Data")
    np.save('train_X.npy', train_x)
    np.save('train_y.npy', train_y)
    print("Save Train Data to .npy")

    path = img_dir_ts
    test_x, test_y = imgToArray(path)

    print("Start Saving Test Data")
    np.save('test_X.npy', test_x)
    np.save('test_y.npy', test_y)
    print("Save Test Data to .npy")
    '''

    train_x = np.load('train_X.npy')
    train_y = np.load('train_y.npy')
    test_x = np.load('test_X.npy')
    test_y = np.load('test_y.npy')

    model = keras.Sequential(
        [
            keras.Input(shape=(30000,)),
            RandomFourierFeatures(
                output_dim=4096, scale=10.0, kernel_initializer="gaussian"
            ),
            layers.Dense(units=131),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.hinge,
        metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
    )   

    # Preprocess the data by flattening & scaling it
    print("Now Flatten")
    train_x = train_x.reshape(-1, 30000).astype("float32") / 255
    test_x = test_x.reshape(-1, 30000).astype("float32") / 255

    # Categorical (one hot) encoding of the labels
    train_y = keras.utils.to_categorical(train_y)
    test_y = keras.utils.to_categorical(test_y)

    print('--------------------Training--------------------')

    history = model.fit(train_x, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=None)

    print('\nAccuracy: {:.4f}'.format(model.evaluate(test_x, test_y)[1]))

    print('--------------------Plotting--------------------')

    y_loss = history.history['loss']
    x_len = np.arange(len(y_loss))
    plt.plot(x_len, y_loss, marker='.', c='blue', label="Train-set Loss")

    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    y_acc = history.history['acc']
    x_len = np.arange(len(y_acc))
    plt.plot(x_len, y_acc, marker='.', c='red', label="Train-set Accuracy")
    
    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()
