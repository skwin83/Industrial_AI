import os
import numpy as np
from PIL import Image

'''
이미지 폴더를 방문하여 그 이미지들과 label 들을 Numpy Array로 바꿔주는 과정이다.
'''

def imgToArray(path_dir):
    path_dir = path_dir
    file_list = os.listdir(path_dir)
    data_X = []
    data_y = []
    print("Get data from", path_dir)
    for label, file in enumerate(file_list):
        if label % 10 == 0:
            print("Now", label, "folders are converted.")
        path_dir2 = path_dir + '\\' + file
        file_name_list = os.listdir(path_dir2)
        for file2 in file_name_list:
            image = Image.open(path_dir2 + '\\' + file2)
            val = np.array(image)
            data_X.append(val)
            data_y.append(label)

    return np.array(data_X), np.array(data_y)


if __name__ == "__main__":
    # Get Training Data
    path = r'fruits-360/Training'
    train_X, train_y = imgToArray(path)

    print("Start Saving Train Data")
    np.save('train_X.npy', train_X)
    np.save('train_y.npy', train_y)
    print("Save Train Data to .npy")

    # Get Test Data
    path = r'fruits-360/Test'
    test_X, test_y = imgToArray(path)

    print("Start Saving Test Data")
    np.save('test_X.npy', test_X)
    np.save('test_y.npy', test_y)
    print("Save Test Data to .npy")
