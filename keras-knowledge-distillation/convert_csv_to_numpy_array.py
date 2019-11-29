import numpy as np
from math import ceil
import pickle
from numpy import loadtxt
from numpy import reshape
import os

def save_image_data():
    train_file_path = "sign_mnist_train.csv"
    test_file_path = "sign_mnist_test.csv"
    # x_train is a uint8 array of grayscale image data with shape (num_samples, 28, 28).
    # y_train is a uint8 array of digit labels (integers in range 0-9) with shape (num_samples,).
    num_train = 20592
    num_val = 6863
    num_test = 7172
    imHeight = 28
    imWidth = 28
    imDimensions = (imHeight,imWidth)
    trainDimensions = (num_train,imHeight,imWidth)
    valDimensions = (num_val,imHeight,imWidth)
    testDimensions = (num_test,imHeight,imWidth)
    # Initialize the output arrays
    x_train = np.empty(trainDimensions,dtype='uint8')
    y_train = np.empty((num_train,),dtype='uint8')
    x_val = np.empty(valDimensions,dtype='uint8')
    y_val = np.empty((num_val,),dtype='uint8')
    x_test = np.empty(testDimensions,dtype='uint8')
    y_test = np.empty((num_test,),dtype='uint8')

    # load train dataset (skipping over header row) and convert to list
    sheet = np.loadtxt(train_file_path,dtype=int,delimiter=',',skiprows=1)
    data = sheet.tolist()
    row = 1
    # process each training image in data
    for pic in data:
        pixels = pic[1:]
        # put every 4th label into y_val
        # reshape image data into 28x28 and store every 4th image in x_val
        if (row % 4 == 0):
            index = int(row/4 - 1)
            y_val[index] = pic[0]
            x_val[index] = np.reshape(pixels,imDimensions)
        else:
            index = int(row - ceil(row/4))
            y_train[index] = pic[0]
            x_train[index] = np.reshape(pixels,imDimensions)

        row += 1
    
    # load test dataset (skipping over header row) and convert to list
    sheet = np.loadtxt(test_file_path,dtype=int,delimiter=',',skiprows=1)
    data = sheet.tolist()
    row = 0
    # process each image in data
    for pic in data:
        # put picture label into y_test
        y_test[row] = pic[0]
        # reshape image data into 28x28 and store in x_test
        pixels = pic[1:]
        x_test[row] = np.reshape(pixels,imDimensions)
        row += 1

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
    
def main():
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = save_image_data()
    print("x_train.shape = ", x_train.shape)
    print("y_train.shape = ", y_train.shape)

    print("x_train[0] = ",x_train[0])
    print("y_train[0] = ", y_train[0])

    print("x_test.shape = ", x_test.shape)
    print("y_test.shape = ", y_test.shape)

    print("x_test[0] = ",x_test[0])
    print("y_test[0] = ", y_test[0])

    print("x_val.shape = ", x_val.shape)
    print("y_val.shape = ", y_val.shape)

    print("x_val[0] = ",x_val[0])
    print("y_val[0] = ", y_val[0])
    data = ((x_train, y_train), (x_val, y_val), (x_test, y_test))
    with open('sign_mnist_numpy_data.pickle','wb') as f:
        pickle.dump(data,f)

main()
