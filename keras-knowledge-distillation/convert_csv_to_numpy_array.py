import numpy as np
from math import ceil
import pickle
import png
from numpy import loadtxt
from numpy import reshape
from PIL import Image
import os

def save_image_data():
    train_file_path = "sign_mnist_train.csv"
    test_file_path = "sign_mnist_test.csv"
    # x_train is a uint8 array of grayscale image data with shape (num_samples, 28, 28).
    # y_train is a uint8 array of digit labels (integers in range 0-9) with shape (num_samples,).
    num_train = 20592
    num_val = 6863
    num_test = 7172
    height = 28
    width = 28
    imDimensions = (height,width)
    scaledHeight = 84
    scaledWidth = 84
    channels = 3
    scaledDimensions = (scaledHeight,scaledWidth)
    # image data format of samples,channels,height,width
    trainDimensions = (num_train,channels,scaledHeight,scaledWidth)
    valDimensions = (num_val,channels,scaledHeight,scaledWidth)
    testDimensions = (num_test,channels,scaledHeight,scaledWidth)
    # Initialize the output arrays
    x_train = np.empty(trainDimensions,dtype='uint8')
    y_train = np.empty((num_train,),dtype='uint8')
    x_val = np.empty(valDimensions,dtype='uint8')
    y_val = np.empty((num_val,),dtype='uint8')
    x_test = np.empty(testDimensions,dtype='uint8')
    y_test = np.empty((num_test,),dtype='uint8')

    # initialize array used to store each image as RGB
    rgbPixels = np.empty((channels,scaledHeight,scaledWidth),dtype='uint8')
    # load train dataset (skipping over header row) and convert to list
    sheet = np.loadtxt(train_file_path,dtype=int,delimiter=',',skiprows=1)
    data = sheet.tolist()
    row = 1
    # process each training image in data
    for pic in data:
        # convert CSV to image and scale up, then convert image back to array
        pixels = np.array(pic[1:])
        pixels = np.reshape(pixels,imDimensions)
        imagePixels = Image.fromarray(pixels.astype(np.uint8))
        scaledPixels = imagePixels.resize(scaledDimensions)
        arrayPixels = np.asarray(scaledPixels)
        # store arrayPixels in rgbPixels to simulate RGB image
        for i in range(3):
            rgbPixels[i] = np.reshape(arrayPixels,scaledDimensions)
        # put every 4th label into y_val
        # reshape image data into 3x84x84 and store every 4th image in x_val
        if (row % 4 == 0):
            index = int(row/4 - 1)
            y_val[index] = pic[0]
            x_val[index] = rgbPixels
        else:
            index = int(row - ceil(row/4))
            y_train[index] = pic[0]
            x_train[index] = rgbPixels

        row += 1
    
    # load test dataset (skipping over header row) and convert to list
    sheet = np.loadtxt(test_file_path,dtype=int,delimiter=',',skiprows=1)
    data = sheet.tolist()
    row = 0
    # process each image in data
    for pic in data:
        # convert CSV to image and scale up, then convert image back to array
        pixels = np.array(pic[1:])
        pixels = np.reshape(pixels,imDimensions)
        imagePixels = Image.fromarray(pixels.astype(np.uint8))
        scaledPixels = imagePixels.resize(scaledDimensions)
        arrayPixels = np.asarray(scaledPixels)
        # store arrayPixels in rgbPixels to simulate RGB image
        for i in range(3):
            rgbPixels[i] = np.reshape(arrayPixels,scaledDimensions)
        # put picture label into y_test
        y_test[row] = pic[0]
        # reshape image data into 28x28 and store in x_test
        x_test[row] = rgbPixels
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
    with open('sign_mnist_numpy_scaled_rgb_data.pickle','wb') as f:
        pickle.dump(data,f)

main()
