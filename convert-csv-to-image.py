from numpy import loadtxt
import png
import os

def chunks(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]

def main():
    # load mnist dataset, skipping over header row
    valFolder = "validation_images"
    trainFolder = "training_images"
    numTrain = 0
    numVal = 0
    sheet = loadtxt('sign_mnist_train.csv',dtype=int,delimiter=',',skiprows=1)
    # convert dataset to list
    data = sheet.tolist()
    rowNum = 1
    w = png.Writer(width=28,height=28,greyscale=True)
    for row in data:
        # get folder name from first column
        folder = row[0]
        # get image data and convert it into 2d array of 28x28
        pixels = row[1:]
        image = chunks(pixels,28)
        # Use every 4th image for validation instead of training
        if (rowNum % 4 == 0):
            useFolder = valFolder
            numVal += 1
        else:
            useFolder = trainFolder
            numTrain += 1
        # Create folder and path to folder
        folderPath = useFolder + "/" + chr(folder+65)
        if not os.path.isdir(folderPath):
            os.mkdir(folderPath)
        # create image in folder
        name = str(rowNum) + ".png"
        fPath = folderPath + "/" + name
        f = open(fPath,'wb')
        # write row as greyscale image of 28 x 28
        w.write(f,image)
        f.close()
        rowNum = rowNum + 1

    print("Total number of training images: ", numTrain)
    print("Total number of validation images: ", numVal)
main()