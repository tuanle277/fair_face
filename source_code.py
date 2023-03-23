import os 
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import timeit
import json
import matplotlib.pyplot as plt
from matplotlib.image import imread
from getHParams import *
# from scipy.misc import imread

from PIL import Image

from tensorflow.keras.preprocessing.image import ImageDataGenerator

label_file = "fairface_label_train.csv"

# def save_image(image_links):
#     aa =  np.array([[[[0] * 3] * 200] * 200])

#     for i in image_links:
#         a = Image.open("UTKFace/" + i)
#         a = np.asarray(a)
#         if len(a.shape) == 3 and i != 507:
#             aa = np.concatenate((aa, [a]), axis=0)
#         print(aa.shape)

#     np.save("image_data_0", aa) 

# def get_data(num_files):
#     aa =  np.array([[[[0] * 3] * 200] * 200])
#     for i in range(num_files):
#         image_data = np.load("image_data_" + str(i) + ".npy")
#         image_data = np.delete(image_data, 0, 0)
#         aa = np.concatenate((aa, image_data))

#     aa = np.delete(aa, 0, 0)

#     return aa


# def getLabel(image_links):
#     df = pd.read_csv(label_file)

#     labels = [] * len(df)
#     for i in range(len(image_links)):
#         label[image_links[i]] = df.iloc[i].race

#     return np.array(labels)

def getTrainData():
    image_source = "fairface-img-margin025-trainval/train/"
    image_links = os.listdir(image_source)

    aa =  np.array([[[[0] * 3] * 224] * 224])

    for i in image_links:
        a = imread(image_source + i)
        if len(a.shape) == 3:
            aa = np.concatenate((aa, [a]), axis=0)
        print(aa.shape)


    aa = np.delete(aa, 0, 0)

    np.save("image_data", aa) 


def get10ClassData(hParams, flatten=True, proportion=1.0):
    # == get the data set == #
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # plt.figure(figsize=(4, 4))
    # image = x_train[np.random.choice(range(x_train.shape[0]))]
    # plt.imshow(image.astype("uint8"))
    # plt.axis("off")

    # resized_image = tf.image.resize(
    #     tf.convert_to_tensor([image]), size=(x_train.shape[0], x_train.shape[0])
    # )
    # patches = Patches(patch_size)(resized_image)
    # print(f"Image size: {x_train.shape[0]} X {x_train.shape[0]}")
    # print(f"Patch size: {patch_size} X {patch_size}")
    # print(f"Patches per image: {patches.shape[1]}")
    # print(f"Elements per patch: {patches.shape[-1]}")

    # n = int(np.sqrt(patches.shape[1]))
    # plt.figure(figsize=(4, 4))
    # for i, patch in enumerate(patches[0]):
    #     ax = plt.subplot(n, n, i + 1)
    #     patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    #     plt.imshow(patch_img.numpy().astype("uint8"))
    #     plt.axis("off")

    # == slice the dataset == #
    x_train = x_train[:int(proportion * x_train.shape[0]):]
    y_train = y_train[:int(proportion * y_train.shape[0]):]
    x_test = x_test[:int(proportion * x_test.shape[0]):]
    y_test = y_test[:int(proportion * y_test.shape[0]):]

    # == print the shape and structure == #
    print(x_train.shape, x_train)
    print(y_train.shape, y_train)
    print(x_test.shape, x_test)
    print(y_test.shape, y_test)

    # == convert to float == #
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255
    x_test = x_test / 255 

    # == flatten == #
    # pass False when called 
    if flatten:
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

    # == Slice for validation data == #
    x_val = x_train[:int(hParams['valProportion'] * x_train.shape[0])]
    y_val = y_train[:int(hParams['valProportion'] * y_train.shape[0])]

    x_train = x_train[int(hParams['valProportion'] * x_train.shape[0]):]
    y_train = y_train[int(hParams['valProportion'] * y_train.shape[0]):]

    if hParams['valProportion'] != 0.0:
        return x_train, y_train, x_test, y_test, x_val, y_val

    return x_train, y_train, x_test, y_test


def correspondingShuffle(x, y):
    indices = tf.range(start=0, limit=tf.shape(x)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)

    shuffled_x = tf.gather(x, shuffled_indices)
    shuffled_y = tf.gather(y, shuffled_indices)

    return shuffled_x, shuffled_y


# plot diagnostic learning curves
def summarize_diagnostics(history, name):
    # plot loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history['loss'], color='blue', label='train')
    plt.plot(history['val_loss'], color='orange', label='test')
    # plot accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history['accuracy'], color='blue', label='train')
    plt.plot(history['val_accuracy'], color='orange', label='test')
    # save plot to file
    plt.savefig("cifar10_res/" + name + '_plot.png')
    plt.close()

def writeExperimentalResults(hParams, trainResults, testResults):
    # == open file == #
    f = open("results/" + hParams["experimentName"] + ".txt", 'w')

    # == write in file == #
    f.write(str(hParams) + '\n\n')
    f.write(str(trainResults) + '\n\n')
    f.write(str(testResults))

    # == close file == #
    f.close()

def readExperimentalResults(fileName):
    f = open("results/" + fileName + ".txt",'r')

    # == read in file == #
    data = f.read().split('\n\n')

    # == process data to json-convertible == #
    data[0] = data[0].replace("\'", "\"")
    data[1] = data[1].replace("\'", "\"")
    data[2] = data[2].replace("\'", "\"")

    # == convert to json == #
    hParams = json.loads(data[0])
    trainResults = json.loads(data[1])
    testResults = json.loads(data[2])

    return hParams, trainResults, testResults

def plotCurves(x, yList, xLabel="", yLabelList=[], title=""):
    fig, ax = plt.subplots()
    y = np.array(yList).transpose()
    ax.plot(x, y)
    ax.set(xlabel=xLabel, title=title)
    plt.legend(yLabelList, loc='best', shadow=True)
    ax.grid()
    yLabelStr = "__" + "__".join([label for label in yLabelList])
    filepath = "results/" + title + " " + yLabelStr + ".png"
    fig.savefig(filepath)
    print("Figure saved in", filepath)


def plotPoints(xList, yList, pointLabels=[], xLabel="", yLabel="", title="", filename="pointPlot"):
    plt.figure()
    plt.scatter(xList,yList)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    if pointLabels != []:
        for i, label in enumerate(pointLabels):
            plt.annotate(label, (xList[i], yList[i]))
    filepath = "results/" + filename + ".png"
    plt.savefig(filepath)
    print("Figure saved in", filepath)

def buildValAccuracyPlot(fileNames, title):
    # == get hParams == #
    hParams = readExperimentalResults(fileNames[0])[0]

    # == plot curves with yList being the validation accuracies == #
    plotCurves(x=np.arange(0, hParams["numEpochs"]), 
            yList=[readExperimentalResults(name)[1]['val_accuracy'] for name in fileNames], 
            xLabel="Epoch",
            yLabelList=fileNames,
            title= title)


def cnnRBG(dataSubsets, hParams):
    x_train, y_train, x_test, y_test, x_val, y_val = dataSubsets

    width, height = x_train.shape[1], x_train.shape[2]

    x_train = tf.reshape(x_train, (-1, width, height, 3))
    x_val = tf.reshape(x_val, (-1, width, height, 3))
    x_test = tf.reshape(x_test, (-1, width, height, 3))

    # == Shuffle data == #
    x_train, y_train = correspondingShuffle(x_train, y_train)
    x_test, y_test = correspondingShuffle(x_test, y_test)

    # == Sequential Constructor == #
    startTime = timeit.default_timer()
    model = tf.keras.Sequential()
    for layer in hParams['convLayers']:
        model.add(tf.keras.layers.Conv2D(layer['conv_numFilters'], layer['conv_f'], activation=layer['conv_act'], input_shape=(width, height, 3), padding=layer['conv_p']))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(layer['pool_f'], layer['pool_s'])))
        model.add(tf.keras.layers.Dropout(layer['drop_prop']))

    model.add(tf.keras.layers.Flatten())
    for layer in range(len(hParams['denseLayers'])):
        if layer < len(hParams['denseLayers']) - 1:
            model.add(tf.keras.layers.Dense(hParams['denseLayers'][layer], activation='relu'))
        else:
            model.add(tf.keras.layers.Dense(hParams['denseLayers'][layer]))

    # == Loss function == #
    lossFunc = tf.keras.losses.SparseCategoricalCrossentropy(True)

    # == fitting == #
    model.compile(loss=lossFunc, metrics=['accuracy'], optimizer=hParams['optimizer'])
    hist = model.fit(x_train, y_train, 
                    validation_data=(x_val, y_val) 
                                                if hParams['valProportion']!=0.0 
                                                else None, 
                    epochs=hParams['numEpochs'],
                    verbose=1)  
    trainingTime = timeit.default_timer() - startTime

    # == Evaluation == #
    print('============ one unit 10 class, training set size:', x_train.shape[0], ' =============')
    print(model.summary())
    print('Training time:', trainingTime)
    print(model.evaluate(x_test, y_test))
    hParams['paramCount'] = model.count_params()

    return hist.history, model.evaluate(x_test, y_test)



def main():
    theSeed = 50
    np.random.seed(theSeed)
    tf.random.set_seed(theSeed)

    hParams = {
        "datasetProportion": 1.0,
        "numEpochs": 20,
        "denseLayers": [128, 10],
        "valProportion": 0.1,
        "experimentName": "128_10e20",
        "optimizer": "rmsprop"
    }

    expNames = [
    
        'C32_64__d0.0__D128_10__rms', 
        'C32_64__d0.2__D128_10__rms', 
        'C32_64__d0.0__D128_10__adam',
        'C32_64__d0.2__D128_10__adam',

        'C32_64__d0.02__D128_10__adam',
        'C32_64__d0.05__D128_10__adam',
        'C32_64__d0.1__D128_10__adam',
        'C32_64__d0.3__D128_10__adam',

        'C32__d0.2__D128_10__adam',
        'C32_64_128__d0.2__D128_10__adam', 
        'C32_64_128_256__d0.2__D128_10__adam',

        'C32_64__d0.2__D256_128_10__adam', 
        'C32_64__d0.2__D512_256_128_10__adam'

        'C32__d0.2__D128_10__rms',
        'C32_64_128__d0.2__D128_10__rms', 
        'C32_64_128_256__d0.2__D128_10__rms',
        'C32_64__d0.2__D128_10__rms_test'
    ]

    dataSubsets = get10ClassData(hParams, False)
    for currExp in expNames:
        hParams = getHParams(currExp)
        trainResults, testResults = cnnRBG(dataSubsets, hParams)
        summarize_diagnostics(trainResults, currExp)
        writeExperimentalResults(hParams, trainResults, testResults)



main()
fileNames = [
            [
            'C32_64__d0.0__D128_10__adam',
            'C32_64__d0.2__D128_10__adam',
            'C32_64__d0.02__D128_10__adam',
            'C32_64__d0.05__D128_10__adam',
            'C32_64__d0.1__D128_10__adam',
            'C32_64__d0.3__D128_10__adam'
            ],
            [
            'C32_64__d0.0__D128_10__rms', 
            'C32_64__d0.2__D128_10__rms', 
            'C32_64__d0.0__D128_10__adam',
            'C32_64__d0.2__D128_10__adam'
            ],
            [     
            'C32__d0.2__D128_10__adam',
            'C32_64__d0.2__D128_10__adam', 
            'C32_64_128__d0.2__D128_10__adam', 
            'C32_64_128_256__d0.2__D128_10__adam'
            ], 
            [     
            'C32_64__d0.2__D128_10__adam', 
            'C32_64__d0.2__D256_128_10__adam', 
            'C32_64__d0.2__D512_256_128_10__adam'
            ],
            [       
            'C32__d0.2__D128_10__rms',
            'C32_64__d0.2__D128_10__rms', 
            'C32_64_128__d0.2__D128_10__rms', 
            'C32_64_128_256__d0.2__D128_10__rms'
            ]
        ]

# for index in range(len(fileNames)):
#   buildValAccuracyPlot(fileNames[index], "val_accuracy_" + str(index))
