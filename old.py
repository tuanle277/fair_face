import os 
import numpy as np
import pandas as pd
import tensorflow as tf
import timeit
import json
import matplotlib.pyplot as plt
from matplotlib.image import imread

from PIL import Image

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def save_image(image_links):
    aa =  np.array([[[[0] * 3] * 200] * 200])

    for i in image_links[:6000]:
        a = Image.open("UTKFace/" + i)
        a = np.asarray(a)
        if len(a.shape) == 3 and i != 507:
            aa = np.concatenate((aa, [a]), axis=0)
        print(aa.shape)

    np.save("image_data_0", aa) 

def get_data(num_files):
    aa =  np.array([[[[0] * 3] * 200] * 200])
    for i in range(num_files):
        image_data = np.load("image_data_" + str(i) + ".npy")
        image_data = np.delete(image_data, 0, 0)
        aa = np.concatenate((aa, image_data))

    aa = np.delete(aa, 0, 0)

    return aa

# =================== get imbalance race testing and training data ====================== #

def get_race(image_data, race_list, age_list):
    races = {0: "white", 1: "black", 2: "asian", 3: "indian", 4: "other"}
    race = ["white", "black", "asian", "indian", "other"]

    race_cat = {"white": {"image": [], "label": []}, "black": {"image": [], "label": []}, "asian": {"image": [], "label": []}, "indian": {"image": [], "label": []}, "other": {"image": [], "label": []}}
    x_train, y_train = [], []

    for i in range(len(race_list)):
        race_cat[races[race_list[i]]]["image"].append(image_data[i])
        race_cat[races[race_list[i]]]["label"].append(age_list[i])

        x_train.append(image_data[i])
        y_train.append(age_list[i])

    for i in race_cat.keys():
        race_cat[i]["image"] = np.array(race_cat[i]["image"])
        race_cat[i]["label"] = np.array(race_cat[i]["label"])

    x_train, y_train = np.array(x_train), np.array(y_train)
    print()

    return x_train, y_train, race_cat


# =================== get balance race testing and training data ====================== #

def get_balanced_race(image_data, race_list):
    races = {0: "white", 1: "black", 2: "asian", 3: "indian", 4: "other"}

    race_cat = {"white": {"image": [], "label": []}, "black": {"image": [], "label": []}, "asian": {"image": [], "label": []}, "indian": {"image": [], "label": []}, "other": {"image": [], "label": []}}
    balanced_race = {"white": {"image": [], "label": []}, "black": {"image": [], "label": []}, "asian": {"image": [], "label": []}, "indian": {"image": [], "label": []}, "other": {"image": [], "label": []}}

    x_train, y_train = [],[]
    min_ = 6000

    for i in range(len(race_list)):
        race_cat[races[race_list[i]]]["image"].append(image_data[i])
        race_cat[races[race_list[i]]]["label"].append(race_list[i])

    for i in race_cat.keys():
        min_ = min(min_, len(race_cat[i]["image"]))

    for i in range(len(race_list)):
        if len(balanced_race[races[race_list[i]]]["image"]) < min_:
            balanced_race[races[race_list[i]]]["image"].append(image_data[i])
            balanced_race[races[race_list[i]]]["label"].append(race_list[i])

            x_train.append(image_data[i])
            y_train.append(race_list[i])

    for i in balanced_race.keys():
        balanced_race[i]["image"] = np.array(balanced_race[i]["image"])
        balanced_race[i]["label"] = np.array(balanced_race[i]["label"])

    x_train, y_train = np.array(x_train), np.array(y_train)
    print(x_train.shape, y_train.shape)

    return x_train, y_train, balanced_race


def correspondingShuffle(x, y):
    indices = tf.range(start=0, limit=tf.shape(x)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)

    shuffled_x = tf.gather(x, shuffled_indices)
    shuffled_y = tf.gather(y, shuffled_indices)

    return shuffled_x, shuffled_y


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
    plotCurves(x = np.arange(0, hParams["numEpochs"]), 
            yList=[readExperimentalResults(name)[1]['val_accuracy'] for name in fileNames], 
            xLabel="Epoch",
            yLabelList=fileNames,
            title= "val_" + title)

    plotCurves(x = np.arange(0, hParams["numEpochs"]), 
            yList=[readExperimentalResults(name)[1]['accuracy'] for name in fileNames], 
            xLabel="Epoch",
            yLabelList=fileNames,
            title= "acc_" + title)

    # == plot points with xList being the parameter counts of all and yList being the test accuracies == #
    plotPoints(xList=[readExperimentalResults(name)[0]['paramCount'] for name in fileNames],
                yList=[readExperimentalResults(name)[2][0] for name in fileNames],
                pointLabels= [name.split("_")[-1] for name in fileNames],
                xLabel='Number of parameters',
                yLabel='Test set loss',
                title="Test set loss_" + title,
                filename="Test set loss_" + title)

def buildLossPlot(plotNames, title):
    # plotCurves(x=np.arange(0, readExperimentalResults(plotNames[0])[0]['numEpochs']),
    #        yList=[readExperimentalResults(x)[1]['val_loss'] for x in plotNames],
    #        xLabel="Epoch",
    #        yLabelList=plotNames,
    #        title="ValLoss")

    plotCurves(x=np.arange(0, readExperimentalResults(plotNames[0])[0]['numEpochs']),
           yList=[readExperimentalResults(x)[1]['loss'] for x in plotNames],
           xLabel="Epoch",
           yLabelList=plotNames,
           title="Loss")
           
def buildDifPlot(plotNames):
    dif = []
    l1, l2 = [readExperimentalResults(x)[1]['val_loss'] for x in plotNames], [readExperimentalResults(x)[1]['loss'] for x in plotNames]
    print(l1)
    for i in range(len(l1)):
        dif.append([])
        for j in range(len(l1[i])):
            dif[i].append(l1[i][j] - l2[i][j])
        
    plotCurves(x=np.arange(0, readExperimentalResults(plotNames[0])[0]['numEpochs']),
           yList= dif,
           xLabel="Epoch",
           yLabelList=plotNames,
           title= "DifLoss")


def getHParams(expName=None):
    # Set up what's the same for each experiment
    hParams = {
        'experimentName': expName,
        'datasetProportion': 1.0,
        'valProportion': 0.1,
        'numEpochs': 20
    }
    shortTest = False # hardcode to True to run a quick debugging test
    if shortTest:
        print("+++++++++++++++++ WARNING: SHORT TEST +++++++++++++++++")
        hParams['datasetProportion'] = 0.001
        hParams['numEpochs'] = 20

    if (expName is None):
        # Not running an experiment yet, so just return the "common" parameters
        return hParams

    if (expName == 'C32_64__d0.2__D128_10__adam'):
        dropProp = 0.2
        hParams['convLayers'] = [
        {
            'conv_numFilters': 32, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
        {
            'conv_numFilters': 64, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
    ]
        hParams['denseLayers'] = [128, 10]
        hParams['optimizer'] = 'adam'

    return hParams

def cnn(dataSubsets, hParams):
    x_train, y_train, x_test, y_test, x_val, y_val = dataSubsets

    print(x_train.shape)

    x_train = tf.reshape(x_train, (-1, 200, 200, 3))
    x_val = tf.reshape(x_val, (-1, 200, 200, 3))
    x_test = tf.reshape(x_test, (-1, 200, 200, 3))

    # == Shuffle data == #
    x_train, y_train = correspondingShuffle(x_train, y_train)
    x_test, y_test = correspondingShuffle(x_test, y_test)

    # == Sequential Constructor == #
    startTime = timeit.default_timer()
    model = tf.keras.Sequential()

    for layer in hParams['convLayers']:
        model.add(tf.keras.layers.Conv2D(layer['conv_numFilters'], layer['conv_f'], activation=layer['conv_act'], input_shape=(200, 200, 3), padding=layer['conv_p']))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(layer['pool_f'], layer['pool_s'])))
        model.add(tf.keras.layers.Dropout(layer['drop_prop']))

    model.add(tf.keras.layers.Flatten())
    for layer in range(len(hParams['denseLayers'])):
        model.add(tf.keras.layers.Dense(hParams['denseLayers'][layer], activation='relu'))

    model.add(tf.keras.layers.Dense(5))

    # == Loss function == #
    lossFunc = tf.keras.losses.SparseCategoricalCrossentropy(True)

    # == fitting == #
    model.compile(loss=lossFunc, metrics=['accuracy'], optimizer=hParams['optimizer'])
    hist = model.fit(x_train, y_train, 
                    validation_data=(x_val, y_val) 
                                                if hParams['valProportion']!=0.0 
                                                else None, 
                    batch_size=15,
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
        "denseLayers": [64, 32, 10],
        "valProportion": 0.1,
        "experimentName": "128_10e20",
        "optimizer": "adam"
    }

    image_source = "UTKFace"
    image_links = os.listdir(image_source)
    image_data = get_data(2)
    model =  'C32_64__d0.2__D128_10__adam'

    race_list = np.array([int(x.split("_")[2]) for x in image_links[:6000] if len(x) > 3])

    print(image_data.shape, race_list.shape)


    # ================== Run the imbalance training dataset ===================== #

    x_train, y_train, race_cat = get_race(image_data, race_list)
    x_train = x_train.astype('float32')
    x_train = x_train / 255
    print([len(race_cat[x]["image"]) for x in race_cat.keys()])

    # == Slice for validation data == #
    x_val = x_train[:int(hParams['valProportion'] * x_train.shape[0])]
    y_val = y_train[:int(hParams['valProportion'] * y_train.shape[0])]

    x_train = x_train[int(hParams['valProportion'] * x_train.shape[0]):]
    y_train = y_train[int(hParams['valProportion'] * y_train.shape[0]):]

    hParams = getHParams(model)

    for race in race_cat.keys():
        print(race)
        x_test, y_test = race_cat[race]["image"], race_cat[race]["label"]

        x_test = x_test.astype('float32')
        x_test = x_test / 255 

        dataSubsets = x_train, y_train, x_test, y_test, x_val, y_val

        trainResults, testResults = cnn(dataSubsets, hParams)
        hParams["experimentName"] = hParams["experimentName"] + "_" + race
        writeExperimentalResults(hParams, trainResults, testResults)





    # ================== Run the balance training dataset ===================== #

    x_train, y_train, race_cat = get_balanced_race(image_data, race_list)
    x_train = x_train.astype('float32')
    x_train = x_train / 255
    print([len(race_cat[x]["image"]) for x in race_cat.keys()])

    # == Slice for validation data == #
    x_val = x_train[:int(hParams['valProportion'] * x_train.shape[0])]
    y_val = y_train[:int(hParams['valProportion'] * y_train.shape[0])]

    x_train = x_train[int(hParams['valProportion'] * x_train.shape[0]):]
    y_train = y_train[int(hParams['valProportion'] * y_train.shape[0]):]

    hParams = getHParams(model)


    for race in list(race_cat.keys())[2:]:
        print(race)
        x_test, y_test = race_cat[race]["image"], race_cat[race]["label"]

        x_test = x_test.astype('float32')
        x_test = x_test / 255 

        dataSubsets = x_train, y_train, x_test, y_test, x_val, y_val

        trainResults, testResults = cnn(dataSubsets, hParams)
        hParams["experimentName"] = hParams["experimentName"]  + "_balanced_" +  race
        writeExperimentalResults(hParams, trainResults, testResults)


# main()
fileNames = [
            [
            'C32_64__d0.2__D128_10__adam_white', 
            'C32_64__d0.2__D128_10__adam_black', 
            'C32_64__d0.2__D128_10__adam_indian', 
            'C32_64__d0.2__D128_10__adam_asian', 
            'C32_64__d0.2__D128_10__adam_other'
            ],
            # [
            # 'C32_64__d0.2__D128_10__adam_balanced_white', 
            # 'C32_64__d0.2__D128_10__adam_balanced_black', 
            # 'C32_64__d0.2__D128_10__adam_balanced_indian', 
            # 'C32_64__d0.2__D128_10__adam_balanced_asian', 
            # 'C32_64__d0.2__D128_10__adam_balanced_other'
            # ]
        ]

for fileName in fileNames:
    buildValAccuracyPlot(fileName, "race")
    # buildValAccuracyPlot(fileName, "balanced_race")
