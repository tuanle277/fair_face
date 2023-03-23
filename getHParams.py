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
        hParams['datasetProportion'] = 0.0001
        hParams['numEpochs'] = 2

    if (expName is None):
        # Not running an experiment yet, so just return the "common" parameters
        return hParams

    if (expName == 'C32_64__d0.0__D128_10__rms'):
        dropProp = 0.0
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
        hParams['optimizer'] = 'rmsprop'

    elif (expName == 'C32_64__d0.0__D128_10__adam'):
        dropProp = 0.0
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

    elif (expName == 'C32_64__d0.2__D128_10__rms'):
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
        hParams['optimizer'] = 'rmsprop'

    elif (expName == 'C32_64__d0.2__D128_10__rms_test'):
        dropProp = 0.2
        hParams['convLayers'] = [
        {
            'conv_numFilters': 32, 
            'conv_f': 4, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
        {
            'conv_numFilters': 64, 
            'conv_f': 4, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
    ]
        hParams['denseLayers'] = [128, 10]
        hParams['optimizer'] = 'rmsprop'

    elif (expName == 'C32_64__d0.2__D128_10__adam'):
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

    elif (expName == 'C32_64__d0.02__D128_10__adam'):
        dropProp = 0.02
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

    elif (expName == 'C32_64__d0.05__D128_10__adam'):
        dropProp = 0.05
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

    elif (expName == 'C32_64__d0.1__D128_10__adam'):
        dropProp = 0.1
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

    elif (expName == 'C32_64__d0.3__D128_10__adam'):
        dropProp = 0.3
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

    elif (expName == 'C32_64__d0.2__D256_128_10__adam'):
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
        hParams['denseLayers'] = [256, 128, 10]
        hParams['optimizer'] = 'adam'

    elif (expName == 'C32_64__d0.2__D512_256_128_10__adam'):
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
        hParams['denseLayers'] = [512, 256, 128, 10]
        hParams['optimizer'] = 'adam'

    elif (expName == 'C32__d0.2__D128_10__adam'):
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
    ]
        hParams['denseLayers'] = [128, 10]
        hParams['optimizer'] = 'adam'

    elif (expName == 'C32_64_128__d0.2__D128_10__adam'):
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
        {
            'conv_numFilters': 128, 
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


    elif (expName == 'C32_64_128_256__d0.2__D128_10__adam'):
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
        {
            'conv_numFilters': 128, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
        {
            'conv_numFilters': 256, 
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

    elif (expName == 'C32__d0.2__D128_10__rms'):
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
    ]
        hParams['denseLayers'] = [128, 10]
        hParams['optimizer'] = 'rmsprop'

    elif (expName == 'C32_64_128__d0.2__D128_10__rms'):
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
        {
            'conv_numFilters': 128, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
    ]
        hParams['denseLayers'] = [128, 10]
        hParams['optimizer'] = 'rmsprop'

    elif (expName == 'C32_64_128_256__d0.2__D128_10__rms'):
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
        {
            'conv_numFilters': 128, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
        {
            'conv_numFilters': 256, 
            'conv_f': 3, 
            'conv_p': 'same',
            'conv_act': 'relu', 
            'pool_f': 2, 
            'pool_s': 2,
            'drop_prop': dropProp
        },
    ]
        hParams['denseLayers'] = [128, 10]
        hParams['optimizer'] = 'rmsprop'



    return hParams