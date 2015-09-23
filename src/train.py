import nn
import numpy as np
import os
import sys
import yaml
import imageqa_test

"""
Train a neural network
Usage: python train.py
                    -n[ame] {name} 
                    -d[ata] {train/valid/test folder}
                    -m[odel] {model spec} 
                    -s[aved] {saved model id}
                    -e[arly] {early stop score}
                    -c[onfig] {config filename}
                    -w[eights] {input weights}
                    -o[utput] {output folder}
                    -b[oard] {board id}
                    [-imageqa]
Prameters: 
    -n[ame] Name of the model 
    -d[ata] Data folder that contains 'train.npy', 'valid.npy', and 'test.npy'
    -m[odel] Model specification file name
    -s[aved] Saved model ID
    -e[arly] Early stop score
    -c[onfig] Train config file name
    -w[eights] Weighted input (for boosting), filename before '-train' or '-test'
    -o[utput] Training results output folder
    -b[oard] GPU board ID
    [-imageqa] Run Image-QA test scripts
"""

def readFlags():
    params = {}
    params['name'] = None
    params['outputFolder'] = None
    params['configFilename'] = None
    params['trainDataFilename'] = None
    params['testDataFilename'] = None
    params['validDataFilename'] = None
    params['allDataFilename'] = None
    params['modelFilename'] = None
    params['savedModelId'] = None
    params['earlyStopScore'] = None
    params['imageqa'] = True
    params['trainInputWeightsFilename'] = None
    params['validInputWeightsFilename'] = None
    for i, flag in enumerate(sys.argv):
        if flag == '-n' or flag == '-name':
            params['name'] = sys.argv[i + 1]
        elif flag == '-o' or flag == '-out':
            params['outputFolder'] = sys.argv[i + 1]
        elif flag == '-d' or flag == '-data':
            dataFolder = sys.argv[i + 1]
            trainPath = os.path.join(dataFolder, 'train.npy')
            params['trainDataFilename'] = trainPath if os.path.isfile(trainPath) else None
            validPath = os.path.join(dataFolder, 'valid.npy')
            params['validDataFilename'] = validPath if os.path.isfile(validPath) else None
            testPath = os.path.join(dataFolder, 'test.npy')
            params['testDataFilename'] = testPath if os.path.isfile(testPath) else None
            params['dataFolder'] = dataFolder
        elif flag == '-w' or flag == '-weights':
            weightsPath = sys.argv[i + 1]
            params['trainInputWeightsFilename'] = os.path.join(dataFolder, weightsPath + '-train.npy')
            params['validInputWeightsFilename'] = os.path.join(dataFolder, weightsPath + '-valid.npy')
            params['testInputWeightsFilename'] = os.path.join(dataFolder, weightsPath + '-test.npy')
        elif flag == '-m' or flag == '-model':
            params['modelFilename'] = sys.argv[i + 1]
        elif flag == '-s' or flag == '-saved':
            params['savedModelId'] = sys.argv[i + 1]
        elif flag == '-e' or flag == '-early':
            params['earlyStopScore'] = float(sys.argv[i + 1])
        elif flag == '-c' or flag == '-config':
            params['configFilename'] = sys.argv[i + 1]
        elif flag == '-b' or flag == '-board':
            os.environ['GNUMPY_BOARD_ID'] = sys.argv[i + 1]
        elif flag == '-imageqa':
            params['imageqa'] = True
        elif flag == '-noimageqa':
            params['imageqa'] = False

    # Check required parameters.
    if params['configFilename'] is None:
        raise Exception('Config file not specified')
    if params['trainDataFilename'] is None:
        raise Exception('Data file not specified')
    if params['modelFilename'] is None:
        raise Exception('Model file not specified')
    if params['name'] is None:
        params['name'] = params['modelFilename'].split('/')[-1].split('.')[0]

    return params

def runTests(params, model, trainer):
    if params['testDataFilename'] is not None:
        if params['imageqa']:
            imageqa_test.testAll(
                trainer.name, model, params['dataFolder'], params['outputFolder'])
        else:
            testData = np.load(params['testDataFilename'])
            testInput = testData[0]
            testTarget = testData[1]
            model.loadWeights(np.load(trainer.modelFilename))
            testOutput = nn.test(model, testInput)
            testRate, c, t = nn.calcRate(model, testOutput, testTarget)
            print 'Test rate: ', testRate
            with open(os.path.join(
                trainer.outputFolder, 'result.txt'), 'w+') as f:
                f.write('Test rate: %f\n' % testRate)

def combineInputs(
                    trainInput, 
                    trainTarget, 
                    trainInputWeights, 
                    validInput, 
                    validTarget, 
                    validInputWeights):
    allInput = np.concatenate((trainInput, validInput), axis=0)
    allTarget = np.concatenate((trainTarget, validTarget), axis=0)
    if trainInputWeights is not None:
        allInputWeights = np.concatenate(
            (trainInputWeights, validInputWeights), axis=0)
    else:
        allInputWeights = None
    return allInput, allTarget, allInputWeights

def trainValid(
        params,
        trainOpt,
        trainInput, 
        trainTarget, 
        trainInputWeights, 
        validInput, 
        validTarget,
        validInputWeights,
        initWeights=None):
    model = nn.load(params['modelFilename'])
    if initWeights is not None:
        model.loadWeights(initWeights)
    trainer = nn.Trainer(
        name=params['name']+\
        ('-v' if params['validDataFilename'] is not None else ''),
        model=model,
        trainOpt=trainOpt,
        outputFolder=params['outputFolder']
    )

    # Validation training
    trainer.train(
                trainInput=trainInput, 
                trainTarget=trainTarget,
                trainInputWeights=trainInputWeights,
                validInput=validInput, 
                validTarget=validTarget,
                validInputWeights=validInputWeights)
    return model, trainer

def trainAll(
        params,
        trainOpt,
        trainInput, 
        trainTarget, 
        trainInputWeights, 
        validInput, 
        validTarget, 
        validInputWeights,
        initWeights=None):            
    model = nn.load(params['modelFilename'])
    if initWeights is not None:
        model.loadWeights(initWeights)
    trainer = nn.Trainer(
        name=params['name'],
        model=model,
        trainOpt=trainOpt,
        outputFolder=params['outputFolder']
    )
    # Combine train & valid set
    allInput, allTarget, allInputWeights = combineInputs(
            trainInput, 
            trainTarget, 
            trainInputWeights, 
            validInput, 
            validTarget, 
            validInputWeights)

    trainer.train(
            trainInput=allInput, 
            trainTarget=allTarget,
            trainInputWeights=allInputWeights)
    return model, trainer

if __name__ == '__main__':
    # Read params
    params = readFlags()
    
    # Load train options
    with open(params['configFilename']) as f:
        trainOpt = yaml.load(f)
    
    # Load dataset
    trainData = np.load(params['trainDataFilename'])
    trainInput = trainData[0]
    trainTarget = trainData[1]
    
    if params['validDataFilename'] is not None:
        validData = np.load(params['validDataFilename'])
        validInput = validData[0]
        validTarget = validData[1]
    else:
        validInput = None
        validTarget = None

    if params['trainInputWeightsFilename'] is not None:
        trainInputWeights = np.load(params['trainInputWeightsFilename'])
        validInputWeights = np.load(params['validInputWeightsFilename'])
    else:
        trainInputWeights = None
        validInputWeights = None  

    if params['savedModelId'] is not None:
        modelFolder = os.path.join(params['outputFolder'], params['savedModelId'])
        initWeights = np.load(os.path.join(modelFolder, params['savedModelId'] + '.w.npy'))
        if '-v-' in params['savedModelId']:
            # Train model
            model, trainer = trainValid(
                params,
                trainOpt,
                trainInput, 
                trainTarget, 
                trainInputWeights, 
                validInput, 
                validTarget, 
                validInputWeights,
                initWeights=initWeights)

            # Reload model
            model = nn.load(params['modelFilename'])
            model.loadWeights(np.load(trainer.modelFilename))

            # Run tests
            runTests(params, model, trainer)
            
            # Re-train
            if params['testDataFilename'] is not None and \
                params['validDataFilename'] is not None:

                # Setup options
                trainOpt['needValid'] = False
                print 'Stopped score:', trainer.stoppedTrainScore
                trainOpt['stopScore'] = trainer.stoppedTrainScore

                # Train train+valid
                model, trainer = trainAll(
                    params,
                    trainOpt,
                    trainInput, 
                    trainTarget, 
                    trainInputWeights, 
                    validInput, 
                    validTarget, 
                    validInputWeights)

                # Reload model
                model = nn.load(params['modelFilename'])
                model.loadWeights(np.load(trainer.modelFilename))

                # Run tests
                runTests(params, model, trainer)
        else:
            # Set up options
            trainOpt['needValid'] = False
            if params['earlyStopScore'] is not None:
                trainOpt['stopScore'] = params['earlyStopScore']
            else:
                raise Exception('Need to provide early stop score.')

            # Train train+valid
            model, trainer = trainAll(
                params,
                trainOpt,
                trainInput, 
                trainTarget, 
                trainInputWeights, 
                validInput, 
                validTarget, 
                validInputWeights,
                initWeights=initWeights)

            # Reload model
            model = nn.load(params['modelFilename'])
            model.loadWeights(np.load(trainer.modelFilename))

            # Run tests
            runTests(params, model, trainer)
    else:
        if params['earlyStopScore'] is None:
            # Train model
            model, trainer = trainValid(
                params,
                trainOpt,
                trainInput, 
                trainTarget, 
                trainInputWeights, 
                validInput, 
                validTarget, 
                validInputWeights)

            # Reload model
            model = nn.load(params['modelFilename'])
            model.loadWeights(np.load(trainer.modelFilename))

            # Run tests
            runTests(params, model, trainer)
            
            # Re-train
            if params['testDataFilename'] is not None and \
                params['validDataFilename'] is not None:

                # Setup options
                trainOpt['needValid'] = False
                print 'Stopped score:', trainer.stoppedTrainScore
                trainOpt['stopScore'] = trainer.stoppedTrainScore

                # Train train+valid
                model, trainer = trainAll(
                    params,
                    trainOpt,
                    trainInput, 
                    trainTarget, 
                    trainInputWeights, 
                    validInput, 
                    validTarget, 
                    validInputWeights)

                # Reload model
                model = nn.load(params['modelFilename'])
                model.loadWeights(np.load(trainer.modelFilename))

                # Run tests
                runTests(params, model, trainer)
        else:
            # Set up options
            trainOpt['needValid'] = False
            trainOpt['stopScore'] = params['earlyStopScore']

            # Train train+valid
            model, trainer = trainAll(
                params,
                trainOpt,
                trainInput, 
                trainTarget, 
                trainInputWeights, 
                validInput, 
                validTarget, 
                validInputWeights)

            # Reload model
            model = nn.load(params['modelFilename'])
            model.loadWeights(np.load(trainer.modelFilename))

            # Run tests
            runTests(params, model, trainer)

