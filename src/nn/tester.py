import numpy as np

def test(model, X, numExPerBat=100, layerNames=None):
    N = X.shape[0]
    batchStart = 0
    Y = None
    layers = {}
    if layerNames is not None:
        for layerName in layerNames:
            layers[layerName] = []
    while batchStart < N:
        # Batch info
        batchEnd = min(N, batchStart + numExPerBat)
        Ytmp = model.forward(X[batchStart:batchEnd], dropout=False)
        if Y is None:
            Yshape = np.copy(Ytmp.shape)
            Yshape[0] = N
            Y = np.zeros(Yshape)
        if layerNames is not None:
            for layerName in layerNames:
                stage = model
                for stageName in layerName.split(':'):
                    stage = stage.stageDict[stageName]
                layers[layerName].append(stage.getValue())
        Y[batchStart:batchEnd] = Ytmp
        batchStart += numExPerBat
    if layerNames is not None:
        for layerName in layerNames:
            layers[layerName] = np.concatenate(layers[layerName], axis=0)
        return Y, layers
    else:
        return Y

def calcRate(model, Y, T):
    Yfinal = model.predict(Y)
    correct = np.sum(Yfinal.reshape(Yfinal.size) == T.reshape(T.size))
    total = Yfinal.size
    rate = correct / float(total)
    return rate, correct, total