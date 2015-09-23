import numpy as np

def splitData(trainInput, trainTarget, heldOutRatio, validNumber):
    s = np.round(trainInput.shape[0] * heldOutRatio)
    start = s * validNumber
    validInput = trainInput[start : start + s]
    validTarget = trainTarget[start : start + s]
    if validNumber == 0:
        trainInput = trainInput[s:]
        trainTarget = trainTarget[s:]
    else:
        trainInput = np.concatenate((trainInput[0:start], trainInput[start + s:]))
        trainTarget = np.concatenate((trainTarget[0:start], trainTarget[start + s:]))
    return trainInput, trainTarget, validInput, validTarget

def shuffleData(X, T, random=None):
    if random is None:
        random = np.random.RandomState()
    shuffle = np.arange(0, X.shape[0])
    shuffle = random.permutation(shuffle)
    X = X[shuffle]
    T = T[shuffle]
    return X, T