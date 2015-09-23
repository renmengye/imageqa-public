import numpy as np

def meanSqErr(Y, T, weights=None):
    diff =  Y - T.reshape(Y.shape)
    diff2 = np.sum(np.power(diff, 2), axis=-1)
    if weights is not None:
        diff2 *= weights
        weights = weights.reshape(weights.shape[0], 1)
        diff *= weights
    E = 0.5 * np.sum(diff2) / float(Y.shape[0])
    dEdY = diff / float(Y.shape[0])
    return E, dEdY

def hardLimit(Y):
    return (Y > 0.5).astype(int)

def sigmoidFn(X):
    return 1 / (1 + np.exp(-X))

def crossEntIdx(Y, T, weights=None):
    eps = 1e-8
    Y2 = Y.reshape(Y.size / Y.shape[-1], Y.shape[-1])
    T2 = T.reshape(T.size)
    E = 0.0
    dEdY = np.zeros(Y2.shape, float)
    if weights is None:
        for n in range(0, Y2.shape[0]):
            E += -np.log(Y2[n, T2[n]] + eps)
            dEdY[n, T2[n]] = -1 / (Y2[n, T2[n]] + eps)
    else:
        for n in range(0, Y2.shape[0]):
            E += -np.log(Y2[n, T2[n]] + eps) * weights[n]
            dEdY[n, T2[n]] = -1 / (Y2[n, T2[n]] + eps) * weights[n]
    E /= Y2.shape[0]
    dEdY /= Y2.shape[0]
    dEdY = dEdY.reshape(Y.shape)
    return E, dEdY

def crossEntOneIdx(Y, T, weights=None):
    eps = 1e-8
    Y2 = Y.reshape(Y.size / Y.shape[-1], Y.shape[-1])
    T2 = T.reshape(T.size)
    E = 0.0
    dEdY = np.zeros(Y2.shape, float)
    if weights is None:
        for n in range(0, Y.shape[0]):
            E += -np.log(Y2[n, T2[n]] + eps) + np.log(1 - Y2[n, T2[n] + eps])
            E += -np.sum(np.log(1 - Y2[n, :] + eps))
            dEdY[n, :] = 1 / (1 - Y2[n] + eps)
            dEdY[n, T2[n]] = -1 / (Y2[n, T2[n]] + eps)
    else:
        for n in range(0, Y.shape[0]):
            E += (-np.log(Y2[n, T2[n]] + eps) + \
                np.log(1 - Y2[n, T2[n] + eps])) * weights[n]
            E += (-np.sum(np.log(1 - Y2[n, :] + eps))) * weights[n]
            dEdY[n, :] = (1 / (1 - Y2[n] + eps)) * weights[n]
            dEdY[n, T2[n]] = (-1 / (Y2[n, T2[n]] + eps)) * weights[n]
    E /= Y2.shape[0]
    dEdY /= Y2.shape[0]
    dEdY = dEdY.reshape(Y.shape)
    return E, dEdY

def crossEntOneAccIdx(Y, T, weights=None):
    eps = 1e-8
    Y2 = Y.reshape(Y.size / Y.shape[-1], Y.shape[-1])
    T2 = T.reshape(T.size)
    E = 0.0
    dEdY = np.zeros(Y2.shape, float)
    if weights is None:
        for n in range(0, Y.shape[0]):
            t = T2[n]
            E += -np.sum(np.log(Y2[n, t + 1:] + eps))
            E += -np.sum(np.log(1 - Y2[n, :t + 1] + eps))
            dEdY[n, t + 1:] = -1 / (Y2[n, t + 1:] + eps)
            dEdY[n, :t + 1] = 1/ (1 - Y2[n, :t + 1] + eps)
    else:
        for n in range(0, Y.shape[0]):
            t = T2[n]
            E += -np.sum(np.log(Y2[n, t + 1:] + eps)) * weights[n]
            E += -np.sum(np.log(1 - Y2[n, :t + 1] + eps)) * weights[n]
            dEdY[n, t + 1:] = -1 / (Y2[n, t + 1:] + eps) * weights[n]
            dEdY[n, :t + 1] = 1/ (1 - Y2[n, :t + 1] + eps) * weights[n]
    E /= Y2.shape[0]
    dEdY /= Y2.shape[0]
    dEdY = dEdY.reshape(Y.shape)
    return E, dEdY

def crossEntOne(Y, T, weights=None):
    eps = 1e-8
    T = T.reshape(Y.shape)
    cost = -T * np.log(Y + eps) - (1 - T) * np.log(1 - Y + eps)
    dcost = -T / (Y + eps) + (1 - T) / (1 - Y + eps)
    if weights is not None:
        cost *= weights
        dcost *= weights.reshape(weights.shape[0], 1)
    if len(Y.shape) == 0:
        E = cost
        dEdY = dcost
    else:
        E = np.sum(cost) / float(Y.size)
        dEdY = dcost / float(Y.size)
    return E, dEdY

def argmax(Y):
    return np.argmax(Y, axis=-1)

def argmaxDiff(Y):
    Y2 = Y.reshape(Y.size / Y.shape[-1], Y.shape[-1])
    Ydiff = np.zeros(Y2.shape)
    for i in range(Y2.shape[1] - 1):
        Ydiff[:, i] = Y2[:, i + 1] - Y2[:, i]
    Ydiff2 = np.reshape(Ydiff, Y.shape)
    return np.argmax(Ydiff2, axis=-1)

def meanSqErrEye(Y, T, weights=None):
    eye = np.eye(Y.shape[-1])
    T2 = T.reshape(T.size)
    T3 = eye[T2]
    return meanSqErr(Y, T3, weights=weights)

def roundInt(Y):
    return np.round(Y).astype('int')

def rankingLoss(Y, T, weights=None):
    alpha = 0.1
    dEdY = np.zeros(Y.shape)
    E = 0.0
    for n in range(T.size):
        cost = Y[n] - Y[n, T[n]] + alpha
        valid = (cost > 0).astype(int)
        nvalid = np.sum(valid) - 1
        cost = cost * valid
        dEdY[n] = valid
        dEdY[n, T[n]] = -nvalid
        if weights is not None:
            cost *= weights[n]
            dEdY[n] *= weights[n]
        E += np.sum(cost) - alpha
    E /= float(T.size)
    dEdY /= float(T.size)
    return E, dEdY
