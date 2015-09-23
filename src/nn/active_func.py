from func import *

class SoftmaxActiveFn():
    def __init__(self):
        pass

    @staticmethod
    def forward(Z):
        expY = np.exp(Z)
        expYshape = np.copy(Z.shape)
        expYshape[-1] = 1
        Y = expY / np.sum(expY, axis=-1).reshape(expYshape).repeat(Z.shape[-1], axis=-1)
        return Y

    @staticmethod
    def backward(dEdY, Y, Z):
        timespan = Y.shape[0]
        U = dEdY * Y
        dEdZ = U - np.sum(U, axis=-1).reshape(timespan, 1) * Y
        return dEdZ

class SigmoidActiveFn():
    def __init__(self):
        pass

    @staticmethod
    def forward(Z):
        Y = sigmoidFn(Z)
        return Y

    @staticmethod
    def backward(dEdY, Y, Z):
        dEdZ = dEdY * Y * (1 - Y)
        return dEdZ

class TanhActiveFn():
    def __init__(self):
        pass

    @staticmethod
    def forward(Z):
        Y = np.tanh(Z)
        return Y

    @staticmethod
    def backward(dEdY, Y, Z):
        dEdZ = dEdY * (1 - Y * Y)
        return dEdZ

class IdentityActiveFn():
    def __init__(self):
        pass

    @staticmethod
    def forward(Z):
        return Z

    @staticmethod
    def backward(dEdY, Y, Z):
        return dEdY

class ReluActiveFn():
    def __init__(self):
        pass

    @staticmethod
    def forward(Z):
        return np.maximum(0, Z)

    @staticmethod
    def backward(dEdY, Y, Z):
        return (Y > 0).astype(int) * dEdY
