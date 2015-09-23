from stage import *

class ElementProduct(Stage):
    """Stage multiplying first half of the input with second half"""
    def __init__(self, name, inputNames, outputDim,
                 defaultValue=0.0):
        Stage.__init__(
            self,
            name=name,
            inputNames=inputNames,
            outputDim=outputDim,
            defaultValue=defaultValue)
    def forward(self, X):
        self.X = X
        return X[:,:X.shape[1]/2] * X[:,X.shape[1]/2:]
    def backward(self, dEdY):
        self.dEdW = 0.0
        return np.concatenate(
            (self.X[:,self.X.shape[1]/2:] * dEdY,
            self.X[:,:self.X.shape[1]/2] * dEdY),
            axis=-1)