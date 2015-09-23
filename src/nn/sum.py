from stage import *

class Sum(Stage):
    """Stage summing first half of the input with second half."""
    def __init__(self, name, inputNames, numComponents, outputDim,
                 defaultValue=0.0):
        Stage.__init__(
            self,
            name=name,
            inputNames=inputNames,
            outputDim=outputDim,
            defaultValue=defaultValue)
        self.numComponents = numComponents
    def forward(self, X):
        return np.sum(
            X.reshape(X.shape[0],
                self.numComponents,
                X.shape[1] / self.numComponents),
            axis=1)
    def backward(self, dEdY):
        self.dEdW = 0.0
        return np.tile(dEdY, self.numComponents)
