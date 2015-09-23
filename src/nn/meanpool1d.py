from stage import *

class MeanPool1D(Stage):
    """
    1D mean pooling. 
    Padding no longer make sense now. 
    Make sure you have the right size.
    """
    def __init__(self,
                 outputDim,
                 windowSize,
                 inputNames=None,
                 defaultValue=0.0,
                 outputdEdX=True,
                 name=None):
        Stage.__init__(self,
                 name=name,
                 inputNames=inputNames,
                 outputDim=outputDim,
                 defaultValue=defaultValue,
                 outputdEdX=outputdEdX)
        self.windowSize = windowSize
        self.X = 0
        self.Y = 0

    def forward(self, X):
        X = X.reshape(X.shape[0], self.windowSize, X.shape[1] / self.windowSize, X.shape[2])
        Y = np.mean(X, axis=1)
        self.X = X
        return Y

    def backward(self, dEdY):
        dEdX = np.tile(
            dEdY.reshape(dEdY.shape[0], 1, dEdY.shape[1], dEdY.shape[2]), 
            (1, self.windowSize, 1, 1))
        dEdX /= float(self.windowSize)
        dEdX = dEdX.reshape(dEdX.shape[0], dEdX.shape[1] * dEdX.shape[2], dEdX.shape[3])
        return dEdX