from stage import *

class MaxPool1D(Stage):
    """
    1D max pooling.
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
        mod = np.mod(X.shape[1], self.windowSize)
        if mod > 0:
            X = np.concatenate((X, np.zeros((X.shape[0], self.windowSize - mod, X.shape[2]))), axis=1)
        X = X.reshape(X.shape[0], self.windowSize, X.shape[1] / self.windowSize, X.shape[2])
        self.argX = np.argmax(X, axis=1)
        Y = np.max(X, axis=1)
        self.X = X
        self.mod = mod
        return Y

    def backward(self, dEdY):
        """
        Assuming the last dimension is the largest.
        """
        self.dEdW = 0
        dEdX = np.zeros(self.X.shape)
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[2]):
                dEdX[i, self.argX[i, j, :], j, range(0, self.X.shape[3])] = dEdY[i, j, :]
        dEdX = dEdX.reshape(dEdX.shape[0], dEdX.shape[1] * dEdX.shape[2], dEdX.shape[3])
        if self.mod > 0:
            dEdX = dEdX[:, :-(self.windowSize - self.mod), :]
        return dEdX