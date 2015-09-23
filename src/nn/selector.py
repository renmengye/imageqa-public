from stage import *

class Selector(Stage):
    def __init__(self, 
                 name, 
                 inputNames,
                 start, 
                 end, 
                 axis=-1):
        Stage.__init__(
                 self,
                 name=name, 
                 inputNames=inputNames,
                 outputDim=end-start)
        self.start = start
        self.end = end
        self.axis = axis
        if axis < -2 or axis > 2:
            raise Exception('Selector axis=%d not supported' % axis)

    def forward(self, X):
        self.X = X
        if self.axis == -1:
            self.axis = len(X.shape) - 1
        if self.axis == 0:
            return X[self.start:self.end]
        elif self.axis == 1:
            return X[:, self.start:self.end]
        elif self.axis == 2:
            return X[:, :, self.start:self.end]

    def backward(self, dEdY):
        dEdX = np.zeros(self.X.shape)
        if self.axis == 0:
            dEdX[self.start:self.end] = dEdY
        elif self.axis == 1:
            dEdX[:, self.start:self.end] = dEdY
        elif self.axis == 2:
            dEdX[:, :, self.start:self.end] = dEdY
        return dEdX