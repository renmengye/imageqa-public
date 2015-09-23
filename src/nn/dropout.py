from stage import *

class Dropout(Stage):
    def __init__(self,
                 name,
                 inputNames,
                 outputDim,
                 dropoutRate,
                 initSeed,
                 debug=False):
        Stage.__init__(self,
            name=name,
            inputNames=inputNames,
            outputDim=outputDim)
        self.dropout = True
        self.dropoutVec = 0
        self.dropoutRate = dropoutRate
        self.debug = debug
        self.random = np.random.RandomState(initSeed)
        self.seed = initSeed

    def forward(self, X):
        if self.dropoutRate > 0.0 and self.dropout:
            if self.debug:
                self.random = np.random.RandomState(self.seed)
            self.dropoutVec = (self.random.uniform(0, 1, (X.shape[-1])) >
                               self.dropoutRate)
            Y = X * self.dropoutVec
        else:
            Y = X * (1 - self.dropoutRate)
        self.X = X
        return Y

    def backward(self, dEdY):
        dEdX = None
        if self.outputdEdX:
            if self.dropout:
                dEdX = dEdY * self.dropoutVec
            else:
                dEdX = dEdY / (1 - self.dropoutRate)
        return dEdX