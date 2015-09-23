from stage import *

class Normalize(Stage):
    def __init__(self,
                 outputDim,
                 mean,
                 std,
                 name=None,
                 inputNames=None,
                 outputdEdX=True):
        Stage.__init__(self,
                 name=name,
                 inputNames=inputNames,
                 outputDim=outputDim,
                 outputdEdX=outputdEdX)
        self.mean = mean
        self.std = std
        self.X = 0
        self.Y = 0
        pass

    def forward(self, X):
        return (X - self.mean) / self.std

    def backward(self, dEdY):
        return dEdY / self.std