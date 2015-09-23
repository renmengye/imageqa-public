from stage import *

class Active(Stage):
    def __init__(self,
                 activeFn,
                 inputNames,
                 outputDim,
                 defaultValue=0.0,
                 outputdEdX=True,
                 name=None):
        Stage.__init__(self,
                 name=name,
                 inputNames=inputNames,
                 outputDim=outputDim,
                 defaultValue=defaultValue,
                 outputdEdX=outputdEdX)
        self.activeFn = activeFn
    def forward(self, X):
        self.Y = self.activeFn.forward(X)
        return self.Y
    def backward(self, dEdY):
        self.dEdW = 0
        return self.activeFn.backward(dEdY, self.Y, 0)