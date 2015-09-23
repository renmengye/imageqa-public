from stage import *

class ConstValue(Stage):
    def __init__(self,
                 name,
                 inputNames,
                 outputDim,
                 value):
        Stage.__init__(self,
                 name=name,
                 outputDim=outputDim,
                 inputNames=inputNames,
                 outputdEdX=False)
        self.dEdW = 0
        self.value = value
        
    def graphBackward(self):
        self.backward(self.dEdY)

    def forward(self, X):
        return np.zeros((X.shape[0], self.outputDim)) + self.value

    def backward(self, dEdY):
        return None