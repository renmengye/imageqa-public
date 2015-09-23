from container import *

class Sequential(Stage):
    def __init__(self, stages, inputNames=None, name=None, outputDim=0, outputdEdX=True):
        Stage.__init__(self,
                 name=name,
                 outputDim=outputDim,
                 inputNames=inputNames,
                 outputdEdX=outputdEdX)
        self.stages = stages

    def forward(self, X, dropout=True):
        X1 = X
        for stage in self.stages:
            if isinstance(stage, Container) or isinstance(stage, Sequential):
                X1 = stage.forward(X1, dropout)
            elif hasattr(stage, 'dropout'):
                stage.dropout = dropout
                X1 = stage.forward(X1)
            else:
                X1 = stage.forward(X1)
        return X1

    def backward(self, dEdY):
        for stage in reversed(self.stages):
            dEdY = stage.backward(dEdY)
            if dEdY is None: break
        return dEdY if self.outputdEdX else None

    def updateWeights(self):
        for stage in self.stages:
            stage.updateWeights()
        return

    def updateLearningParams(self, numEpoch):
        for stage in self.stages:
            stage.updateLearningParams(numEpoch)
        return

    def getWeights(self):
        weights = []
        for stage in self.stages:
            weights.append(stage.getWeights())
        return np.array(weights, dtype=object)

    def loadWeights(self, W):
        for i in range(W.shape[0]):
            self.stages[i].loadWeights(W[i])