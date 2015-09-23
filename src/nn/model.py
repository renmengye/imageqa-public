from container import *

class GraphModel(Container):
    def __init__(self,
                 stages, 
                 outputStageNames,
                 costFn,
                 inputDim=0,
                 outputDim=0,
                 name=None,
                 decisionFn=None, 
                 specFilename=None):
        Container.__init__(self,
                 name=name,
                 stages=stages,
                 inputNames=['input'],
                 outputStageNames=outputStageNames,
                 inputDim=inputDim,
                 outputDim=outputDim,
                 outputdEdX=True)
        self.getCost = costFn
        self.predict = decisionFn
        self.specFilename = specFilename