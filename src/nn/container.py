from active_func import *
from map import *

class Input(Stage):
    def __init__(self, name, outputDim):
        Stage.__init__(self,
            name=name, 
            inputNames=[],
            outputDim=outputDim)
    def setValue(self, value):
        self.Y = value
    def forward(self, X):
        return X
    def backward(self, dEdY):
        return dEdY

class Output(Stage):
    def __init__(self, name, inputNames, outputDim=0, defaultValue=0):
        Stage.__init__(self,
            name=name, 
            inputNames=inputNames,
            defaultValue=defaultValue,
            outputDim=outputDim)
    def graphForward(self):
        self.Y = self.getInput()
    def graphBackward(self):
        self.sendError(self.dEdY)

class Container(Stage):
    def __init__(self,
                 stages,
                 outputStageNames,
                 inputDim,
                 outputDim,
                 inputNames,
                 name=None,
                 outputdEdX=True):
        Stage.__init__(self,
                name=name, 
                inputNames=inputNames,
                outputDim=outputDim, 
                outputdEdX=outputdEdX)
        self.stages = []
        self.stageDict = {}
        self.inputDim = inputDim
        self.outputStageNames = outputStageNames

        inputStage = self.createInputStage()
        self.stages.append(inputStage)
        self.stageDict['input'] = inputStage

        for stage in stages:
            self.register(stage)
        
        outputStage = self.createOutputStage()
        self.stages.append(outputStage)
        self.stageDict['output'] = outputStage

        self.link()
        self.dEdW = []
        for stage in self.stages:
            self.dEdW.append(0.0)

    def createInputStage(self):
        return Input(name='input', outputDim=self.inputDim)

    def createOutputStage(self):
        return Output(name='output', inputNames=self.outputStageNames)

    def register(self, stage):
        """
        Register a substage
        :param stage: new recurrent substage
        :return:
        """
        #print stage
        if not hasattr(stage, 'used'):
            stage.used = False
        self.stages.append(stage)
        self.stageDict[stage.name] = stage

    def link(self):
        """
        Link substages with their input strings
        :return:
        """
        for stage in self.stages:
            for stageName in stage.inputNames:
                stageInput = self.stageDict[stageName]
                stageInput.used = True
                stage.addInput(stageInput)

    def clearError(self):
        for stage in self.stages:
            stage.clearError()
        self.dEdY = 0.0
        self.receivedError = False

    def graphForward(self, dropout=True):
        self.X = self.getInput()
        self.Y = self.forward(self.X, dropout=dropout)

    #@profile
    def forward(self, X, dropout=True):
        self.stages[0].Y = X
        for s in range(1, len(self.stages) - 1):
            if self.stages[s].used:
                if hasattr(self.stages[s], 'dropout'):
                    self.stages[s].dropout = dropout
                    self.stages[s].graphForward()
                elif isinstance(self.stages[s], Container):
                    self.stages[s].graphForward(dropout=dropout)
                else:
                    self.stages[s].graphForward()
        self.stages[-1].graphForward()
        Y = self.stages[-1].Y

        # Clear error and ready for next batch
        self.clearError()

        self.X = X
        return Y

    #@profile
    def backward(self, dEdY):
        self.stages[-1].sendError(dEdY)
        for s in reversed(range(1, len(self.stages) - 1)):
            #print 'container backward', self.stages[s].name, self.stages[s].used, self.stages[s].receivedError
            if self.stages[s].used and self.stages[s].receivedError:
                self.stages[s].graphBackward()

        # Collect input error
        if self.outputdEdX:
            dEdX = self.stages[0].dEdY

        return dEdX if self.outputdEdX else None

    def updateWeights(self):
        for s in range(1, len(self.stages)-1):
            # Because all stages are "shallow copied", the weights are shared.
            self.stages[s].updateWeights()

    def updateLearningParams(self, numEpoch):
        for s in range(1, len(self.stages)-1):
            # Since only the first stage updates the weights,
            # learning params just need to update in the first stage.
            self.stages[s].updateLearningParams(numEpoch)

    def setGradient(self, value):
        if type(value) is float:
            for s in range(1, len(self.stages) - 1):
                self.stages[s].setGradient(value)
        elif type(value) is np.ndarray:
            for s in range(1, len(self.stages) - 1):
                self.stages[s].setGradient(value[s - 1])
        else:
            raise Exception('Unknown type %s for setGradient' % type(value))

    def getWeights(self):
        weights = []
        for s in range(1, len(self.stages)-1):
            if self.stages[s].gpu:
                weights.append(gpu.as_numpy_array(self.stages[s].getWeights()))
            else:
                weights.append(self.stages[s].getWeights())
        return np.array(weights, dtype=object)

    def loadWeights(self, W):
        for s in range(1, len(self.stages) - 1):
            self.stages[s].loadWeights(W[s - 1])
