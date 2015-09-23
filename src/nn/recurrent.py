import os
verbose = os.environ.get('VERBOSE', 'no') == 'yes'
from container import *

class RecurrentStage:
    def __init__(self):
        pass
    def initTime(self, timespan):
        """
        Initialize stages accross timespan.
        :param timespan:
        :return:
        """
        pass
    def timeForward(self, time):
        """
        Forward one timestep. Need to implement in child classes.
        :param time: integer
        :return:
        """
        pass
    def timeBackward(self, time):
        """
        Backward one timestep. Need to implement in child classes.
        :param time: integer
        :return:
        """
        pass
    def getStage(self, time):
        """
        Get the stage at a timestep. Need to implement in child classes.
        :param time: integer
        :return: Stage object
        """
        pass

class AttentionPenalty(RecurrentStage):
    def __init__(self, name, inputNames, errorConst=1.0):
        self.inputs = []
        self.name = name
        self.errorConst = errorConst
        self.inputNames = inputNames
        self.used = True
        self.X = 0.0
        self.dEdX = 0.0
        self.timespan = 0
        self.gpu = False
        
    def initTime(self, timespan):
        self.timespan = timespan

    def addInput(self, stage):
        self.inputs.append(stage)

    def clearError(self):
        pass

    def syncWeights(self):
        pass

    def syncGradient(self):
        pass

    def updateWeights(self):
        pass

    def updateLearningParams(self, epoch):
        pass

    def getWeights(self):
        return 0

    def loadWeights(self, W):
        pass

    def getGradient(self):
        return 0

    def timeForward(self, time):
        X = self.inputs[time].Y
        if time == 0:
            self.X = np.zeros((X.shape[0], self.timespan, X.shape[1]))
            self.dEdX = 0.0
        self.X[:, time, :] = X
        self.receivedError = True

    def timeBackward(self, time):
        if type(self.dEdX) != np.ndarray:
            s = np.sum(self.X, axis=1)
            one = np.ones((self.X.shape[0], self.X.shape[2])) * self.timespan / float(self.X.shape[2])
            self.dEdX = self.errorConst * (s - one)
        self.inputs[time].dEdY += self.dEdX
        if verbose:
            print 'backward out', self.name, self.dEdX.shape, np.mean(self.dEdX)
        
    def getStage(self, time):
        return self

class RecurrentAdapter(Stage, RecurrentStage):
    """
    Convert a standard stage into a recurrent stage.
    """
    def __init__(self, stage):
        Stage.__init__(self,
                            name=stage.name,
                            inputNames=stage.inputNames,
                            outputDim=stage.outputDim)
        stage.name = stage.name + '-0'
        self.timespan = 0
        self.stages = [stage]

    def getStage(self, time):
        return self.stages[time]

    def initTime(self, timespan):
        self.timespan = timespan
        for s in range(1, timespan):
            stage2 = self.stages[0].copy()
            stage2.name = self.stages[0].name[:-2] + ('-%d' % s)
            self.stages.append(stage2)

    def clearError(self):
        for stage in self.stages:
            stage.clearError()

    def timeForward(self, time):
        """
        Forward one timestep.
        :param time: integer
        :return:
        """
        self.stages[time].graphForward()

    def timeBackward(self, time):
        """
        Backward one timestep
        :param time: integer
        :return:
        """
        self.stages[time].graphBackward()

    def syncGradient(self):
        # Sum error through time
        W = self.stages[0].getWeights()
        if self.stages[0].gpu and self.stages[0].learningRate > 0.0:
            tmp = gpu.zeros((self.timespan, W.shape[0], W.shape[1]))
            for t in range(self.timespan):
                tmp[t] = self.stages[t].getGradient()
                self.stages[t].setGradient(0.0)
            self.stages[0].setGradient(gpu.sum(tmp, axis=0))
        elif type(W) is np.ndarray and self.stages[0].learningRate > 0.0:
            tmp = np.zeros((self.timespan, W.shape[0], W.shape[1]))
            for t in range(self.timespan):
                tmp[t] = self.stages[t].getGradient()
                self.stages[t].setGradient(0.0)
            self.stages[0].setGradient(np.sum(tmp, axis=0))

    def syncWeights(self):
        for t in range(1, self.timespan):
            self.stages[t].loadWeights(self.stages[0].getWeights())

    def updateWeights(self):
        if self.stages[0].learningRate > 0.0:
            self.stages[0].updateWeights()
            self.syncWeights()

    def updateLearningParams(self, numEpoch):
        # Since only the first stage updates the weights,
        # learning params just need to update in the first stage.
        self.stages[0].updateLearningParams(numEpoch)

    def getWeights(self):
        return self.stages[0].getWeights()

    def loadWeights(self, W):
        for t in range(0, self.timespan):
            self.stages[t].loadWeights(W)

    def getGradient(self):
        return self.stages[0].getGradient()

    def getValue(self):
        Y = np.zeros((self.stages[0].Y.shape[0],
                    self.timespan,
                    self.stages[0].Y.shape[1]))
        for t, stage in enumerate(self.stages):
            Y[:, t] = stage.getValue()
        return Y

class Constant(Stage):
    def __init__(self,
        name,
        value):
        Stage.__init__(self,
            name=name,
            inputNames=[],
            outputDim=value.size)
        self.value = np.reshape(value, (1, value.size))
    def forward(self, X):
        return np.tile(self.value, (X.shape[0], 1))
    def graphForward(self):
        self.Y = self.forward(self.X)

class RecurrentContainer(Container, RecurrentStage):
    def __init__(self,
                 stages,
                 outputStageNames,
                 inputDim,
                 outputDim,
                 timespan,
                 defaultValue=0,
                 multiInput=True,
                 multiOutput=True,
                 cutOffZeroEnd=True,
                 name=None,
                 inputNames=None,
                 outputdEdX=True):
        self.multiInput = multiInput
        self.multiOutput = multiOutput
        self.cutOffZeroEnd = cutOffZeroEnd
        self.timespan = timespan
        self.Xend = 0
        self.XendAll = 0
        self.constStages = []
        self.init = False
        Container.__init__(self,
                 stages=stages,
                 outputStageNames=outputStageNames,
                 inputDim=inputDim,
                 outputDim=outputDim,
                 name=name,
                 inputNames=inputNames,
                 outputdEdX=outputdEdX)

    def initTime(self, timespan):
        for s in self.stages:
            s.initTime(timespan)

    def getStage(self, time):
        return self.stages[-1].getStage(time)

    def createInputStage(self):
        return RecurrentAdapter(
            stage=Container.createInputStage(self))

    def createOutputStage(self):
        return RecurrentAdapter(
            stage=Output(name='output', 
                inputNames=self.outputStageNames, 
                outputDim=self.outputDim, 
                defaultValue=self.defaultValue))

    def link(self):
        """
        Links the stage with time references.
        :return:
        """
        self.initTime(self.timespan)
        for t in range(self.timespan):
            self.stages[-1].getStage(t).used = True
            for s in range(1, len(self.stages) - 1):
                if not hasattr(self.stages[s].getStage(t), 'used'):
                    self.stages[s].getStage(t).used = False
        for t in range(self.timespan):
            for stage in self.stages:
                names = stage.inputNames
                for inputStageStr in names:
                    if '(' in inputStageStr:
                        stageName = inputStageStr[:inputStageStr.index('(')]
                        stageTimeStr = \
                            inputStageStr[inputStageStr.index('(') + 1 : inputStageStr.index(')')]
                        if stageTimeStr[0] == '$':
                            stageTime = int(stageTimeStr[1:]) - t
                        else:
                            stageTime = int(stageTimeStr)
                    else:
                        stageName = inputStageStr
                        stageTime = 0
                    if stageTime > 0:
                        raise Exception('Recurrent model definition is non-causal.')
                    if t + stageTime < 0:
                        stageInputContainer = self.stageDict[stageName]
                        stageInput = Constant(
                            name=('%s-%s-%d'%('const', stageName, -t-stageTime)),
                            value=stageInputContainer.getStage(0).defaultValue)
                        self.constStages.append(stageInput)
                    else:
                        stageInput = self.stageDict[stageName].getStage(t + stageTime)
                        stageInput.used = True
                    if isinstance(stage, RecurrentContainer):
                        stage.stages[0].getStage(time=t).addInput(stageInput)
                    else:
                        stage.getStage(time=t).addInput(stageInput)

    def clearError(self):
        for stage in self.stages:
            stage.clearError()
        for stage in self.constStages:
            stage.clearError()
        self.dEdY = 0.0
        self.receivedError = False

    def syncGradient(self):
        for stage in self.stages:
            stage.syncGradient()

    def syncWeights(self):
        for stage in self.stages:
            stage.syncWeights()

    def timeForward(self, time, dropout=True):
        """
        Forward one step (used as a recurrent stage/container).
        :param time: integer
        :return:
        """
        self.stages[0].getStage(time=time).graphForward()
        # If time=0 need to set const stages as well!!
        if time == 0:
            X = self.stages[0].getStage(time=time).Y
            for s in self.constStages:
                s.Y = np.tile(s.value, (X.shape[0], 1))
        for s in range(0, len(self.stages)):
            if self.stages[s].getStage(time=time).used:
                if hasattr(self.stages[s], 'dropout'):
                    self.stages[s].getStage(time=time).dropout = dropout
                elif isinstance(self.stages[s], RecurrentContainer):
                    self.stages[s].timeForward(time=time, dropout=dropout)
                else:
                    self.stages[s].timeForward(time=time)
        self.Y = self.stages[-1].Y
        return self.Y

    #@profile
    def forward(self, X, dropout=True):
        """
        Forward an entire sequence (used as a standard container).
        :param X: integer
        :param dropout: whether or not dropout
        :return:
        """
        # Sync weights if new.
        if not self.init:
            self.init = True
            XX = np.ones(X.shape, dtype=X.dtype)
            self.forward(XX)
            self.syncWeights()

        N = X.shape[0]
        self.Xend = np.zeros(N, dtype=int) + self.timespan
        reachedEnd = np.sum(X, axis=-1) == 0.0
        if self.multiOutput:
            Y = np.zeros((N, self.timespan, self.outputDim))
        else:
            Y = np.zeros((N, self.outputDim))

        # Scan for the end of the sequence.
        if self.cutOffZeroEnd:
            for n in range(N):
                for t in range(X.shape[1]):
                    if reachedEnd[n, t]:
                        self.Xend[n] = t
                        break
        self.XendAll = np.max(self.Xend)

        # Set value for constant stages.
        for s in self.constStages:
            s.Y = np.tile(s.value, (X.shape[0], 1))

        # Propagating through time.
        for t in range(self.XendAll):
            if self.multiInput:
                self.stages[0].getStage(time=t).Y = X[:, t, :]
            else:
                if t == 0:
                    self.stages[0].getStage(time=t).Y = X
                else:
                    self.stages[0].getStage(time=t).Y = np.zeros(X.shape)
            for s in range(1, len(self.stages)):
                if self.stages[s].getStage(time=t).used:
                    if hasattr(self.stages[s], 'dropout'):
                        self.stages[s].getStage(time=t).dropout = dropout
                    elif isinstance(self.stages[s], RecurrentContainer):
                        self.stages[s].timeForward(time=t, dropout=dropout)
                    else:
                        self.stages[s].timeForward(time=t)

        # Gather output.
        if self.multiOutput:
            for n in range(N):
                if self.Xend[n] > 0:
                    for t in range(self.Xend[n]):
                        Y[n, t, :] = self.stages[-1].getStage(time=t).Y[n]
        else:
            for n in range(N):
                if self.Xend[n] > 0:
                    Y[n, :] = self.stages[-1].getStage(time=self.Xend[n] - 1).Y[n]

        # Clear error for backward
        self.clearError()
        self.Y = Y
        self.X = X
        return Y

    def timeBackward(self, time):
        """
        Backward one time step (used as a recurrent stage/container).
        :param time: integer
        :return:
        """
        outputStage = self.stages[-1].getStage(time=time)
        outputStage.graphBackward()
        for s in reversed(range(0, len(self.stages) - 1)):
            if self.stages[s].getStage(time=time).used:
                self.stages[s].timeBackward(time=time)

    def sendError(self, dEdX):
        """
        Iterates over input list and sends dEdX.
        """
        if len(self.inputs) > 1:
            s = 0
            for stage in self.inputs:
                s2 = s + stage.Y.shape[-1]
                stage.dEdY += dEdX[:, :, s : s2]
                s = s2
                stage.receivedError = True
        else:
            self.inputs[0].dEdY += dEdX
            self.inputs[0].receivedError = True

    #@profile
    def backward(self, dEdY):
        """
        Backward an entire sequence (used as a standard container).
        :param dEdY: numpy array, error from the output
        :return:
        """
        N = self.X.shape[0]
        dEdX = np.zeros(self.X.shape)

        # Send errors from output stages.
        if self.multiOutput:
            for t in range(self.XendAll):
                self.stages[-1].getStage(time=t).sendError(dEdY[:, t, :])
        else:
            for n in range(N):
                if self.Xend[n] > 0:
                    err = np.zeros(dEdY.shape)
                    err[n] = dEdY[n]
                    self.stages[-1].getStage(time=self.Xend[n] - 1).sendError(err)

        # Back propagating through time.
        for t in reversed(range(self.XendAll)):
            for s in reversed(range(1, len(self.stages) - 1)):
                if self.stages[s].getStage(time=t).used:
                    self.stages[s].timeBackward(time=t)

        # Collect input error
        if self.outputdEdX:
            if self.multiInput:
                for t in range(self.XendAll):
                    dEdX[:, t, :] = self.stages[0].getStage(time=t).dEdY
            else:
                dEdX = self.stages[0].getStage(time=0).dEdY
        
        self.syncGradient()
        return dEdX if self.outputdEdX else None