import numpy as np
import copy
import os
use_gpu = os.environ.get('GNUMPY_USE_GPU', 'yes') == 'yes'
verbose = os.environ.get('VERBOSE', 'no') == 'yes'
if use_gpu:
    import gnumpy as gpu

class Stage:
    def __init__(self,
                 name,
                 inputNames,
                 outputDim,
                 defaultValue=0.0,
                 learningRate=0.0,
                 learningRateAnnealConst=0.0,
                 momentum=0.0,
                 deltaMomentum=0.0,
                 weightClip=0.0,
                 gradientClip=0.0,
                 weightRegConst=0.0,
                 gpu=False,
                 outputdEdX=True):
        self.name = name
        self.inputNames = inputNames
        self.inputs = None
        self.outputDim = outputDim
        self.defaultValue = np.zeros(outputDim) + defaultValue
        self.startLearningRate = learningRate
        self.learningRate = learningRate
        self.learningRateAnnealConst = learningRateAnnealConst
        self.momentum = momentum
        self.deltaMomentum = deltaMomentum
        self.weightClip = weightClip
        self.gradientClip = gradientClip
        self.weightRegConst = weightRegConst
        self.outputdEdX=outputdEdX
        self.dEdWnorm = 0.0
        self.Wnorm = 0.0
        self.dEdW = 0.0
        self.lastdW = 0.0
        self.W = 0.0
        self.Y = 0.0
        self.X = 0.0
        self.dEdY = 0.0
        self.gpu = gpu
        self.splX = None
        self.receivedError = False
    def __str__(self):
        return self.name

    def addInput(self, stage):
        if self.inputs is None:
            self.inputs = [stage]
        else:
            self.inputs.append(stage)

    def getInput(self):
        """
        Fetches input from each input stage.
        Concatenates input into one vector.
        """
        #print self.name
        if len(self.inputs) > 1:
            self.splX = []
            for stage in self.inputs:
                X = stage.Y
                self.splX.append(X)
                #print self.name, 'get input', stage.name, X.dtype
                #print '>', stage.name, X.shape
            return np.concatenate(self.splX, axis=-1)
        else:
            #print self.name,'get input', self.inputs[0].Y.dtype
            return self.inputs[0].Y

    def clearError(self):
        self.dEdY = 0.0
        self.receivedError = False

    def sendError(self, dEdX):
        """
        Iterates over input list and sends dEdX.
        """
        if len(self.inputs) > 1:
            s = 0
            for stage in self.inputs:
                s2 = s + stage.Y.shape[-1]
                stage.dEdY += dEdX[:, s : s2]
                s = s2
                stage.receivedError = True
        else:
            #if type(self.inputs[0].dEdY) == np.ndarray:
            #    print self.name, self.inputs[0].name, self.inputs[0].dEdY.shape, dEdX.shape
            self.inputs[0].dEdY += dEdX
            self.inputs[0].receivedError = True
            #print self.name, 'send error', self.inputs[0].name

    def getValue(self):
        """
        Gets the output value.
        """
        return self.Y

    def getGradient(self):
        """
        Gets the gradient with regard to the weights.
        """
        return self.dEdW

    def setGradient(self, value):
        """
        Sets the gradient with regard to the weights.
        :param value: float or numpy array
        :return:
        """
        self.dEdW = value

    def graphForward(self):
        """
        Forward propagates.
        """
        self.X = self.getInput()
        if verbose and hasattr(self.X, 'shape'):
            print 'forward in', self.name, self.X.shape
        self.Y = self.forward(self.X)
        if verbose and hasattr(self.Y, 'shape'):
            print 'forward out', self.name, self.Y.shape

    def forward(self, X):
        """
        Abstract method. Forward pass input to the stage.
        :param X: The input. At least two dimensional numpy array.
        The first dimension is always the number of examples.
        :return: The output of the stage.
        """
        return

    def graphBackward(self):
        """
        Backward propagates.
        """
        if verbose and hasattr(self.dEdY, 'shape'):
            print 'backward in', self.name, self.dEdY.shape, np.mean(self.dEdY)
        dEdX = self.backward(self.dEdY)
        if self.outputdEdX:
            self.sendError(dEdX)
        if verbose and hasattr(dEdX, 'shape'):
            print 'backward out', self.name, dEdX.shape, np.mean(dEdX)

    def backward(self, dEdY):
        """
        Abstract method. Backward propagate error in the stage.
        :param dEdY: The error of the output.
        :return: The error of the input.
        """
        return

    def updateWeights(self):
        self._updateWeights(self.dEdW)

    def _updateWeights(self, dEdW):
        if self.gpu:
            if self.gradientClip > 0.0:
                self.dEdWnorm = gpu.sqrt(gpu.sum(dEdW ** 2))
                if self.dEdWnorm > self.gradientClip:
                    dEdW *= self.gradientClip / self.dEdWnorm
            if self.learningRate > 0.0:
                self.lastdW = -self.learningRate * dEdW + \
                           self.momentum * self.lastdW
                self.W += self.lastdW
            if self.weightRegConst > 0.0:
                a = self.learningRate * self.weightRegConst
                self.W -= a * self.W
            if self.weightClip > 0.0:
                self.Wnorm = gpu.sqrt(gpu.sum(self.W ** 2))
                if self.Wnorm > self.weightClip:
                    self.W *= self.weightClip / self.Wnorm
        else:
            if self.gradientClip > 0.0:
                self.dEdWnorm = np.sqrt(np.sum(np.power(dEdW, 2)))
                if self.dEdWnorm > self.gradientClip:
                    dEdW *= self.gradientClip / self.dEdWnorm
            if self.learningRate > 0.0:
                self.lastdW = -self.learningRate * dEdW + \
                           self.momentum * self.lastdW
                self.W += self.lastdW
            if self.weightRegConst > 0.0:
                a = self.learningRate * self.weightRegConst
                self.W -= a * self.W
            if self.weightClip > 0.0:
                self.Wnorm = np.sqrt(np.sum(np.power(self.W, 2)))
                if self.Wnorm > self.weightClip:
                    self.W *= self.weightClip / self.Wnorm

    def updateLearningParams(self, numEpoch):
        self.learningRate = self.startLearningRate / \
                                   (1.0 + self.learningRateAnnealConst * numEpoch)
        self.momentum -= self.deltaMomentum

        if self.gradientClip > 0.0 or self.weightClip > 0.0:
            print 'ST: %11s ' % self.name,
            if self.gradientClip > 0.0:
                print 'GN: %8.4f ' % self.dEdWnorm,
                print 'GC: %8.4f ' % self.gradientClip,
            if self.weightClip > 0.0:
                print 'WN: %8.4f ' % self.Wnorm,
                print 'WC: %8.4f ' % self.weightClip,
            print

    def getWeights(self):
        if self.gpu:
            return gpu.as_numpy_array(self.W)
        else:
            return self.W

    def loadWeights(self, W):
        if self.gpu:
            self.W = gpu.as_garray(W)
        else:
            self.W = W

    def copy(self):
        return copy.copy(self)