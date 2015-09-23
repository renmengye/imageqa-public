from stage import *
import os
use_gpu = os.environ.get('GNUMPY_USE_GPU', 'yes') == 'yes'

class LUT(Stage):
    """
    Look-up table.
    WARNING: this implementation of LUT is index 1-based.
    0 will mean an all-zero entry.
    The first row of the weight matrix is one.
    """
    def __init__(self,
                 inputNames,
                 inputDim,
                 outputDim,
                 lazyInit=True,
                 initRange=1.0,
                 initSeed=2,
                 intConversion=False,
                 needInit=True,
                 initWeights=0,
                 sparse=False,
                 learningRate=0.0,
                 learningRateAnnealConst=0.0,
                 momentum=0.0,
                 deltaMomentum=0.0,
                 weightClip=0.0,
                 gradientClip=0.0,
                 weightRegConst=0.0,
                 outputdEdX=False,
                 name=None):
        Stage.__init__(self,
                 name=name,
                 inputNames=inputNames,
                 learningRate=learningRate,
                 outputDim=outputDim,
                 learningRateAnnealConst=learningRateAnnealConst,
                 momentum=momentum,
                 deltaMomentum=deltaMomentum,
                 weightClip=weightClip,
                 gradientClip=gradientClip,
                 weightRegConst=weightRegConst,
                 gpu=False,
                 outputdEdX=outputdEdX)
        self.outputDim = outputDim
        self.inputDim = inputDim
        self.initRange = initRange
        self.random = np.random.RandomState(initSeed)
        self.needInit = needInit
        self.intConversion = intConversion

        # Zeroth rows of the weight matrix is reserved
        # for empty word at the end of a sentence.
        if needInit:
            if lazyInit:
                self.W = None
            else:
                self.initWeights()
        else:
            self.W = initWeights
            if use_gpu and self.W.dtype != np.float32:
                self.W = self.W.astype('float32')
        self.X = 0
        self.Y = 0
        self.sparse = sparse
        self.dEdW = 0.0

    def initWeights(self):
        # print self.name
        self.W = self.random.uniform(
            -self.initRange/2.0, self.initRange/2.0,
            (self.inputDim, self.outputDim))
        if use_gpu and self.W.dtype != np.float32:
            self.W = self.W.astype('float32')

    def forward(self, X):
        if self.W is None: self.initWeights()
        if self.intConversion: X = X.astype(int)
        self.X = X
        X = X.reshape(X.size)
        Y = np.zeros((X.shape[0], self.outputDim), self.W.dtype)
        for n in range(0, X.shape[0]):
            if self.sparse:
                if X[n] != 0:
                    Y[n] = self.W[X[n] - 1].todense()
            else:
                if X[n] != 0:
                    Y[n] = self.W[X[n] - 1]
        return Y

    def backward(self, dEdY):
        X = self.X
        if self.learningRate > 0.0:
            self.dEdW = np.zeros(self.W.shape, self.W.dtype)
            for n in range(0, X.shape[0]):
                self.dEdW[X[n] - 1] += dEdY[n]
        if self.outputdEdX:
            return np.zeros(X.shape)
        else:
            return None

    def loadWeights(self, W):
        if self.learningRate == 0.0:
            return
        else:
            Stage.loadWeights(self, W)

    def getWeights(self):
        if self.learningRate == 0.0:
            return 0
        else:
            return self.W
